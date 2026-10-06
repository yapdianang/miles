"""CPU checks of patched SGLang: DFlash returns the support each committed token was drawn from.

Run by test_patch_sglang_dflash_sampling_mask.py in a subprocess whose ``sglang`` is the patched
copy. The CUDA verify kernel is replaced by a line-for-line Python port of
``tree_speculative_sampling_target_only`` for chains; flashinfer's renorm kernels fall back to torch.
dflash_worker_v2 needs CUDA to import, so its patched ``_accept_block`` and helpers are compiled
from the patched source.
"""

import ast
import contextlib
import logging
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

from sglang.srt.layers.logits_processor import LogitsProcessorOutput, SamplingMaskStatus
from sglang.srt.layers.sampler import Sampler
from sglang.srt.managers.scheduler_components.batch_result_processor import SchedulerBatchResultProcessor
from sglang.srt.speculative import dflash_utils
from sglang.srt.speculative.dflash_utils import (
    DFlashSupportCapture,
    build_dflash_sampling_mask_output,
    build_dflash_verify_target_probs,
    compute_dflash_correct_drafts_and_bonus,
    compute_dflash_sampling_correct_drafts_and_bonus,
    dflash_sampling_mask_unsupported_reason,
)

BLOCK, VOCAB = 4, 24


def reference_target_only(
    *,
    predicts,
    accept_index,
    accept_token_num,
    candidates,
    retrive_index,
    retrive_next_token,
    retrive_next_sibling,
    uniform_samples,
    uniform_samples_for_final_sampling,
    target_probs,
    draft_probs,
    threshold_single,
    threshold_acc,
    deterministic,
):
    """sgl_kernel/speculative/sampling.cuh TreeSpeculativeSamplingTargetOnly, one block per row."""
    bs, num_draft = candidates.shape
    num_spec = accept_index.shape[1]
    target = target_probs.view(bs, num_draft, -1)
    draft = draft_probs.view(bs, num_draft, -1)
    for b in range(bs):
        prob_acc, row, cur, accepted = 0.0, 0, 0, 0
        coin = float(uniform_samples[b, 0])
        last = int(retrive_index[b, 0])
        accept_index[b, 0] = last
        for _ in range(1, num_spec):
            cur = int(retrive_next_token[b, cur])
            while cur != -1:
                token = int(candidates[b, cur])
                p = float(target[b, row, token])
                prob_acc += p
                if p > 0 and (coin < prob_acc / threshold_acc or p >= threshold_single):
                    prob_acc, row, coin = 0.0, cur, float(uniform_samples[b, cur])
                    predicts[last] = token
                    accepted += 1
                    accept_index[b, accepted] = int(retrive_index[b, cur])
                    last = int(retrive_index[b, cur])
                    break
                draft[b, row, token] = target[b, row, token]
                cur = int(retrive_next_sibling[b, cur])
            if cur == -1:
                break
        accept_token_num[b] = accepted
        residual = target[b, row] - (draft[b, row] if accepted != num_spec - 1 else 0)
        residual = residual.clamp_min(0)
        u = float(uniform_samples_for_final_sampling[b]) * float(residual.sum())
        hits = torch.nonzero((torch.cumsum(residual, 0) > u) & (residual > 0)).flatten()
        predicts[last] = int(hits[0]) if len(hits) else int(torch.nonzero(residual > 0).flatten()[-1])


THRESHOLDS = {"single": 1.0, "acc": 1.0}
dflash_utils.get_spec = lambda: SimpleNamespace(
    speculative_accept_threshold_single=THRESHOLDS["single"], speculative_accept_threshold_acc=THRESHOLDS["acc"]
)
dflash_utils.get_parallel = lambda: SimpleNamespace(pp_size=1)
dflash_utils.tree_speculative_sampling_target_only = reference_target_only
dflash_utils._DFLASH_SAMPLING_VERIFY_AVAILABLE = True
dflash_utils.borrow_graph_pool = lambda user: contextlib.nullcontext()


def _worker_functions() -> dict:
    """The patched worker's ``_accept_block`` and module helpers, compiled outside the CUDA-only module."""
    tree = ast.parse((Path(dflash_utils.__file__).parent / "dflash_worker_v2.py").read_text())
    wanted = {"_commit_accept", "_is_all_greedy", "_accept_block"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    worker = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DFlashWorkerV2")
    nodes += [node for node in worker.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    namespace = {
        "torch": torch,
        "logger": logging.getLogger("dflash"),
        "SpecTpSyncSite": SimpleNamespace(DFLASH_SELECTOR=0, DFLASH_ACCEPT_SAMPLE=1, DFLASH_ACCEPT_GREEDY=2),
        "is_dflash_sampling_verify_available": lambda: True,
        "compute_dflash_sampling_correct_drafts_and_bonus": compute_dflash_sampling_correct_drafts_and_bonus,
        "compute_dflash_correct_drafts_and_bonus": compute_dflash_correct_drafts_and_bonus,
        "DFlashSupportCapture": DFlashSupportCapture,
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "dflash_worker_v2.py", "exec"), namespace)
    return namespace


WORKER = _worker_functions()
_commit_accept = WORKER["_commit_accept"]


def sampler(max_tokens: int = VOCAB) -> Sampler:
    instance = Sampler.__new__(Sampler)
    torch.nn.Module.__init__(instance)
    instance.sampling_mask_max_tokens = max_tokens
    instance.tp_sync_group = None
    instance.cp_sync_group = None
    return instance


def sampling_info(top_ks, top_ps, mask_rows, support_rows=None, greedy=False):
    bs = len(top_ks)
    return SimpleNamespace(
        temperatures=torch.ones(bs, 1),
        top_ks=torch.tensor(top_ks),
        top_ps=torch.tensor(top_ps, dtype=torch.float32),
        need_top_k_sampling=any(k < VOCAB for k in top_ks),
        need_top_p_sampling=any(p < 1 for p in top_ps),
        is_all_greedy=greedy,
        sampling_mask_batch_indices=torch.tensor(mask_rows),
        sampling_support_logprobs_capture_indices=None if support_rows is None else torch.tensor(support_rows),
    )


def joint_reference(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    """The plain sampler's joint filter: top-p cutoff of the full distribution intersected with top-k."""
    probs = torch.softmax(logits, -1)
    order = torch.argsort(probs, descending=True)
    sorted_probs = probs[order]
    keep_sorted = (torch.arange(len(probs)) < top_k) & (torch.cumsum(sorted_probs, 0) - sorted_probs <= top_p)
    keep = torch.zeros_like(probs, dtype=torch.bool)
    keep[order[keep_sorted]] = True
    filtered = torch.where(keep, probs, 0.0)
    return filtered / filtered.sum()


def verify(info, logits, candidates, *, max_tokens=VOCAB, coins=None):
    """Patched verify + capture + host materialization; returns targets, commits and per-request rows."""
    bs = candidates.shape[0]
    capture = DFlashSupportCapture.for_batch(info, logits, BLOCK, sampled=True)
    kwargs = {} if coins is None else {"uniform_samples": coins, "uniform_samples_for_final_sampling": torch.rand(bs)}
    accept_len, bonus = compute_dflash_sampling_correct_drafts_and_bonus(
        candidates=candidates,
        next_token_logits=logits,
        sampling_info=info,
        max_top_k=max(int(k) for k in info.top_ks),
        threshold_single=1.0,
        threshold_acc=1.0,
        support_capture=capture,
        **kwargs,
    )
    out_tokens, commit_lens = _commit_accept(candidates, accept_len, bonus)
    target = build_dflash_verify_target_probs(
        next_token_logits=logits, sampling_info=info, draft_token_num=BLOCK, bs=bs, max_top_k=max(info.top_ks.tolist())
    )
    output = build_dflash_sampling_mask_output(sampler(max_tokens), info, capture, out_tokens, commit_lens)
    return target, capture, out_tokens, commit_lens, materialize(info, output, bs)


def materialize(info, output, bs):
    rows = set(info.sampling_mask_batch_indices.tolist())
    support = (
        set()
        if info.sampling_support_logprobs_capture_indices is None
        else set(info.sampling_mask_batch_indices[info.sampling_support_logprobs_capture_indices].tolist())
    )
    reqs = [
        SimpleNamespace(
            return_sampling_mask=b in rows, sampling_logprobs_mode="support" if b in support else "selected"
        )
        for b in range(bs)
    ]
    logits_output = LogitsProcessorOutput(next_token_logits=None)
    logits_output.sampling_mask_output = output
    SchedulerBatchResultProcessor.materialize_sampling_mask_output(reqs, logits_output)
    return logits_output


def check_joint_filter_order():
    logits = torch.log(torch.tensor([[0.5, 0.3, 0.2] + [1e-12] * (VOCAB - 3)]))
    info = sampling_info([2], [0.55], [0])
    probs = build_dflash_verify_target_probs(next_token_logits=logits, sampling_info=info, draft_token_num=1, bs=1)
    torch.testing.assert_close(probs.flatten()[:3], torch.tensor([0.625, 0.375, 0.0]))
    torch.manual_seed(0)
    logits = torch.randn(3 * BLOCK, VOCAB) * 2
    info = sampling_info([5, 12, VOCAB], [0.7, 0.9, 0.8], [0])
    probs = build_dflash_verify_target_probs(next_token_logits=logits, sampling_info=info, draft_token_num=BLOCK, bs=3)
    for row in range(3 * BLOCK):
        b = row // BLOCK
        expected = joint_reference(logits[row], int(info.top_ks[b]), float(info.top_ps[b]))
        torch.testing.assert_close(probs.view(-1, VOCAB)[row], expected)


def check_packed_sets_match_the_target_distribution():
    torch.manual_seed(1)
    cases = {
        "top-k path": ([6, 9, 7], [1.0, 1.0, 1.0]),
        "dense joint path": ([6, 9, 7], [0.8, 0.95, 0.6]),
        "dense top-p path": ([VOCAB] * 3, [0.8, 0.95, 0.6]),
    }
    for name, (top_ks, top_ps) in cases.items():
        info = sampling_info(top_ks, top_ps, mask_rows=[0, 2], support_rows=[1])
        logits = torch.randn(3 * BLOCK, VOCAB) * 2
        candidates = torch.randint(0, VOCAB, (3, BLOCK))
        target, capture, out_tokens, commit_lens, output = verify(info, logits, candidates)
        torch.testing.assert_close(capture.probs, target[[0, 2]].float(), msg=name)
        for b, mode in ((0, "selected"), (2, "support")):
            masks, logprobs = output.next_token_sampling_mask_idx[b], output.next_token_sampling_logprobs[b]
            assert output.next_token_sampling_mask_status[b] == SamplingMaskStatus.OK, name
            assert len(masks) == int(commit_lens[b]), name
            for j, (mask, values) in enumerate(zip(masks, logprobs, strict=True)):
                row = target[b, j]
                assert set(mask.tolist()) == set(torch.nonzero(row).flatten().tolist()), (name, b, j)
                token = int(out_tokens[b, j])
                if mode == "selected":
                    assert math.isclose(float(values[0]), math.log(float(row[token])), rel_tol=1e-5), (name, b, j)
                else:
                    torch.testing.assert_close(torch.as_tensor(values), row[torch.as_tensor(mask).long()].log())
        assert output.next_token_sampling_mask_idx[1] is None


def check_rows_follow_every_accept_length():
    """Coins of zero accept a draft iff it has mass; drafts past ``accept_len`` fall outside the top-k."""
    for accept_len in range(BLOCK):
        logits = torch.zeros(BLOCK, VOCAB)
        candidates = torch.tensor([[0, 1, 2, 3]])
        for j in range(BLOCK - 1):
            logits[j, candidates[0, j + 1]] = 5.0 if j < accept_len else -50.0
        info = sampling_info([4], [1.0], mask_rows=[0])
        target, _, out_tokens, commit_lens, output = verify(info, logits, candidates, coins=torch.zeros(1, BLOCK))
        assert int(commit_lens[0]) == accept_len + 1, accept_len
        assert out_tokens[0, :accept_len].tolist() == candidates[0, 1 : accept_len + 1].tolist()
        masks = output.next_token_sampling_mask_idx[0]
        assert len(masks) == accept_len + 1
        for j, mask in enumerate(masks):
            assert set(mask.tolist()) == set(torch.nonzero(target[0, j]).flatten().tolist()), (accept_len, j)
            assert int(out_tokens[0, j]) in set(mask.tolist())


def check_invalid_and_overflow_only_count_committed_rows():
    info = sampling_info([3], [1.0], mask_rows=[0])
    logits = torch.zeros(BLOCK, VOCAB)
    logits[:, :3] = 4.0  # rows support {0, 1, 2}
    capture = DFlashSupportCapture.for_batch(info, logits, BLOCK, sampled=True)
    capture.probs.copy_(
        build_dflash_verify_target_probs(
            next_token_logits=logits, sampling_info=info, draft_token_num=BLOCK, bs=1, max_top_k=3
        )
    )
    statuses = {}
    for name, tokens, commit, max_tokens in (
        ("ok", [0, 1, 2, 9], 3, VOCAB),
        ("invalid", [0, 9, 2, 1], 3, VOCAB),
        ("invalid past commit", [0, 1, 9, 9], 2, VOCAB),
        ("overflow", [0, 1, 2, 0], 1, 2),
    ):
        output = build_dflash_sampling_mask_output(
            sampler(max_tokens), info, capture, torch.tensor([tokens]), torch.tensor([commit], dtype=torch.int32)
        )
        statuses[name] = materialize(info, output, 1).next_token_sampling_mask_status[0]
    assert statuses == {
        "ok": SamplingMaskStatus.OK,
        "invalid": SamplingMaskStatus.INVALID,
        "invalid past commit": SamplingMaskStatus.OK,
        "overflow": SamplingMaskStatus.OVERFLOW,
    }, statuses


def check_greedy_and_selector_blocks():
    info = sampling_info([1], [1.0], mask_rows=[0], greedy=True)
    logits = torch.randn(BLOCK, VOCAB)
    greedy = DFlashSupportCapture.for_batch(info, logits, BLOCK, sampled=False)
    tokens = torch.tensor([[5, 6, 7, 8]])
    commit = torch.tensor([3], dtype=torch.int32)
    output = materialize(info, build_dflash_sampling_mask_output(sampler(), info, greedy, tokens, commit), 1)
    assert [mask.tolist() for mask in output.next_token_sampling_mask_idx[0]] == [[5], [6], [7]]
    assert [float(value[0]) for value in output.next_token_sampling_logprobs[0]] == [0.0, 0.0, 0.0]
    selector = materialize(info, build_dflash_sampling_mask_output(sampler(), info, None, tokens, commit), 1)
    assert selector.next_token_sampling_mask_status[0] == SamplingMaskStatus.INVALID


def check_rows_stop_where_output_stops():
    rows = []
    req = SimpleNamespace(
        output_ids=[1, 2, 3, 4, 5],
        finished_len=4,
        sampling_mask_rows=SimpleNamespace(append=lambda *row: rows.append(row)),
    )
    output = SimpleNamespace(next_token_sampling_mask_idx=[["a", "b", "c"]], next_token_sampling_logprobs=[[1, 2, 3]])
    processor = SimpleNamespace(_visible_output_len=SchedulerBatchResultProcessor._visible_output_len)
    SchedulerBatchResultProcessor.add_sampling_mask_return_values(processor, 0, req, output, accept_len=3)
    assert rows == [("a", 1), ("b", 2)], rows  # the step committed outputs 3..5; output stops after 4


def check_accept_block_returns_the_capture():
    torch.manual_seed(2)
    worker = SimpleNamespace(
        _selector_sample=None,
        block_size=BLOCK,
        _tp_sync=SimpleNamespace(sync=lambda *args: None),
        _use_triton_accept_bonus=False,
    )
    info = sampling_info([6, 9], [0.9, 0.9], mask_rows=[1])
    logits = torch.randn(2 * BLOCK, VOCAB)
    result = WORKER["_accept_block"](
        worker,
        candidates=torch.randint(0, VOCAB, (2, BLOCK)),
        next_token_logits=logits,
        sampling_info=info,
        draft_input=SimpleNamespace(max_top_k=9, uniform_top_k_value=None),
        prefix_lens=torch.zeros(2, dtype=torch.int64),
        bs=2,
    )
    capture = result[-1]
    target = build_dflash_verify_target_probs(
        next_token_logits=logits, sampling_info=info, draft_token_num=BLOCK, bs=2, max_top_k=9
    )
    torch.testing.assert_close(capture.probs, target[[1]].float())
    greedy_info = sampling_info([1, 1], [1.0, 1.0], mask_rows=[0], greedy=True)
    greedy = WORKER["_accept_block"](
        worker,
        candidates=torch.randint(0, VOCAB, (2, BLOCK)),
        next_token_logits=logits,
        sampling_info=greedy_info,
        draft_input=SimpleNamespace(max_top_k=1, uniform_top_k_value=None),
        prefix_lens=torch.zeros(2, dtype=torch.int64),
        bs=2,
    )[-1]
    assert greedy.probs is None and greedy.rows.tolist() == [0]


def check_requests_are_admitted_only_for_exact_samples():
    dflash = SimpleNamespace(is_dflash=lambda: True)
    eagle = SimpleNamespace(is_dflash=lambda: False)

    def reason(top_k=50, min_p=0.0, algorithm=dflash):
        req = SimpleNamespace(sampling_params=SimpleNamespace(top_k=top_k, min_p=min_p))
        return dflash_sampling_mask_unsupported_reason(req, algorithm)

    assert reason() is None and reason(top_k=1) is None
    assert "speculative decoding" in reason(algorithm=eagle)
    assert "min_p" in reason(min_p=0.1)
    THRESHOLDS["acc"] = 0.9
    assert "thresholds" in reason()
    THRESHOLDS["acc"] = 1.0
    dflash_utils._DFLASH_SAMPLING_VERIFY_AVAILABLE = False
    assert "kernel" in reason() and reason(top_k=1) is None
    dflash_utils._DFLASH_SAMPLING_VERIFY_AVAILABLE = True


if __name__ == "__main__":
    failures = 0
    for name, check in list(globals().items()):
        if name.startswith("check_") and callable(check):
            try:
                check()
                print(f"PASS {name}")
            except Exception as error:  # noqa: BLE001 - report every failing check
                failures += 1
                print(f"FAIL {name}: {type(error).__name__}: {error}")
    sys.exit(1 if failures else 0)
