"""Return the sampling support of every token DFlash commits, as non-speculative decoding does.

SGLang rejects ``return_sampling_mask`` under speculative decoding. DFlash verify draws each
committed token (accepted drafts and the bonus) exactly from its per-position target
distribution, so the support is that distribution's non-zero set. The patch:

- builds the target distribution with the plain sampler's joint filter order (top-p cutoff
  from the full distribution, intersected with top-k) instead of top-k first;
- copies the opted-in requests' verify distributions out of the borrowed graph pool and packs
  each committed token's row with the plain sampler's own packer, statuses included;
- lets the result processor keep one row per committed token, trimmed where output stops;
- accepts mask requests under DFlash when every committed token is an exact sample.

The sources are those of sglang-miles f19dcfb; the script refuses another SGLang build.
"""

import importlib.metadata
from pathlib import Path

SGLANG_VERSION = "0.5.22.dev36+gf19dcfb"
SGLANG_SRT = Path("/sgl-workspace/sglang/python/sglang/srt")
MARKER = "DFLASH_SAMPLING_MASK_PATCH"

DFLASH_UTILS_PATCHES = (
    (
        """\
from sglang.srt.layers.sampler import (
    apply_custom_logit_processor,
    top_p_normalize_probs_torch,
)
""",
        """\
from sglang.srt.layers.logits_processor import SamplingMaskOutput, SamplingMaskStatus
from sglang.srt.layers.sampler import (
    _SamplingMaskCapture,
    apply_custom_logit_processor,
    top_p_normalize_probs_torch,
)
""",
    ),
    (
        """\
from sglang.srt.runtime_context import get_spec
from sglang.srt.speculative.spec_utils import sample_simulated_acc_len
""",
        """\
from sglang.srt.runtime_context import get_parallel, get_spec
from sglang.srt.speculative.spec_utils import SIMULATE_ACC_LEN, sample_simulated_acc_len
""",
    ),
    (
        """\
    use_sparse_topk: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    \"\"\"Compute DFlash accept lengths and bonus tokens for non-greedy sampling.
""",
        """\
    use_sparse_topk: bool = True,
    support_capture: Optional["DFlashSupportCapture"] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    \"\"\"Compute DFlash accept lengths and bonus tokens for non-greedy sampling.
""",
    ),
    (
        """\
            use_sparse_topk=use_sparse_topk,
        )
        draft_probs = torch.zeros_like(target_probs)
""",
        """\
            use_sparse_topk=use_sparse_topk,
        )
        if support_capture is not None:
            # target_probs dies with the borrowed graph pool; the capture buffer does not
            support_capture.probs.copy_(target_probs.index_select(0, support_capture.rows))
        draft_probs = torch.zeros_like(target_probs)
""",
    ),
    (
        """\
    if use_sparse_topk and need_top_k:
""",
        """\
    # The sparse path renormalizes over the top-k; with top-p, the joint cutoff needs the full distribution.
    if use_sparse_topk and need_top_k and not need_top_p:
""",
    ),
    (
        """\
    if not sparse_topk_applied:
        target_probs = F.softmax(scaled_logits, dim=-1)
        if need_top_k:
            target_probs = _dflash_top_k_renorm_prob(
                target_probs,
                torch.repeat_interleave(sampling_info.top_ks, draft_token_num, dim=0),
            )
        if need_top_p:
            target_probs = _dflash_top_p_renorm_prob(
                target_probs,
                torch.repeat_interleave(sampling_info.top_ps, draft_token_num, dim=0),
            )
""",
        """\
    if not sparse_topk_applied:
        # Joint filter order, as the plain sampler: both cutoffs from the full distribution.
        probs = F.softmax(scaled_logits, dim=-1)
        target_probs = probs
        if need_top_k:
            target_probs = _dflash_top_k_renorm_prob(
                probs,
                torch.repeat_interleave(sampling_info.top_ks, draft_token_num, dim=0),
            )
        if need_top_p:
            top_p_probs = _dflash_top_p_renorm_prob(
                probs,
                torch.repeat_interleave(sampling_info.top_ps, draft_token_num, dim=0),
            )
            if need_top_k:
                target_probs = target_probs.masked_fill(top_p_probs <= 0, 0.0)
                target_probs = target_probs / target_probs.sum(dim=-1, keepdim=True)
            else:
                target_probs = top_p_probs
""",
    ),
)

DFLASH_UTILS_APPENDIX = '''

# Sampling masks under DFlash (docker/compat/patch_sglang_dflash_sampling_mask.py in Miles).
DFLASH_SAMPLING_MASK_PATCH = 1


@dataclass
class DFlashSupportCapture:
    """Verify distributions of the requests that return sampling masks, kept past the verify step.

    ``rows`` are their batch rows; ``probs`` is ``[len(rows), block, vocab]``, or None when the
    block was verified by greedy argmax.
    """

    rows: torch.Tensor
    probs: Optional[torch.Tensor]

    @classmethod
    def for_batch(
        cls, sampling_info: Any, next_token_logits: torch.Tensor, block: int, sampled: bool
    ) -> Optional["DFlashSupportCapture"]:
        rows = getattr(sampling_info, "sampling_mask_batch_indices", None)
        if rows is None:
            return None
        probs = None
        if sampled:
            shape = (rows.numel(), block, next_token_logits.shape[-1])
            probs = torch.empty(shape, dtype=torch.float32, device=next_token_logits.device)
        return cls(rows=rows, probs=probs)


def dflash_sampling_mask_unsupported_reason(req: Req, spec_algorithm: Any) -> Optional[str]:
    """Why committed tokens would not be exact samples from the captured distribution, if they would not."""
    if not spec_algorithm.is_dflash():
        return "return_sampling_mask is not supported with speculative decoding."
    if get_parallel().pp_size > 1:
        return "return_sampling_mask with DFlash does not support pipeline parallelism."
    if req.sampling_params.min_p > 0:
        return "return_sampling_mask with DFlash does not support min_p."
    if SIMULATE_ACC_LEN > 0:
        return "return_sampling_mask is not supported with simulated speculative acceptance."
    spec = get_spec()
    if spec.speculative_accept_threshold_single != 1.0 or spec.speculative_accept_threshold_acc != 1.0:
        return "return_sampling_mask with DFlash requires acceptance thresholds of 1.0."
    if req.sampling_params.top_k != 1 and not is_dflash_sampling_verify_available():
        return "return_sampling_mask with non-greedy DFlash needs the sampling verification kernel."
    return None


def build_dflash_sampling_mask_output(
    sampler: Any,
    sampling_info: Any,
    capture: Optional[DFlashSupportCapture],
    out_tokens: torch.Tensor,
    commit_lens: torch.Tensor,
) -> SamplingMaskOutput:
    """Pack the support of every verify position with the plain sampler's packer.

    Verify row ``(b, j)`` is the distribution ``out_tokens[b, j]`` was drawn from: an accepted draft
    for ``j < accept_len``, the bonus at ``accept_len``. Rows are ``[request, position]`` flattened;
    the host keeps each request's first ``num_accept_tokens``. A block verified by a path that
    does not expose its distribution (``capture`` None) is invalid.
    """
    rows = sampling_info.sampling_mask_batch_indices
    block = out_tokens.shape[1]
    tokens = out_tokens.index_select(0, rows).reshape(-1)
    flat_rows = torch.arange(tokens.numel(), device=tokens.device)
    support_rows = sampling_info.sampling_support_logprobs_capture_indices
    if support_rows is not None:
        support_rows = (support_rows[:, None] * block + torch.arange(block, device=support_rows.device)).reshape(-1)
    if capture is None or capture.probs is None:
        output = sampler._build_greedy_sampling_mask_output(flat_rows, tokens, support_rows)
        if capture is None:
            output.statuses.fill_(int(SamplingMaskStatus.INVALID))
    else:
        weights = capture.probs.view(tokens.numel(), -1)
        output = sampler._build_sampling_mask_output(
            tokens,
            _SamplingMaskCapture(weights=weights, token_ids=None, selected_weight=None, batch_rows=flat_rows),
            support_rows,
        )
    output.num_accept_tokens = commit_lens.index_select(0, rows).to(torch.int32)
    return output
'''

WORKER_PATCHES = (
    (
        """\
    compute_dflash_sampling_correct_drafts_and_bonus,
    is_dense_head_weight,
""",
        """\
    compute_dflash_sampling_correct_drafts_and_bonus,
    DFlashSupportCapture,
    build_dflash_sampling_mask_output,
    is_dense_head_weight,
""",
    ),
    (
        """\
        new_seq_lens = None
        target_predict = None
        if self._selector_sample is not None:
""",
        """\
        new_seq_lens = None
        target_predict = None
        # A selector-verified block keeps no capture; its sampling masks are invalid.
        support_capture = None
        if self._selector_sample is not None:
""",
    ),
    (
        """\
            accept_len, bonus = compute_dflash_sampling_correct_drafts_and_bonus(
                candidates=candidates,
                next_token_logits=next_token_logits,
                sampling_info=sampling_info,
                max_top_k=draft_input.max_top_k,
                uniform_top_k_value=draft_input.uniform_top_k_value,
            )
""",
        """\
            support_capture = DFlashSupportCapture.for_batch(
                sampling_info, next_token_logits, int(self.block_size), sampled=True
            )
            accept_len, bonus = compute_dflash_sampling_correct_drafts_and_bonus(
                candidates=candidates,
                next_token_logits=next_token_logits,
                sampling_info=sampling_info,
                max_top_k=draft_input.max_top_k,
                uniform_top_k_value=draft_input.uniform_top_k_value,
                support_capture=support_capture,
            )
""",
    ),
    (
        """\
        else:
            target_predict = torch.argmax(next_token_logits, dim=-1).view(
                bs, int(self.block_size)
            )
""",
        """\
        else:
            support_capture = DFlashSupportCapture.for_batch(
                sampling_info, next_token_logits, int(self.block_size), sampled=False
            )
            target_predict = torch.argmax(next_token_logits, dim=-1).view(
                bs, int(self.block_size)
            )
""",
    ),
    (
        """\
        return accept_len, commit_lens, bonus, out_tokens, new_seq_lens, target_predict
""",
        """\
        return accept_len, commit_lens, bonus, out_tokens, new_seq_lens, target_predict, support_capture
""",
    ),
    (
        """\
            new_seq_lens,
            target_predict,
        ) = self._accept_block(
""",
        """\
            new_seq_lens,
            target_predict,
            support_capture,
        ) = self._accept_block(
""",
    ),
    (
        """\
        if batch.return_logprob:
            compute_spec_logprobs(
                batch,
                logits_output,
                out_tokens.reshape(-1),
                chain_stride=block_size,
            )
""",
        """\
        if sampling_info is not None and sampling_info.sampling_mask_batch_indices is not None:
            logits_output.sampling_mask_output = build_dflash_sampling_mask_output(
                self.target_worker.model_runner.sampler, sampling_info, support_capture, out_tokens, commit_lens
            )

        if batch.return_logprob:
            compute_spec_logprobs(
                batch,
                logits_output,
                out_tokens.reshape(-1),
                chain_stride=block_size,
            )
""",
    ),
)

LOGITS_PROCESSOR_PATCHES = (
    (
        """\
    support_logprobs: Optional[torch.Tensor]
    statuses: torch.Tensor
""",
        """\
    support_logprobs: Optional[torch.Tensor]
    statuses: torch.Tensor
    # Speculative verify: rows are [request, position]; a request keeps its committed prefix.
    num_accept_tokens: Optional[torch.Tensor] = None
""",
    ),
    (
        """\
        self.statuses = fn(self.statuses)
""",
        """\
        self.statuses = fn(self.statuses)
        if self.num_accept_tokens is not None:
            self.num_accept_tokens = fn(self.num_accept_tokens)
""",
    ),
)

SCHEDULER_PATCHES = (
    (
        """\
        if req.return_sampling_mask and not self.spec_algorithm.is_none():
            # Spec workers do not emit one sampling support per accepted token, so
            # the returned mask would not align 1:1 with generated tokens. Reject
            # the combination instead of silently returning a misaligned mask.
            error_msg = (
                "return_sampling_mask is not supported with speculative decoding."
            )
            self._reject_sampling_mask_request(req, error_msg)
            return
""",
        """\
        if req.return_sampling_mask and not self.spec_algorithm.is_none():
            # DFlash emits one support per committed token; other spec workers do not.
            from sglang.srt.speculative.dflash_utils import (
                dflash_sampling_mask_unsupported_reason,
            )

            error_msg = dflash_sampling_mask_unsupported_reason(req, self.spec_algorithm)
            if error_msg is not None:
                self._reject_sampling_mask_request(req, error_msg)
                return
""",
    ),
)

RESULT_PROCESSOR_PATCHES = (
    (
        """\
            if req.return_sampling_mask:
                # return_sampling_mask + speculative decoding is rejected at
                # request entry, so this remains one support mask per token.
                self.add_sampling_mask_return_values(i, req, logits_output)
""",
        """\
            if req.return_sampling_mask:
                self.add_sampling_mask_return_values(
                    i, req, logits_output, accept_len=new_accept_len
                )
""",
    ),
    (
        """\
        output: LogitsProcessorOutput,
    ) -> None:
        \"\"\"Attach sparse sampling support metadata to the return values.\"\"\"
        req.sampling_mask_rows.append(
            output.next_token_sampling_mask_idx[i],
            output.next_token_sampling_logprobs[i],
        )
""",
        """\
        output: LogitsProcessorOutput,
        *,
        accept_len: int = 1,
    ) -> None:
        \"\"\"Attach sparse sampling support metadata to the return values.\"\"\"
        masks = output.next_token_sampling_mask_idx[i]
        logprobs = output.next_token_sampling_logprobs[i]
        if not isinstance(masks, list):
            req.sampling_mask_rows.append(masks, logprobs)
            return
        # Speculative rows: one per committed token, cut where output stops like the logprobs.
        step_start = len(req.output_ids) - accept_len
        visible = max(0, min(accept_len, self._visible_output_len(req) - step_start))
        for token in range(min(visible, len(masks))):
            req.sampling_mask_rows.append(masks[token], logprobs[token])
""",
    ),
    (
        """\
        assert len(batch_indices) == len(lengths)
""",
        """\
        speculative = sampling_output.num_accept_tokens is not None
        assert speculative or len(batch_indices) == len(lengths)
""",
    ),
    (
        """\
        support_row = 0
        for row, batch_index in enumerate(batch_indices):
            returns_support_logprobs = (
""",
        """\
        if speculative:
            # Rows are [request, position]; keep each request's committed prefix.
            block = len(lengths) // max(len(batch_indices), 1)
            num_accept = sampling_output.num_accept_tokens.tolist()
            support_row = 0
            for row, batch_index in enumerate(batch_indices):
                returns_support = reqs[batch_index].sampling_logprobs_mode == "support"
                first, count = row * block, int(num_accept[row])
                status = max(int(value) for value in statuses[first : first + count])
                status_by_batch[batch_index] = status
                if status == SamplingMaskStatus.OK:
                    masks[batch_index] = [token_ids[first + j, : lengths[first + j]] for j in range(count)]
                    if returns_support:
                        logprobs[batch_index] = [
                            support_logprobs[support_row * block + j, : lengths[first + j]] for j in range(count)
                        ]
                    else:
                        logprobs[batch_index] = [selected_logprobs[first + j : first + j + 1] for j in range(count)]
                if returns_support:
                    support_row += 1
            output.next_token_sampling_mask_idx = masks
            output.next_token_sampling_logprobs = logprobs
            output.next_token_sampling_mask_status = status_by_batch
            output.sampling_mask_output = None
            return

        support_row = 0
        for row, batch_index in enumerate(batch_indices):
            returns_support_logprobs = (
""",
    ),
)


def patch(path: Path, replacements, appendix: str = "") -> None:
    source = path.read_text()
    for before, after in replacements:
        if after in source:
            continue
        if before not in source:
            raise RuntimeError(f"expected source not found in {path}:\n{before}")
        source = source.replace(before, after, 1)
    if appendix and MARKER not in source:
        source += appendix
    path.write_text(source)


def main(srt: Path = SGLANG_SRT, sglang_version: str | None = SGLANG_VERSION) -> None:
    if sglang_version is not None and importlib.metadata.version("sglang") != sglang_version:
        raise RuntimeError(
            f"the DFlash sampling-mask patch targets SGLang {sglang_version}, "
            f"found {importlib.metadata.version('sglang')}"
        )
    patch(srt / "speculative/dflash_utils.py", DFLASH_UTILS_PATCHES, DFLASH_UTILS_APPENDIX)
    patch(srt / "speculative/dflash_worker_v2.py", WORKER_PATCHES)
    patch(srt / "layers/logits_processor.py", LOGITS_PROCESSOR_PATCHES)
    patch(srt / "managers/scheduler.py", SCHEDULER_PATCHES)
    patch(srt / "managers/scheduler_components/batch_result_processor.py", RESULT_PROCESSOR_PATCHES)


if __name__ == "__main__":
    main()
