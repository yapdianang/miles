"""The DFlash support probe passes a consistent fake engine and flags each broken contract."""

import json
import math
from argparse import Namespace

import httpx
import pytest
from examples.multi_lora.tools import dflash_support_check as probe
from transformers import BatchEncoding

VOCAB_TEXT = {" the": [42]}


class FakeTokenizer:
    """Chat templates tokenize to a BatchEncoding, as in the MiMo image's transformers."""

    def apply_chat_template(self, messages, add_generation_prompt, tokenize):
        rendered = f"<user>{messages[0]['content']}<assistant>"
        return BatchEncoding({"input_ids": self.encode(rendered, False)}) if tokenize else rendered

    def encode(self, text, add_special_tokens):
        return VOCAB_TEXT.get(text, [len(text) % 50 + 3, 5, 6])


def fake_engine(breakage: str | None = None):
    """SGLang /generate with DFLASH: masks per committed token, spec_verify_ct per request."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/server_info":
            return httpx.Response(200, json={"speculative_algorithm": "DFLASH", "speculative_num_draft_tokens": 8})
        body = json.loads(request.content)
        params = body["sampling_params"]
        greedy = params.get("temperature", 1.0) == 0.0
        tokens = [10, 11, 42, 12] if not params.get("ignore_eos") else [10, 11, 12, 13, 14, 15, 16, 17]
        finish = {"type": "length"}
        if 42 in params.get("stop_token_ids", []):
            tokens, finish = tokens[:3], {"type": "stop", "matched": 42}
        if breakage == "overflow" and params.get("top_k") == 50:
            return httpx.Response(400, text="Sampling support exceeds --sampling-mask-max-tokens=4096")
        meta = {
            "output_token_logprobs": [(math.log(0.5), token, None) for token in tokens],
            "completion_tokens": len(tokens),
            "finish_reason": finish,
            "spec_verify_ct": 4 if body.get("return_sampling_mask") and breakage == "slower_accept" else 2,
        }
        if body.get("return_sampling_mask"):
            masks = [[token] if greedy else [token, token + 100] for token in tokens]
            if params.get("top_k") == -1 and params.get("top_p") == 1.0:
                # The whole vocabulary; the packed path would cap it at 4096 ids.
                masks = [list(range(4096 if breakage == "capped" else 5000)) for _ in tokens]
            if breakage == "missing_token":
                masks[1] = [999, 998]
            if breakage == "descending" and params.get("top_k") == -1:
                masks = [mask[::-1] for mask in masks]
            if body.get("sampling_logprobs_mode") == "support":
                values = [
                    [0.0] if greedy else [math.log(0.6 if entry == token else 0.4) for entry in mask]
                    for token, mask in zip(tokens, masks, strict=True)
                ]
            else:
                values = [0.0 if greedy else math.log(0.6) for _ in tokens]
            if breakage == "trimmed_wrong":
                masks, values = masks + [[1]], values + [values[0]]
            meta |= {"output_token_sampling_mask": masks, "output_token_sampling_logprobs": values}
        return httpx.Response(200, json={"meta_info": meta})

    return handler


def run_probe(monkeypatch, breakage=None, packed_ids=False) -> tuple[dict, list[str]]:
    transport = httpx.MockTransport(fake_engine(breakage))
    real_client = httpx.AsyncClient
    monkeypatch.setattr(probe.httpx, "AsyncClient", lambda **kwargs: real_client(transport=transport, **kwargs))
    monkeypatch.setattr(probe.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: FakeTokenizer())
    args = Namespace(
        url="http://engine",
        tokenizer_path="unused",
        top_p=0.97,
        top_k=1024,
        max_tokens=8,
        stop_text=" the",
        case_requests=4,
        sweep_requests=12,
        throughput_requests=8,
        concurrency=4,
        tolerance=1e-5,
        accept_z=3.0,
        timeout_seconds=30.0,
        packed_ids=packed_ids,
    )
    report = probe.asyncio.run(probe.run(args))
    return report, probe.failures_of(report, args)


def test_a_consistent_engine_passes(monkeypatch):
    report, failures = run_probe(monkeypatch)
    assert failures == []
    json.dumps(report)  # prompts are plain token ids, not a BatchEncoding
    assert report["cases"]["stopped_on_stop_token"] == 4
    assert report["cases"]["widest_mask"]["h_full_vocab"] == 5000
    assert set(report["sweep"]["requests"]) == {"a_support", "b_selected", "d_top_k", "f_top_p", "g_top_p_selected"}
    assert report["throughput"]["on"]["accept_length"] == report["throughput"]["off"]["accept_length"] == 4.0


def test_packed_ids_skip_bitmap_only_checks(monkeypatch):
    report, failures = run_probe(monkeypatch, "descending", packed_ids=True)
    assert failures == []
    assert set(report["cases"]["requests"]) == {"a_support", "b_selected", "c_greedy", "d_top_k", "e_stop"}


@pytest.mark.parametrize(
    "breakage, expected",
    [
        ("missing_token", "is not in its mask"),
        ("trimmed_wrong", "masks,"),
        ("overflow", "aborted: Sampling support exceeds"),
        ("slower_accept", "accept length with masks differs"),
        ("descending", "not ascending"),
        ("capped", "expected at least 4097"),
    ],
)
def test_each_broken_contract_fails(monkeypatch, breakage, expected):
    report, failures = run_probe(monkeypatch, breakage)
    assert any(expected in failure for failure in failures), failures[:5]
    if breakage == "overflow":
        assert report["sweep"]["aborts"]["OVERFLOW"] == report["sweep"]["requests"]["d_top_k"] == 2


def test_check_response_rejects_unnormalized_support_and_non_singleton_greedy():
    support = {"mode": "support", "top_k": 1024, "ascending": True}
    meta = {
        "output_token_logprobs": [(-1.0, 7, None)],
        "completion_tokens": 1,
        "finish_reason": {"type": "length"},
        "output_token_sampling_mask": [[7, 8]],
        "output_token_sampling_logprobs": [[math.log(0.5), math.log(0.4)]],
    }
    assert "do not sum to one" in probe.check_response(meta, support, 1e-5)[0]
    greedy = {"mode": "selected", "top_k": 1, "greedy": True, "ascending": True}
    meta |= {"output_token_sampling_mask": [[7]], "output_token_sampling_logprobs": [-0.1]}
    assert "greedy mask" in probe.check_response(meta, greedy, 1e-5)[0]


@pytest.mark.parametrize(
    "probabilities, passes",
    [
        ([0.4, 0.3, 0.1, 0.1, 0.1], True),  # top_k 3: the third probability is tied twice past the cutoff
        ([0.4, 0.3, 0.12, 0.1, 0.08], False),  # top_k 3 with two smaller ids: a real overflow
    ],
)
def test_masks_past_top_k_pass_only_for_cutoff_ties(probabilities, passes):
    case = {"mode": "support", "top_k": 3, "ascending": True}
    meta = {
        "output_token_logprobs": [(math.log(0.4), 7, None)],
        "completion_tokens": 1,
        "finish_reason": {"type": "length"},
        "output_token_sampling_mask": [[7, 8, 9, 10, 11]],
        "output_token_sampling_logprobs": [[math.log(value) for value in probabilities]],
    }
    failures = probe.check_response(meta, case, 1e-5)
    assert (failures == []) is passes, failures
    selected = meta | {"output_token_sampling_logprobs": [math.log(0.4)]}
    selected_case = {"mode": "selected", "top_k": 3, "ascending": True}
    assert probe.check_response(selected, selected_case, 1e-5), "selected mode has no tie evidence"
    short_row = meta | {"output_token_sampling_logprobs": [[math.log(0.4), math.log(0.6)]]}
    assert probe.check_response(short_row, case, 1e-5), "a row shorter than its mask is reported, not raised"
