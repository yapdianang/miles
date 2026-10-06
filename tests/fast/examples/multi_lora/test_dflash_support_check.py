import math

from examples.multi_lora.tools.dflash_support_check import check_engine_response


def _meta(tokens, masks, rows, tops=None):
    return {
        "output_token_logprobs": [(-1.0, token, None) for token in tokens],
        "output_token_sampling_mask": masks,
        "output_token_sampling_logprobs": rows,
        "output_top_logprobs": tops,
    }


def _support(probabilities):
    total = sum(probabilities)
    return [math.log(value / total) for value in probabilities]


def test_a_consistent_response_passes_and_reconstructs():
    full = {7: math.log(0.5), 3: math.log(0.3), 9: math.log(0.15), 1: math.log(0.05)}
    tops = [[(value, token, None) for token, value in full.items()]] * 2
    meta = _meta([7, 3], [[7, 3, 9], [7, 3]], [_support([0.5, 0.3, 0.15]), _support([0.5, 0.3])], tops)
    failures, stats = check_engine_response(meta, top_k=1024, tolerance=1e-4)
    assert failures == [] and stats["reconstructed"] == 2 and max(stats["renorm_errors"]) < 1e-9


def test_a_token_outside_its_set_an_oversized_set_and_a_wrong_logprob_fail():
    full = {7: math.log(0.5), 3: math.log(0.3), 9: math.log(0.2)}
    tops = [[(value, token, None) for token, value in full.items()]] * 3
    rows = [_support([0.5, 0.3]), _support([0.5, 0.3, 0.2]), [math.log(0.7), math.log(0.3)]]
    meta = _meta([9, 7, 7], [[7, 3], [7, 3, 9], [7, 3]], rows, tops)
    failures, _ = check_engine_response(meta, top_k=2, tolerance=1e-4)
    assert any("token 0 (9) is not in its set" in failure for failure in failures)
    assert any("token 1 (7): set of 3 ids" in failure for failure in failures)
    assert any("token 2 (7): renormalized" in failure for failure in failures)


def test_missing_supports_fail():
    failures, _ = check_engine_response(_meta([7], None, None), top_k=1024, tolerance=1e-4)
    assert failures == ["the engine returned no sampling supports"]
