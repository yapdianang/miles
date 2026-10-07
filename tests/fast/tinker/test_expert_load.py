"""Expert-load metrics of replayed routes reach the forward_backward result metrics."""

import numpy as np
import pytest
import torch

from miles.tinker.expert_load import expert_counts, expert_load_metrics, moe_layers
from miles.tinker.runtime import MilesBackend, RoutedExpertsCache
from miles.tinker.server.encoding import render_result
from miles.tinker.server.proto_codec import encode_forward_backward_output
from tinker.lib.chunked_fwdbwd_helpers import _metrics_reduction
from tinker.proto.response_conv import deserialize_forward_backward_output


def test_moe_layers_follow_megatron_moe_layer_freq():
    assert moe_layers(1, 3) == [0, 1, 2]
    assert moe_layers(2, 5) == [0, 2, 4]
    assert moe_layers([0, 1, 1, 0], 4) == [1, 2]


def test_counts_cover_only_moe_layers():
    # [tokens, layers, topk]; layer 0 is dense, and the engine leaves its routes zero
    routes = np.array([[[0, 0], [1, 3], [2, 0]], [[0, 0], [1, 2], [2, 1]], [[0, 0], [3, 2], [2, 3]]])
    counts = expert_counts(routes, [1, 2], num_experts=4)
    np.testing.assert_array_equal(counts, [[0, 0, 0, 0], [0, 2, 2, 2], [1, 1, 3, 1]])


def test_metrics_per_layer_and_over_layers():
    counts = np.array(
        [
            [0, 0, 0, 0, 0],  # a dense layer has no load and no metrics
            [8, 4, 4, 4, 0],  # mean 4: cv sqrt(32 / 5) / 4, peak 2, one cold expert
            [5, 5, 5, 5, 5],  # balanced
            [2, 0, 9, 9, 10],  # mean 6: 0.6 is the cold bound, so 2 is warm
        ]
    )
    metrics = expert_load_metrics(counts)
    cv = [np.sqrt(32 / 5) / 4, 0.0, np.std([2, 0, 9, 9, 10]) / 6]
    peak, cold = [2.0, 1.0, 10 / 6], [0.2, 0.0, 0.2]
    for layer, expected in zip((1, 2, 3), zip(cv, peak, cold, strict=True), strict=True):
        actual = [metrics[f"expert_load/{name}/layer{layer}:mean"] for name in ("cv", "peak", "cold")]
        assert actual == pytest.approx(expected)
    for name, values in (("cv", cv), ("peak", peak), ("cold", cold)):
        assert metrics[f"expert_load/{name}/layer_mean:mean"] == pytest.approx(np.mean(values))
        assert metrics[f"expert_load/{name}/layer_max:max"] == pytest.approx(max(values))
    assert not any("layer0" in key for key in metrics)
    assert len(metrics) == 3 * (3 + 2)
    assert expert_load_metrics(np.zeros((2, 4), dtype=np.int64)) == {}


def _datum(tokens: list[int]) -> dict:
    return {"tokens": tokens, "target_len": len(tokens) - 1, "target_tokens": tokens[1:]}


async def test_forward_backward_results_carry_the_expert_load_of_their_datums():
    generator = np.random.default_rng(0)
    datums = [_datum([1, 2, 3, 4]), _datum([5, 6, 7])]
    cache = RoutedExpertsCache(max_bytes=1 << 20)
    # [tokens, layers, topk] with layer 0 dense; every MoE token picks two distinct experts out of 4
    routes = [
        np.stack(
            [np.zeros((len(datum["tokens"]) - 1, 2), dtype=np.int16)]
            + [np.stack([generator.permutation(4)[:2] for _ in range(len(datum["tokens"]) - 1)]) for _ in range(2)],
            axis=1,
        )
        for datum in datums
    ]
    for datum, route in zip(datums, routes, strict=True):
        cache.put(datum["tokens"][:-1], route)
    backend = MilesBackend(
        trainer=None, router_url="http://router", dp_size=4, routed_experts=cache, moe_layers=[1, 2], num_experts=4
    )

    async def fake_call_trainer(method, batch_id, train_data):
        per_datum = [
            {"sample_index": index, "loss": 1.0, "logprobs": torch.zeros(len(tokens) - 1)}
            for index, tokens in enumerate(train_data["tokens"])
        ]
        return [{"per_datum": per_datum}]

    backend._call_trainer = fake_call_trainer
    outputs = await backend.forward_backward(1, [(0, datum) for datum in datums], "cross_entropy", {})

    # the two DP-padding datums add no load
    expected = sum(expert_counts(route.astype(np.int32), [1, 2], 4) for route in routes)
    assert expected.sum() == (3 + 2) * 2 * 2
    np.testing.assert_array_equal(sum(output["expert_counts"] for output in outputs), expected)
    result = {"op": "forward_backward", "outputs": outputs}
    rendered = render_result(result)["metrics"]
    assert rendered == {"loss:sum": 2.0, **expert_load_metrics(expected)}
    assert {f"expert_load/{name}/layer{layer}:mean" for name in ("cv", "peak", "cold") for layer in (1, 2)} < set(
        rendered
    )
    parsed = deserialize_forward_backward_output(encode_forward_backward_output(result))
    assert dict(parsed.metrics) == pytest.approx(rendered)
    # the SDK combines a chunked forward_backward by each key's reduction suffix
    assert _metrics_reduction([parsed, parsed]) == pytest.approx(rendered | {"loss:sum": 4.0})


async def test_results_without_routes_report_only_the_loss():
    backend = MilesBackend(trainer=None, router_url="http://router", moe_layers=[1, 2], num_experts=4)

    async def fake_call_trainer(method, batch_id, train_data):
        return [{"per_datum": [{"sample_index": 0, "loss": 1.5, "logprobs": torch.zeros(1)}]}]

    backend._call_trainer = fake_call_trainer
    outputs = await backend.forward_backward(1, [(0, _datum([1, 2]))], "cross_entropy", {})
    assert render_result({"op": "forward_backward", "outputs": outputs})["metrics"] == {"loss:sum": 1.5}
