"""Expert-load balance of the engine routes a forward_backward replays (MiMo-V2.6 section 5.4, Figure 11)."""

import numpy as np

# an expert is cold below this fraction of its layer's mean load
COLD_LOAD = 0.1


def moe_layers(moe_layer_freq: int | list[int], num_layers: int) -> list[int]:
    """Decoder layers with routed experts, from Megatron's --moe-layer-freq."""
    if isinstance(moe_layer_freq, int):
        return [layer for layer in range(num_layers) if layer % moe_layer_freq == 0]
    return [layer for layer, moe in enumerate(moe_layer_freq) if moe]


def expert_counts(routes: np.ndarray, layers: list[int], num_experts: int) -> np.ndarray:
    """Tokens routed to each expert as ``[num_layers, num_experts]`` from ``[tokens, num_layers, topk]`` routes.

    Rows of layers without routed experts stay zero: the engine leaves their routes unset.
    """
    counts = np.zeros((routes.shape[1], num_experts), dtype=np.int64)
    for layer in layers:
        counts[layer] = np.bincount(routes[:, layer].ravel(), minlength=num_experts)
    return counts


def expert_load_metrics(counts: np.ndarray) -> dict[str, float]:
    """Per-layer CV (std/mean), peak load (max/mean) and cold-expert fraction, and their mean and max over layers.

    Keys end in the reduction the Tinker SDK applies when it combines chunked forward_backward results.
    """
    layers = np.flatnonzero(counts.sum(-1))
    if not len(layers):
        return {}
    loads = counts[layers]
    mean = loads.mean(-1)
    statistics = {
        "cv": loads.std(-1) / mean,
        "peak": loads.max(-1) / mean,
        "cold": (loads < COLD_LOAD * mean[:, None]).mean(-1),
    }
    metrics = {}
    for name, values in statistics.items():
        metrics |= {
            f"expert_load/{name}/layer{layer}:mean": float(value) for layer, value in zip(layers, values, strict=True)
        }
        metrics[f"expert_load/{name}/layer_mean:mean"] = float(values.mean())
        metrics[f"expert_load/{name}/layer_max:max"] = float(values.max())
    return metrics
