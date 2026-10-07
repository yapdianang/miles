"""SGLang forward hook that saves the DFlash target hidden states of each prefill (``--forward-hooks``).

Registered on the target's ``logits_processor``. Under DFLASH a prefill's ``LogitsProcessorOutput.hidden_states``
holds the concatenated ``target_layer_ids`` residual stream of every extend token, the rows the draft KV cache
is built from. For each request with a ``<dir>/<rid>.spans.json``, TP rank 0 saves the rows at positions inside
those spans to ``<dir>/<rid>.<prefix length>.pt`` (``positions``, ``hidden``), one file per prefill chunk.
"""

import json
import os
from pathlib import Path

import torch
import torch.distributed as dist

from examples.multi_lora.dflash_rl.data import spans_mask


def make_hook(config: dict):
    directory = Path(config["dir"])

    def hook(module, args, output) -> None:
        forward_batch = args[3]
        if forward_batch.forward_mode.name != "EXTEND" or (dist.is_initialized() and dist.get_rank() != 0):
            return
        save_extend_rows(
            directory,
            rids=forward_batch.rids,
            prefix_lens=forward_batch.extend_prefix_lens_cpu,
            extend_lens=forward_batch.extend_seq_lens_cpu,
            hidden=output.hidden_states,
        )

    return hook


def save_extend_rows(
    directory: Path, *, rids: list[str], prefix_lens: list[int], extend_lens: list[int], hidden: torch.Tensor | None
) -> None:
    # A padded batch lists dummy requests after the real ones.
    prefix_lens, extend_lens = prefix_lens[: len(rids)], extend_lens[: len(rids)]
    starts = [sum(extend_lens[:index]) for index in range(len(extend_lens))]
    requests = [
        (rid, prefix, length, start)
        for rid, prefix, length, start in zip(rids, prefix_lens, extend_lens, starts, strict=True)
        if (directory / f"{rid}.spans.json").exists()
    ]
    if requests and (hidden is None or hidden.shape[0] < sum(extend_lens)):
        raise RuntimeError("prefill hidden_states must hold every extend token; run the server with DFLASH")
    for rid, prefix, length, start in requests:
        positions = torch.arange(prefix, prefix + length)
        spans = json.loads((directory / f"{rid}.spans.json").read_text())
        rows = torch.nonzero(spans_mask(positions, spans)).squeeze(1)
        chunk = {"positions": positions[rows], "hidden": hidden[start : start + length][rows.to(hidden.device)].cpu()}
        tmp = directory / f"{rid}.{prefix}.pt.tmp"
        torch.save(chunk, tmp)
        os.replace(tmp, directory / f"{rid}.{prefix}.pt")
