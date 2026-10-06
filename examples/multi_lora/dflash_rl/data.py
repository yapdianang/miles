"""Rollout and hidden-state shard formats for DFlash drafter training.

A rollout is one trajectory's full token sequence with ``loss_mask`` 1 on every sampled token. Its shard keeps
only the positions training reads: each sampled turn and the ``sliding_window - 1`` positions before it (the
context of the turn's first block). A shard holds ``input_ids``, ``positions`` (absolute, ascending),
``loss_mask`` and ``hidden`` ([n, len(target_layer_ids) * hidden_size], the target layers' residual stream).
"""

from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def turn_spans(loss_mask: list[int]) -> list[tuple[int, int]]:
    """[start, end) of each maximal run of sampled tokens."""
    spans, start = [], None
    for index, value in enumerate([*loss_mask, 0]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            spans.append((start, index))
            start = None
    return spans


def context_spans(loss_mask: list[int], window: int) -> list[tuple[int, int]]:
    """Merged [start, end) spans of each turn plus the ``window - 1`` positions before it."""
    merged: list[tuple[int, int]] = []
    for start, end in turn_spans(loss_mask):
        start = max(0, start - (window - 1))
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def replay_turns(tokens: list[int], loss_mask: list[int]) -> list[dict]:
    """The rollout as replay_bench turns: each turn's prompt and its sampled token count."""
    return [{"prompt": tokens[:start], "output_len": end - start} for start, end in turn_spans(loss_mask)]


def spans_mask(positions: torch.Tensor, spans: list[tuple[int, int]]) -> torch.Tensor:
    keep = torch.zeros_like(positions, dtype=torch.bool)
    for start, end in spans:
        keep |= (positions >= start) & (positions < end)
    return keep


def assemble_shard(
    tokens: list[int], loss_mask: list[int], spans: list[tuple[int, int]], chunks: list[dict], hidden_size: int
) -> dict[str, torch.Tensor]:
    """Join the capture hook's per-chunk files (``positions``, ``hidden``) into a shard; fail on gaps."""
    chunks = sorted(chunks, key=lambda chunk: int(chunk["positions"][0]) if len(chunk["positions"]) else -1)
    positions = torch.cat([chunk["positions"] for chunk in chunks]).long()
    hidden = torch.cat([chunk["hidden"] for chunk in chunks])
    expected = torch.nonzero(spans_mask(torch.arange(len(tokens)), spans)).squeeze(1)
    if not torch.equal(positions, expected):
        raise ValueError(f"captured {len(positions)} positions, expected {len(expected)} (missing prefill chunks?)")
    if hidden.shape[1] != hidden_size:
        raise ValueError(f"captured hidden width {hidden.shape[1]}, drafter expects {hidden_size}")
    return {
        "input_ids": torch.tensor(tokens, dtype=torch.int64)[positions],
        "positions": positions,
        "loss_mask": torch.tensor(loss_mask, dtype=torch.bool)[positions],
        "hidden": hidden.to(torch.bfloat16).contiguous(),
    }


def save_shard(shard: dict[str, torch.Tensor], path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    save_file({**shard, "loss_mask": shard["loss_mask"].to(torch.uint8)}, tmp)
    tmp.replace(path)


def load_shard(path: Path, device: str = "cpu") -> dict[str, torch.Tensor]:
    shard = load_file(path, device=device)
    return {**shard, "loss_mask": shard["loss_mask"].bool()}


def valid_anchors(shard: dict[str, torch.Tensor]) -> torch.Tensor:
    """Indices whose token and next token are sampled and adjacent: block starts with at least one label."""
    loss_mask, positions = shard["loss_mask"], shard["positions"]
    valid = loss_mask[:-1] & loss_mask[1:] & (positions[1:] == positions[:-1] + 1)
    return torch.nonzero(valid).squeeze(1)


@dataclass(frozen=True)
class Chunk:
    block_ids: torch.Tensor  # [N, B]: anchor token, then mask tokens
    block_positions: torch.Tensor  # [N, B]
    labels: torch.Tensor  # [N, B - 1]: token at anchor + k
    label_mask: torch.Tensor  # [N, B - 1]: label is a sampled token at that position
    ctx_hidden: torch.Tensor  # [C, F]
    ctx_positions: torch.Tensor  # [C]
    ctx_visible: torch.Tensor  # [N, B, C]


def make_chunk(
    shard: dict[str, torch.Tensor], anchors: torch.Tensor, *, block_size: int, window: int, mask_token_id: int
) -> Chunk:
    """Blocks at ascending shard indices ``anchors`` and the context rows they attend to."""
    input_ids, positions, loss_mask = shard["input_ids"], shard["positions"], shard["loss_mask"]
    offsets = torch.arange(block_size, device=anchors.device)
    anchor_positions = positions[anchors]
    index = (anchors[:, None] + offsets).clamp(max=len(positions) - 1)
    present = positions[index] == anchor_positions[:, None] + offsets
    block_ids = torch.full_like(index, mask_token_id)
    block_ids[:, 0] = input_ids[anchors]
    lo = int(torch.searchsorted(positions, anchor_positions[0] - (window - 1)))
    hi = int(anchors[-1])
    ctx_positions = positions[lo:hi]
    lower = anchor_positions[:, None, None] + offsets[None, :, None] - (window - 1)
    ctx_visible = (ctx_positions < anchor_positions[:, None, None]) & (ctx_positions >= lower)
    if not torch.equal(ctx_visible[:, 0].sum(dim=-1), anchor_positions.clamp(max=window - 1)):
        raise ValueError("shard misses context positions inside the draft window")
    return Chunk(
        block_ids=block_ids,
        block_positions=anchor_positions[:, None] + offsets,
        labels=input_ids[index[:, 1:]],
        label_mask=(present & loss_mask[index])[:, 1:],
        ctx_hidden=shard["hidden"][lo:hi],
        ctx_positions=ctx_positions,
        ctx_visible=ctx_visible,
    )


def walk_accept(shard: dict[str, torch.Tensor], anchors: list[int], accepted: list[int]) -> tuple[int, int]:
    """Tokens and verify steps of decoding each turn block by block, as the engine does: (sum of 1 + accepted, steps)."""
    accepted_at = dict(zip(anchors, accepted, strict=True))
    tokens = steps = 0
    for start, _ in turn_spans(shard["loss_mask"].tolist()):
        index = start
        while index in accepted_at:
            tokens += accepted_at[index] + 1
            steps += 1
            index += accepted_at[index] + 1
    return tokens, steps
