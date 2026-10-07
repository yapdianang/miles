"""Sliding-window attention under zigzag context parallelism (MiMo-V2.6 section 6.4).

Rank r holds chunks r and 2 * cp - 1 - r of every packed sequence (TE's DualChunkSwap layout). A
sliding-window query reaches only the ``window`` keys before it, so each local chunk needs only the
``window`` keys and values that precede it, wherever they live. One all-to-all moves exactly those
rows, at most ``2 * window`` per sequence and rank at any sequence length, and the layer attends
locally over ``[halo | chunk]`` with a bottom-right causal window. A learnable sink needs no
exchange: every query sees all of its keys on one rank.
"""

from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.distributed as dist
from torch import Tensor


@dataclass(frozen=True)
class SwaHaloPlan:
    """This rank's side of the halo exchange and the layout of its local attention.

    Attributes:
        send_rows: local rows other ranks need, grouped by destination rank.
        send_splits: rows sent to each rank.
        recv_splits: rows received from each rank.
        kv_rows: rows of ``cat([local, received])`` laid out as ``[halo | chunk]`` per local chunk.
        cu_seqlens_q: int32 boundaries of the local chunks.
        cu_seqlens_kv: int32 boundaries of the ``[halo | chunk]`` segments.
        max_seqlen_q: longest local chunk.
        max_seqlen_kv: longest ``[halo | chunk]`` segment.
    """

    send_rows: Tensor
    send_splits: tuple[int, ...]
    recv_splits: tuple[int, ...]
    kv_rows: Tensor
    cu_seqlens_q: Tensor
    cu_seqlens_kv: Tensor
    max_seqlen_q: int
    max_seqlen_kv: int


@dataclass(frozen=True)
class _HaloPiece:
    """Rows ``[src_row, src_row + rows)`` of rank ``src``: consecutive positions before a chunk."""

    src: int
    src_row: int
    rows: int


def _chunk_owner(chunk: int, cp_size: int) -> int:
    return chunk if chunk < cp_size else 2 * cp_size - 1 - chunk


def _local_chunks(cu_seqlens: tuple[int, ...], *, cp_rank: int, cp_size: int, window: int):
    """Rank ``cp_rank``'s chunks in local order as (first local row, rows, halo pieces by position)."""
    chunks = []
    for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:], strict=True):
        if end <= start or (end - start) % (2 * cp_size):
            raise ValueError(f"packed sequence [{start}, {end}) is not a positive multiple of 2 * cp = {2 * cp_size}")
        chunk_rows = (end - start) // (2 * cp_size)
        # every rank holds 2 * chunk_rows rows of each sequence, so a sequence starts on the same local row everywhere
        seq_row = start // cp_size
        for slot, chunk in enumerate((cp_rank, 2 * cp_size - 1 - cp_rank)):
            first = max(0, chunk * chunk_rows - window)
            halo = []
            for src_chunk in range(first // chunk_rows, chunk):
                lo = max(first, src_chunk * chunk_rows)
                src_slot = 0 if src_chunk < cp_size else 1
                src_row = seq_row + src_slot * chunk_rows + lo - src_chunk * chunk_rows
                rows = (src_chunk + 1) * chunk_rows - lo
                halo.append(_HaloPiece(src=_chunk_owner(src_chunk, cp_size), src_row=src_row, rows=rows))
            chunks.append((seq_row + slot * chunk_rows, chunk_rows, halo))
    return chunks


# enough for every micro-batch in flight; a miss only rebuilds the plan
@lru_cache(maxsize=32)
def plan_swa_halo(
    cu_seqlens: tuple[int, ...], *, cp_rank: int, cp_size: int, window: int, device: torch.device | str
) -> SwaHaloPlan:
    """The halo exchange of a THD stream with global boundaries ``cu_seqlens`` (padding included).

    Every sequence must be a multiple of ``2 * cp_size`` long, as Miles pads it. Pieces bound for one
    rank are sent in that rank's chunk order, which is the order it consumes them. Plans are memoized,
    so every layer and every recompute of a micro-batch shares one.
    """
    num_local = cu_seqlens[-1] // cp_size
    send_starts, send_lens, send_splits = [], [], [0] * cp_size
    for dst in range(cp_size):
        for _, _, halo in _local_chunks(cu_seqlens, cp_rank=dst, cp_size=cp_size, window=window):
            for piece in halo:
                if piece.src == cp_rank:
                    send_starts.append(piece.src_row)
                    send_lens.append(piece.rows)
                    send_splits[dst] += piece.rows

    chunks = _local_chunks(cu_seqlens, cp_rank=cp_rank, cp_size=cp_size, window=window)
    recv_splits = [0] * cp_size
    for _, _, halo in chunks:
        for piece in halo:
            recv_splits[piece.src] += piece.rows
    # next unread row of each source's block in cat([local, received])
    cursor = [num_local + sum(recv_splits[:src]) for src in range(cp_size)]
    kv_starts, kv_lens, q_lens, kv_seg_lens = [], [], [], []
    for row, rows, halo in chunks:
        for piece in halo:
            kv_starts.append(cursor[piece.src])
            kv_lens.append(piece.rows)
            cursor[piece.src] += piece.rows
        kv_starts.append(row)
        kv_lens.append(rows)
        q_lens.append(rows)
        kv_seg_lens.append(rows + sum(piece.rows for piece in halo))

    return SwaHaloPlan(
        send_rows=_concat_ranges(send_starts, send_lens, device),
        send_splits=tuple(send_splits),
        recv_splits=tuple(recv_splits),
        kv_rows=_concat_ranges(kv_starts, kv_lens, device),
        cu_seqlens_q=_boundaries(q_lens, device),
        cu_seqlens_kv=_boundaries(kv_seg_lens, device),
        max_seqlen_q=max(q_lens),
        max_seqlen_kv=max(kv_seg_lens),
    )


def _to_device(host: Tensor, device: torch.device | str) -> Tensor:
    if torch.device(device).type == "cuda":
        host = host.pin_memory()  # a copy from pageable memory would wait for the stream to drain
    return host.to(device, non_blocking=True)


def _concat_ranges(starts: list[int], lengths: list[int], device: torch.device | str) -> Tensor:
    pieces = [torch.arange(start, start + length) for start, length in zip(starts, lengths, strict=True)]
    return _to_device(torch.cat(pieces) if pieces else torch.empty(0, dtype=torch.int64), device)


def _boundaries(lengths: list[int], device: torch.device | str) -> Tensor:
    return _to_device(torch.tensor([0, *lengths], dtype=torch.int32).cumsum(0, dtype=torch.int32), device)


class _HaloAllToAll(torch.autograd.Function):
    """``all_to_all_single`` whose backward returns each received row's gradient to its sender."""

    @staticmethod
    def forward(ctx, rows, recv_splits, send_splits, cp_group):
        ctx.splits, ctx.cp_group = (recv_splits, send_splits), cp_group
        return _all_to_all(rows, recv_splits=recv_splits, send_splits=send_splits, cp_group=cp_group)

    @staticmethod
    def backward(ctx, grad):
        recv_splits, send_splits = ctx.splits
        return (
            _all_to_all(grad, recv_splits=send_splits, send_splits=recv_splits, cp_group=ctx.cp_group),
            None,
            None,
            None,
        )


def _all_to_all(rows: Tensor, *, recv_splits, send_splits, cp_group) -> Tensor:
    received = rows.new_empty(sum(recv_splits), *rows.shape[1:])
    dist.all_to_all_single(
        received,
        rows.contiguous(),
        output_split_sizes=list(recv_splits),
        input_split_sizes=list(send_splits),
        group=cp_group,
    )
    return received


def exchange_swa_halo(
    key: Tensor, value: Tensor, plan: SwaHaloPlan, cp_group: dist.ProcessGroup
) -> tuple[Tensor, Tensor]:
    """``[t, g, d]`` local keys and values -> the same rows plus their halos in ``plan.kv_rows`` order.

    Keys and values share one all-to-all; its backward returns each halo row's gradient to its owner.
    """
    key_width = key.flatten(1).shape[1]
    send = torch.cat([key[plan.send_rows].flatten(1), value[plan.send_rows].flatten(1)], dim=1)
    received = _HaloAllToAll.apply(send, plan.recv_splits, plan.send_splits, cp_group)
    halo_key, halo_value = received.split([key_width, send.shape[1] - key_width], dim=1)
    key = torch.cat([key, halo_key.reshape(-1, *key.shape[1:])])[plan.kv_rows]
    value = torch.cat([value, halo_value.reshape(-1, *value.shape[1:])])[plan.kv_rows]
    return key, value
