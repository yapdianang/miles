"""Per-sequence records of how the engine sampled, assembled per datum for training-time replay.

A record holds one sampled sequence: its output tokens, the sampling support (the top-k/top-p
candidate set) of each output token with the sampler's log-probabilities over it, and the
experts the engine routed each token through (R3).

Routes are a delta: the engine returns them only from ``routes_start``, the end of the
record's parent, which is the latest record of the same adapter whose prompt and output but
its last token are a prefix of this record's prompt. The engine records a cached prefix
token's routes once, when it first computes them, so the parent chain concatenates to the
routes of the full sequence. A new adapter version prefills afresh, so parents never cross
adapters.
"""

from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np

_HASH_BASE = 0x9E3779B97F4A7C15
_HASH_MODULUS = 1 << 64


def prefix_hashes(tokens: np.ndarray) -> np.ndarray:
    """``hashes[L - 1]`` hashes ``tokens[:L]``: sum of (token + 1) * B**position modulo 2**64."""
    powers = np.full(len(tokens), _HASH_BASE, dtype=np.uint64)
    powers[:1] = 1
    return np.cumsum((tokens.astype(np.uint64) + np.uint64(1)) * np.cumprod(powers), dtype=np.uint64)


def _extend_hash(prefix_hash: int, prefix_len: int, tokens: np.ndarray) -> int:
    """hash(prefix + tokens) from hash(prefix)."""
    if not len(tokens):
        return prefix_hash
    return (prefix_hash + pow(_HASH_BASE, prefix_len, _HASH_MODULUS) * int(prefix_hashes(tokens)[-1])) % _HASH_MODULUS


@dataclass(eq=False)
class SequenceRecord:
    sequence_id: str
    lora_name: str | None
    prompt_len: int
    tokens: np.ndarray
    # CSR over output tokens: token j drew from support_ids[support_offsets[j]:support_offsets[j + 1]]
    support_ids: np.ndarray | None = None
    support_offsets: np.ndarray | None = None
    support_log_probs: np.ndarray | None = None
    # [covered_len - routes_start, layers, topk]; positions before routes_start live in the parent chain
    routes: np.ndarray | None = None
    parent: "SequenceRecord | None" = None
    children: list["SequenceRecord"] = field(default_factory=list)

    @property
    def covered_len(self) -> int:
        """The engine ran the prompt and every output token but the last."""
        return self.prompt_len + len(self.tokens) - 1

    @property
    def routes_start(self) -> int:
        return self.parent.covered_len if self.parent is not None else 0

    @property
    def nbytes(self) -> int:
        arrays = (self.tokens, self.support_ids, self.support_offsets, self.support_log_probs, self.routes)
        return sum(array.nbytes for array in arrays if array is not None)

    def chain(self) -> Iterator["SequenceRecord"]:
        """This record and its ancestors, nearest first."""
        record = self
        while record is not None:
            yield record
            record = record.parent


class _KeyIndex:
    """Records by uint64 key; the keys stay sorted so a token array's prefixes are tested in one pass."""

    def __init__(self) -> None:
        self._records: dict[int, list[SequenceRecord]] = {}
        self._keys = np.empty(0, dtype=np.uint64)

    def add(self, key: int, record: SequenceRecord) -> None:
        bucket = self._records.setdefault(key, [])
        if not bucket:
            self._keys = np.insert(self._keys, np.searchsorted(self._keys, np.uint64(key)), np.uint64(key))
        bucket.append(record)

    def remove(self, key: int, record: SequenceRecord) -> None:
        bucket = self._records[key]
        bucket.remove(record)
        if not bucket:
            del self._records[key]
            self._keys = np.delete(self._keys, np.searchsorted(self._keys, np.uint64(key)))

    def get(self, key: int) -> list[SequenceRecord]:
        return self._records.get(key, [])

    def hits(self, hashes: np.ndarray) -> np.ndarray:
        """Indices ``i`` whose ``hashes[i]`` is a key, ascending."""
        if not len(self._keys):
            return np.empty(0, dtype=np.int64)
        found = self._keys[np.minimum(np.searchsorted(self._keys, hashes), len(self._keys) - 1)] == hashes
        return np.flatnonzero(found)


# (first target row, record, first record output position, rows)
Segment = tuple[int, SequenceRecord, int, int]


class SamplerRecordStore:
    """Records of recent samples by sequence id and by token prefix, evicted oldest chain first."""

    def __init__(self, max_bytes: int, *, supports: bool, routes: bool) -> None:
        self.max_bytes = max_bytes
        self.supports = supports
        self.routes = routes
        self.num_bytes = 0
        self._lru: OrderedDict[SequenceRecord, None] = OrderedDict()
        self._by_sequence: dict[str, SequenceRecord] = {}
        self._by_prompt = _KeyIndex()
        self._by_covered = _KeyIndex()
        self._keys: dict[SequenceRecord, tuple[int, int]] = {}

    def __len__(self) -> int:
        return len(self._lru)

    def get(self, sequence_id: str) -> SequenceRecord | None:
        return self._by_sequence.get(sequence_id)

    def route_parent(self, prompt_hashes: np.ndarray, lora_name: str | None) -> SequenceRecord | None:
        """The latest routed record of this adapter covering the longest prefix of the prompt."""
        for index in self._by_covered.hits(prompt_hashes)[::-1].tolist():
            for record in reversed(self._by_covered.get(int(prompt_hashes[index]))):
                if record.lora_name == lora_name and record.covered_len == index + 1 and record.routes is not None:
                    return record
        return None

    def put(self, record: SequenceRecord, prompt_hash: int) -> None:
        if record.parent is not None and record.parent not in self._lru:
            # evicted while the engine sampled: the routes before routes_start are gone
            record.parent, record.routes = None, None
        if record.parent is not None:
            record.parent.children.append(record)
            for ancestor in reversed(list(record.parent.chain())):
                self._lru.move_to_end(ancestor)
        covered_hash = _extend_hash(prompt_hash, record.prompt_len, record.tokens[:-1])
        self._keys[record] = (prompt_hash, covered_hash)
        self._by_prompt.add(prompt_hash, record)
        self._by_covered.add(covered_hash, record)
        self._by_sequence[record.sequence_id] = record
        self._lru[record] = None
        self.num_bytes += record.nbytes
        while self.num_bytes > self.max_bytes and len(self._lru) > 1:
            self._drop_descendants(next(iter(self._lru)))

    def _drop_descendants(self, root: SequenceRecord) -> None:
        """Evict a record with every record whose routes build on it."""
        if root.parent is not None:
            root.parent.children.remove(root)
        stack = [root]
        while stack:
            record = stack.pop()
            stack.extend(record.children)
            record.children.clear()
            prompt_hash, covered_hash = self._keys.pop(record)
            self._by_prompt.remove(prompt_hash, record)
            self._by_covered.remove(covered_hash, record)
            if self._by_sequence.get(record.sequence_id) is record:
                del self._by_sequence[record.sequence_id]
            del self._lru[record]
            self.num_bytes -= record.nbytes

    # -------- per-datum assembly --------

    def datum_routes(self, datum: dict, hashes: np.ndarray) -> np.ndarray | None:
        """Routes of ``tokens[:-1]`` from the parent chain of the sample that ends the datum, if complete."""
        num_inputs = datum["target_len"]
        if not num_inputs:
            return None
        record = self._provenance_last_record(datum)
        if record is None or record.covered_len != num_inputs or record.routes is None:
            matches = [
                candidate
                for candidate in self._by_covered.get(int(hashes[-1]))
                if candidate.covered_len == num_inputs and candidate.routes is not None
            ]
            record = matches[-1] if matches else None
        if record is None:
            return None
        routes = np.concatenate([link.routes for link in reversed(list(record.chain()))])
        return routes if len(routes) == num_inputs else None

    def datum_supports(self, datum: dict, hashes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """CSR ``(ids, offsets, log_probs)`` over the datum's targets; rows the engine did not sample are empty."""
        targets = np.asarray(datum["target_tokens"], dtype=np.int64)
        if datum.get("provenance") is not None:
            segments = self._provenance_segments(datum["provenance"][1], targets)
        else:
            segments = self._prefix_segments(np.asarray(datum["tokens"], dtype=np.int64), targets, hashes)
        return _assemble(len(targets), segments)

    def _provenance_last_record(self, datum: dict) -> SequenceRecord | None:
        """The sample whose output ends the datum, as the client's provenance names it."""
        if datum.get("provenance") is None:
            return None
        kind, sequence_id, offset, length = datum["provenance"][1][-1]
        record = self._by_sequence.get(sequence_id)
        if kind != "sampled" or record is None or offset + length != len(record.tokens):
            return None
        return record

    def _provenance_segments(self, loss_spans: list[tuple], targets: np.ndarray) -> list[Segment]:
        """Target rows a sampled span claims, where the target is the sampled token."""
        segments, row = [], 0
        for kind, sequence_id, offset, length in loss_spans:
            record = self._by_sequence.get(sequence_id)
            if kind == "sampled" and record is not None and record.support_ids is not None:
                count = max(0, min(length, len(record.tokens) - offset))
                agrees = targets[row : row + count] == record.tokens[offset : offset + count]
                for start, end in _true_runs(agrees):
                    segments.append((row + start, record, offset + start, end - start))
            row += length
        return segments

    def _prefix_segments(self, tokens: np.ndarray, targets: np.ndarray, hashes: np.ndarray) -> list[Segment]:
        """Target rows of every sample whose prompt is a prefix of the datum, later prompts winning overlaps.

        A sample with prompt ``tokens[:p]`` covers target ``p - 1 + j`` while its ``out[: j + 1]`` matches
        the datum's tokens and targets.
        """
        segments: list[Segment] = []
        for index in self._by_prompt.hits(hashes).tolist():
            prompt_len = index + 1
            record, count = _longest_match(self._by_prompt.get(int(hashes[index])), prompt_len, tokens, targets)
            if count == 0:
                continue
            start = prompt_len - 1
            if segments and segments[-1][0] + segments[-1][3] > start:
                first, previous, previous_start, _ = segments[-1]
                segments[-1] = (first, previous, previous_start, start - first)
            segments.append((start, record, 0, count))
        return segments


def _longest_match(
    records: list[SequenceRecord], prompt_len: int, tokens: np.ndarray, targets: np.ndarray
) -> tuple[SequenceRecord | None, int]:
    """The sample of this prompt whose output agrees longest with the datum's tokens (ties go to the latest),
    and how many targets it covers: coverage also stops where a target leaves the sampled output."""
    best, best_count, available = None, 0, 0
    for record in records:
        if record.prompt_len != prompt_len or record.support_ids is None:
            continue
        record_available = min(len(record.tokens), len(targets) - prompt_len + 1)
        count = _agreement(record.tokens[:record_available], tokens[prompt_len : prompt_len + record_available])
        if count >= best_count:
            best, best_count, available = record, count, record_available
    if best is None:
        return None, 0
    return best, min(
        best_count, _agreement(best.tokens[:available], targets[prompt_len - 1 : prompt_len - 1 + available])
    )


def _agreement(first: np.ndarray, second: np.ndarray) -> int:
    """Length of the common prefix of two equal-length arrays."""
    differs = np.flatnonzero(first != second)
    return int(differs[0]) if differs.size else len(first)


def _true_runs(values: np.ndarray) -> list[tuple[int, int]]:
    """``[start, end)`` of each run of True."""
    edges = np.flatnonzero(np.diff(np.concatenate(([False], values, [False])).astype(np.int8)))
    return list(zip(edges[::2].tolist(), edges[1::2].tolist(), strict=True))


def _assemble(num_targets: int, segments: list[Segment]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lengths = np.zeros(num_targets, dtype=np.int64)
    for row, record, start, count in segments:
        lengths[row : row + count] = np.diff(record.support_offsets[start : start + count + 1])
    offsets = np.zeros(num_targets + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    ids = np.empty(offsets[-1], dtype=np.int32)
    log_probs = np.empty(offsets[-1], dtype=np.float32)
    for row, record, start, count in segments:
        source = slice(record.support_offsets[start], record.support_offsets[start + count])
        ids[offsets[row] : offsets[row + count]] = record.support_ids[source]
        log_probs[offsets[row] : offsets[row + count]] = record.support_log_probs[source]
    return ids, offsets, log_probs


def parse_supports(
    output_tokens: list[int], supports: list[list[int]], support_log_probs: list[list[float]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[float]]:
    """SGLang support-mode rows -> CSR ``(ids, offsets, log_probs)`` and each sampled token's log-probability."""
    if len(supports) != len(output_tokens) or len(support_log_probs) != len(output_tokens):
        raise ValueError(f"{len(supports)} sampling supports for {len(output_tokens)} output tokens")
    lengths = np.fromiter(map(len, supports), dtype=np.int64, count=len(supports))
    if (lengths == 0).any() or (
        np.fromiter(map(len, support_log_probs), dtype=np.int64, count=len(lengths)) != lengths
    ).any():
        raise ValueError("sampling supports and their log-probabilities must be non-empty and aligned")
    offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    ids = np.fromiter((token for row in supports for token in row), dtype=np.int32, count=int(offsets[-1]))
    log_probs = np.fromiter((value for row in support_log_probs for value in row), dtype=np.float32, count=len(ids))
    sampled = ids == np.repeat(np.asarray(output_tokens, dtype=np.int32), lengths)
    if sampled.sum() != len(lengths) or (len(lengths) and not np.logical_or.reduceat(sampled, offsets[:-1]).all()):
        raise ValueError("each sampled token must occur exactly once in its sampling support")
    return ids, offsets, log_probs, log_probs[sampled].astype(np.float64).tolist()
