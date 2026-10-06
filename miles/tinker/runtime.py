"""Translate gateway datums to trainer batches and sampling requests to SGLang."""

import asyncio
import hashlib
import itertools
import logging
import os
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
import pybase64
import torch

from miles.ray.rollout.train_data_conversion import ROLLOUT_DATA_VALUE_SPEC
from miles.tinker.core.types import UserInputError
from miles.utils import object_store
from miles.utils.http_utils import post
from tinker.types.topk_logprobs import MASK_LOGPROB

logger = logging.getLogger(__name__)

# internal datum key -> trainer batch key
DATUM_TO_BATCH_KEYS = {"weights": "loss_weights", "advantages": "advantages", "sampling_logprobs": "rollout_log_probs"}


def _tokens_key(tokens: list[int]) -> bytes:
    return hashlib.blake2b(np.asarray(tokens, dtype=np.int32).tobytes(), digest_size=16).digest()


class RoutedExpertsCache:
    """Engine-routed experts of recent samples, keyed by the tokens a datum feeds the trainer.

    A sample of ``prompt`` with output ``out`` routes ``prompt + out[:-1]``: a datum's tokens without
    its last target. Entries hold int16 expert ids (``[tokens, layers, topk]``); the oldest
    are evicted first, so a multi-turn trajectory's earlier calls leave before its last one.
    """

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max_bytes
        self.num_bytes = 0
        self._entries: OrderedDict[bytes, np.ndarray] = OrderedDict()

    def put(self, tokens: list[int], routes: np.ndarray) -> None:
        key = _tokens_key(tokens)
        if key in self._entries:
            self.num_bytes -= self._entries.pop(key).nbytes
        self._entries[key] = routes
        self.num_bytes += routes.nbytes
        while self.num_bytes > self.max_bytes and len(self._entries) > 1:
            self.num_bytes -= self._entries.popitem(last=False)[1].nbytes

    def get(self, tokens: list[int]) -> np.ndarray | None:
        return self._entries.get(_tokens_key(tokens))


_PREFIX_HASH_BASE = np.uint64(0x9E3779B97F4A7C15)


def _prefix_hashes(tokens: np.ndarray) -> np.ndarray:
    """``hashes[L - 1]`` hashes ``tokens[:L]``: a polynomial in a fixed odd base modulo 2**64."""
    powers = np.full(len(tokens), _PREFIX_HASH_BASE, dtype=np.uint64)
    powers[:1] = 1
    return np.cumsum((tokens.astype(np.uint64) + np.uint64(1)) * np.cumprod(powers), dtype=np.uint64)


@dataclass(frozen=True, eq=False)
class _SampledSupports:
    """One sample's output tokens and, per token, the support (CSR ``ids``/``offsets``) it was drawn from."""

    prompt_len: int
    tokens: np.ndarray
    ids: np.ndarray
    offsets: np.ndarray

    @property
    def nbytes(self) -> int:
        return self.tokens.nbytes + self.ids.nbytes + self.offsets.nbytes


class SamplingSupportCache:
    """Engine sampling supports (top-k/top-p candidate sets) of recent samples, per sampled token.

    A sample of ``prompt`` drew ``out[j]`` from the support recorded for ``prompt + out[:j]``. A
    multi-turn datum spans several samples (each turn's prompt extends the previous prompt and
    output), so samples are keyed by a hash of their prompt and a datum finds every sample whose
    prompt is one of its prefixes in one pass. The oldest samples are evicted first.
    """

    def __init__(self, max_bytes: int) -> None:
        self.max_bytes = max_bytes
        self.num_bytes = 0
        self._entries: OrderedDict[_SampledSupports, int] = OrderedDict()
        self._by_prompt: dict[int, list[_SampledSupports]] = {}
        self._sorted_keys: np.ndarray | None = None

    def put(self, prompt_tokens: list[int], samples: list[tuple[list[int], list[list[int]]]]) -> None:
        """Record ``(output tokens, per-token supports)`` of each sample of one prompt."""
        if not prompt_tokens:
            return
        key = int(_prefix_hashes(np.asarray(prompt_tokens, dtype=np.int64))[-1])
        for output_tokens, supports in samples:
            entry = _sampled_supports(len(prompt_tokens), output_tokens, supports)
            self._by_prompt.setdefault(key, []).append(entry)
            self._entries[entry] = key
            self.num_bytes += entry.nbytes
        self._sorted_keys = None
        while self.num_bytes > self.max_bytes and len(self._entries) > 1:
            entry, key = self._entries.popitem(last=False)
            self.num_bytes -= entry.nbytes
            bucket = self._by_prompt[key]
            bucket.remove(entry)
            if not bucket:
                del self._by_prompt[key]

    def lookup(self, tokens: list[int], target_tokens: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """A datum's supports as CSR ``(ids, offsets)`` over its targets; positions no sample covers are empty.

        Target ``i`` scores ``tokens[i + 1]`` after ``tokens[: i + 1]``. A sample with prompt
        ``tokens[:p]`` covers target ``p - 1 + j`` while its ``out[: j + 1]`` matches the datum's tokens and
        targets; where samples overlap, the one with the longer prompt (the later call) wins.
        """
        num_targets = len(target_tokens)
        tokens = np.asarray(tokens, dtype=np.int64)
        targets = np.asarray(target_tokens, dtype=np.int64)
        segments: list[tuple[int, _SampledSupports, int]] = []
        if self._by_prompt:
            hashes = _prefix_hashes(tokens[:num_targets])
            keys = self._keys()
            found = keys[np.minimum(np.searchsorted(keys, hashes), len(keys) - 1)] == hashes
            for prompt_len in (np.flatnonzero(found) + 1).tolist():
                entry, count = self._longest_match(int(hashes[prompt_len - 1]), prompt_len, tokens, targets)
                if count == 0:
                    continue
                start = prompt_len - 1
                if segments and segments[-1][0] + segments[-1][2] > start:
                    first, previous, _ = segments[-1]
                    segments[-1] = (first, previous, start - first)
                segments.append((start, entry, count))
        lengths = np.zeros(num_targets, dtype=np.int64)
        for start, entry, count in segments:
            lengths[start : start + count] = np.diff(entry.offsets[: count + 1])
        offsets = np.zeros(num_targets + 1, dtype=np.int64)
        np.cumsum(lengths, out=offsets[1:])
        ids = np.empty(offsets[-1], dtype=np.int32)
        for start, entry, count in segments:
            ids[offsets[start] : offsets[start + count]] = entry.ids[: entry.offsets[count]]
        return ids, offsets

    def _keys(self) -> np.ndarray:
        if self._sorted_keys is None:
            self._sorted_keys = np.sort(np.fromiter(self._by_prompt, dtype=np.uint64, count=len(self._by_prompt)))
        return self._sorted_keys

    def _longest_match(
        self, key: int, prompt_len: int, tokens: np.ndarray, targets: np.ndarray
    ) -> tuple[_SampledSupports | None, int]:
        """The sample of this prompt whose output agrees longest with the datum's tokens (ties go to the latest),
        and how many of its targets it covers: coverage also stops where a target leaves the sampled output."""
        best, best_count, available = None, 0, 0
        for entry in self._by_prompt[key]:
            if entry.prompt_len != prompt_len:
                continue
            entry_available = min(len(entry.tokens), len(targets) - prompt_len + 1)
            count = _agreement(entry.tokens[:entry_available], tokens[prompt_len : prompt_len + entry_available])
            if count >= best_count:
                best, best_count, available = entry, count, entry_available
        if best is None:
            return None, 0
        return best, min(
            best_count, _agreement(best.tokens[:available], targets[prompt_len - 1 : prompt_len - 1 + available])
        )


def _agreement(first: np.ndarray, second: np.ndarray) -> int:
    """Length of the common prefix of two equal-length arrays."""
    differs = np.flatnonzero(first != second)
    return int(differs[0]) if differs.size else len(first)


def _sampled_supports(prompt_len: int, output_tokens: list[int], supports: list[list[int]]) -> _SampledSupports:
    if len(supports) != len(output_tokens):
        raise ValueError(f"{len(supports)} sampling supports for {len(output_tokens)} output tokens")
    lengths = np.fromiter(map(len, supports), dtype=np.int64, count=len(supports))
    if (lengths == 0).any():
        raise ValueError("an empty sampling support")
    offsets = np.zeros(len(supports) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    ids = np.fromiter(itertools.chain.from_iterable(supports), dtype=np.int32, count=int(offsets[-1]))
    tokens = np.asarray(output_tokens, dtype=np.int32)
    if len(tokens) and not np.logical_or.reduceat(ids == np.repeat(tokens, lengths), offsets[:-1]).all():
        raise ValueError("a sampled token is outside its sampling support")
    return _SampledSupports(prompt_len, tokens, ids, offsets)


def _write_exported_checkpoint(path: str, checkpoint_files: dict[str, bytes]) -> None:
    """Publish returned trainer files safely when the checkpoint mount is shared."""
    checkpoint = Path(path)
    checkpoint.mkdir(parents=True, exist_ok=True)
    for name, contents in checkpoint_files.items():
        destination = checkpoint / name
        if destination.parent != checkpoint:
            raise ValueError(f"checkpoint file name must be flat: {name!r}")
        descriptor, temporary_path = tempfile.mkstemp(prefix=f".{name}.", dir=checkpoint)
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(contents)
            os.replace(temporary_path, destination)
        except Exception:
            Path(temporary_path).unlink(missing_ok=True)
            raise


def _pad_to_dp_multiple(slot_datums: list, dp_size: int) -> list:
    """Equalize DP shares with zero-mask datums; their outputs are dropped after the loss pass."""
    remainder = len(slot_datums) % dp_size
    if remainder == 0:
        return slot_datums
    slot, datum = slot_datums[-1]
    filler = dict(datum) | {"padding": True}
    return slot_datums + [(slot, filler)] * (dp_size - remainder)


def _build_train_data(
    slot_datums: list,
    routed_experts: RoutedExpertsCache | None = None,
    sampling_supports: SamplingSupportCache | None = None,
) -> dict:
    """Miles response_lengths select label positions here, including prompt targets."""
    datums = [datum for _, datum in slot_datums]
    train_data = {
        "tokens": [datum["tokens"] for datum in datums],
        "target_tokens": [datum["target_tokens"] for datum in datums],
        "loss_masks": [[0 if datum.get("padding") else 1] * datum["target_len"] for datum in datums],
        "response_lengths": [datum["target_len"] for datum in datums],
        "total_lengths": [len(datum["tokens"]) for datum in datums],
        "sample_indices": list(range(len(datums))),
        "adapter_slots": [slot for slot, _ in slot_datums],
        "dynamic_global_batch_size": len(datums),
    }
    for datum_key, batch_key in DATUM_TO_BATCH_KEYS.items():
        if datum_key in datums[0]:
            train_data[batch_key] = [datum[datum_key] for datum in datums]
    if routed_experts is not None:
        routes = [routed_experts.get(datum["tokens"][:-1]) for datum in datums]
        missing = sum(route is None for route in routes)
        if missing:
            # Replay is all or nothing per pass; a datum the engine did not sample trains on its own routing.
            logger.warning(f"routing replay skipped: {missing}/{len(routes)} datums have no engine routes")
        else:
            train_data["rollout_routed_experts"] = [route.astype(np.int32) for route in routes]
    if sampling_supports is not None:
        supports = [sampling_supports.lookup(datum["tokens"], datum["target_tokens"]) for datum in datums]
        _warn_unsupported_loss_positions(datums, supports)
        if any(offsets[-1] for _, offsets in supports):
            train_data["rollout_sampling_mask_ids"] = [torch.from_numpy(ids) for ids, _ in supports]
            train_data["rollout_sampling_mask_offsets"] = [torch.from_numpy(offsets) for _, offsets in supports]
    return train_data


def _warn_unsupported_loss_positions(datums: list[dict], supports: list[tuple[np.ndarray, np.ndarray]]) -> None:
    """Loss positions without an engine support normalize over the full vocabulary; report how many."""
    num_missing = num_loss = num_datums = 0
    for datum, (_, offsets) in zip(datums, supports, strict=True):
        if datum.get("padding"):
            continue
        loss = np.zeros(datum["target_len"], dtype=bool)
        for key in ("weights", "advantages"):
            if key in datum:
                loss |= np.asarray(datum[key]) != 0
        missing = int((loss & (np.diff(offsets) == 0)).sum())
        num_missing += missing
        num_loss += int(loss.sum())
        num_datums += missing > 0
    if num_missing:
        logger.warning(
            f"sampling-support replay: {num_missing}/{num_loss} loss positions in {num_datums} datums have no "
            "engine support and normalize over the full vocabulary"
        )


class MilesBackend:
    def __init__(
        self,
        trainer,
        router_url: str,
        dp_size: int = 1,
        inference_controller=None,
        routed_experts: RoutedExpertsCache | None = None,
        num_layers: int | None = None,
        sampling_supports: SamplingSupportCache | None = None,
    ) -> None:
        self.trainer = trainer
        self.router_url = router_url
        self.dp_size = dp_size
        self.inference_controller = inference_controller
        self.routed_experts = routed_experts
        self.num_layers = num_layers
        self.sampling_supports = sampling_supports

    async def trainer_dead(self) -> bool:
        return await self.trainer.has_errored_cell()

    async def load_slot(self, slot: int, rank: int, alpha: float, ckpt_path: str | None = None, load_optimizer: bool = True) -> dict | None:
        return _slot_failure(await self.trainer.load_slot(slot, rank, alpha, ckpt_path=ckpt_path, load_optimizer=load_optimizer))

    async def unload_slot(self, slot: int) -> dict | None:
        return _slot_failure(await self.trainer.unload_slot(slot))

    async def forward_backward(self, batch_id: int, slot_datums: list, loss_fn: str, loss_fn_config: dict) -> list[dict] | dict:
        return await self._execute_batch("forward_backward", batch_id, slot_datums, loss_fn, loss_fn_config)

    async def forward_only(self, batch_id: int, slot_datums: list, loss_fn: str, loss_fn_config: dict) -> list[dict] | dict:
        return await self._execute_batch("forward_only", batch_id, slot_datums, loss_fn, loss_fn_config)

    async def _execute_batch(self, method: str, batch_id: int, slot_datums: list, loss_fn: str, loss_fn_config: dict) -> list[dict] | dict:
        train_data = _build_train_data(
            _pad_to_dp_multiple(slot_datums, self.dp_size), self.routed_experts, self.sampling_supports
        )
        train_data["loss_fn"] = loss_fn
        train_data["loss_fn_config"] = loss_fn_config
        worker_results = await self._call_trainer(method, batch_id, train_data)
        by_index: dict[int, dict] = {}
        for worker_result in worker_results:
            if "error" in worker_result:
                return worker_result
            for datum_output in worker_result["per_datum"]:
                index = int(datum_output["sample_index"])
                if index not in by_index:
                    by_index[index] = {
                        "loss": float(datum_output["loss"]),
                        "logprobs": datum_output["logprobs"].tolist(),
                    }
        return [by_index[index] for index in range(len(slot_datums))]

    async def _call_trainer(self, method: str, batch_id: int, train_data: dict) -> list:
        store = object_store.get_instance()
        data_ref = store.put(value=train_data, value_spec=ROLLOUT_DATA_VALUE_SPEC)
        try:
            return await getattr(self.trainer, method)(batch_id=batch_id, data_ref=data_ref)
        finally:
            store.remove(data_ref)

    async def optim_step(self, adam_params_by_slot: dict[int, dict]) -> dict[int, dict]:
        worker_results = await self.trainer.optim_step(adam_params_by_slot=adam_params_by_slot)
        return worker_results[0]

    async def save_slot(self, slot: int, path: str, metadata: dict | None = None) -> dict | None:
        return _slot_failure(await self.trainer.save_slot(slot=slot, path=path, metadata=metadata))

    async def export_slot(self, slot: int, rank: int, alpha: float, path: str, metadata: dict | None = None) -> dict | None:
        worker_results = await self.trainer.export_slot(slot=slot, rank=rank, alpha=alpha, path=path, metadata=metadata)
        failure = next(
            (result for result in worker_results if result is not None and "error" in result),
            None,
        )
        if failure is not None or self.inference_controller is None:
            return failure
        checkpoint_files = next(result["checkpoint_files"] for result in worker_results if result is not None and "checkpoint_files" in result)

        _write_exported_checkpoint(path, checkpoint_files)
        checkpoint = Path(path)
        assert checkpoint.parent.name == "sampler_weights", f"unexpected sampler checkpoint path: {path}"
        lora_name = f"{checkpoint.parent.parent.name}@{checkpoint.name}"
        await self.inference_controller.load_lora_adapter(lora_name=lora_name, lora_path=path)
        return None

    # -------- sampling --------

    async def sample(self, payload: dict, lora_name: str | None, lora_path: str | None = None) -> dict:
        request = self._generate_request(payload, lora_name, lora_path)
        try:
            responses = await asyncio.gather(*[post(f"{self.router_url}/generate", _with_sample_seed(request, index)) for index in range(payload["num_samples"])])
        except httpx.HTTPError as error:
            return {"error": str(error)}
        sequences = [_to_sequence(response) for response in responses]
        for sequence in sequences:
            if "error" in sequence:
                return sequence
        if self.routed_experts is not None:
            for sequence, response in zip(sequences, responses, strict=True):
                self._cache_routes(payload["prompt_tokens"], sequence["tokens"], response)
        if request.get("return_sampling_mask"):
            try:
                self._cache_supports(payload["prompt_tokens"], sequences, responses)
            except (KeyError, ValueError) as error:
                return {"error": f"the engine returned no valid sampling supports: {error!r}"}
        result = {"sequences": sequences}
        if payload["prompt_logprobs"]:
            result["prompt_logprobs"] = _prompt_logprobs(responses[0])
        if payload["topk_prompt_logprobs"]:
            result["topk_prompt_logprobs"] = _topk_prompt_logprobs(responses[0], payload["topk_prompt_logprobs"])
        return result

    def _cache_routes(self, prompt_tokens: list[int], output_tokens: list[int], response: dict) -> None:
        tokens = prompt_tokens + output_tokens[:-1]
        encoded = response["meta_info"]["routed_experts"]
        routes = np.frombuffer(pybase64.b64decode(encoded.encode("ascii")), dtype=np.int32)
        routes = routes.reshape(len(tokens), self.num_layers, -1)
        self.routed_experts.put(tokens, routes.astype(np.int16))

    def _cache_supports(self, prompt_tokens: list[int], sequences: list[dict], responses: list[dict]) -> None:
        """Record each sample's supports and return its logprobs renormalized within them, as the trainer scores them."""
        samples = []
        for sequence, response in zip(sequences, responses, strict=True):
            meta = response["meta_info"]
            logprobs = [float(logprob) for logprob in meta["output_token_sampling_logprobs"]]
            if len(logprobs) != len(sequence["tokens"]):
                raise ValueError(f"{len(logprobs)} sampling logprobs for {len(sequence['tokens'])} output tokens")
            sequence["logprobs"] = logprobs
            samples.append((sequence["tokens"], meta["output_token_sampling_mask"]))
        self.sampling_supports.put(prompt_tokens, samples)

    def _generate_request(self, payload: dict, lora_name: str | None, lora_path: str | None = None) -> dict:
        params = payload["sampling_params"]
        max_tokens = params.get("max_tokens")
        if max_tokens is None:
            raise UserInputError("sampling_params.max_tokens is required")
        sampling_params = {
            "max_new_tokens": max_tokens,
            "temperature": params.get("temperature", 1.0),
            "top_p": params.get("top_p", 1.0),
            "top_k": params.get("top_k", -1),
        }
        if params.get("seed") is not None:
            sampling_params["sampling_seed"] = params["seed"]
        stop = params.get("stop")
        if stop is not None:
            if stop == []:
                # tinker defines stop=[] as disabling every stop token, EOS included
                sampling_params["ignore_eos"] = True
            elif isinstance(stop, list) and isinstance(stop[0], int):
                sampling_params["stop_token_ids"] = stop
            else:
                sampling_params["stop"] = stop
        request = {"input_ids": payload["prompt_tokens"], "sampling_params": sampling_params, "return_logprob": True}
        if self.routed_experts is not None:
            request["return_routed_experts"] = True
        if self.sampling_supports is not None and _draws_from_support(sampling_params):
            request["return_sampling_mask"] = True
        if payload["prompt_logprobs"] or payload["topk_prompt_logprobs"]:
            request["logprob_start_len"] = 0
        if payload["topk_prompt_logprobs"]:
            request["top_logprobs_num"] = payload["topk_prompt_logprobs"]
        if lora_name is not None:
            request["lora_path"] = lora_name
            if lora_path is not None:
                # request-carried backfill source: the engine refills an evicted version itself
                request["lora_backfill_paths"] = {lora_name: lora_path}
        return request


def _draws_from_support(sampling_params: dict) -> bool:
    """Whether the engine samples from a top-k/top-p support narrower than the vocabulary."""
    top_k = sampling_params["top_k"]
    if sampling_params["temperature"] <= 0 or top_k == 1:
        return False
    if top_k > 1:
        return True
    if sampling_params["top_p"] < 1:
        raise UserInputError(
            "top_p below 1 needs a top_k bound: the engine returns sampling supports only under a finite top_k"
        )
    return False


def _slot_failure(worker_results: list) -> dict | None:
    return next((result for result in worker_results if result is not None), None)


def _with_sample_seed(request: dict, index: int) -> dict:
    """Give each sample a distinct seed when the caller pins the request seed."""
    request = dict(request)
    params = dict(request["sampling_params"])
    if (seed := params.get("sampling_seed")) is not None:
        params["sampling_seed"] = seed + index
    request["sampling_params"] = params
    return request


def _prompt_logprobs(response: dict) -> list[float]:
    entries = response["meta_info"]["input_token_logprobs"]
    return [float("nan") if entry[0] is None else float(entry[0]) for entry in entries]


def _topk_prompt_logprobs(response: dict, k: int) -> dict:
    token_ids, logprobs = [], []
    for position in response["meta_info"]["input_top_logprobs"]:
        candidates = position or []
        candidate_token_ids = [entry[1] for entry in candidates][:k]
        candidate_logprobs = [float(entry[0]) for entry in candidates][:k]
        token_ids.append(candidate_token_ids + [0] * (k - len(candidate_token_ids)))
        logprobs.append(candidate_logprobs + [MASK_LOGPROB] * (k - len(candidate_logprobs)))
    return {"token_ids": token_ids, "logprobs": logprobs}


def _to_sequence(response: dict) -> dict:
    output_token_logprobs = response["meta_info"]["output_token_logprobs"]
    finish = response["meta_info"]["finish_reason"]["type"]
    if finish == "abort":
        # a truncated sequence must fail the request, not pass as a completed sample
        return {"error": "the engine aborted this sample; resubmit the request"}
    return {
        "tokens": [entry[1] for entry in output_token_logprobs],
        "logprobs": [entry[0] for entry in output_token_logprobs],
        "stop_reason": "length" if finish == "length" else "stop",
    }
