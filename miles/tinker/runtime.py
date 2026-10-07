"""Translate gateway datums to trainer batches and sampling requests to SGLang."""

import asyncio
import hashlib
import logging
import os
import tempfile
import uuid
from collections import OrderedDict
from pathlib import Path

import httpx
import numpy as np
import pybase64
import torch

from miles.ray.rollout.train_data_conversion import ROLLOUT_DATA_VALUE_SPEC
from miles.tinker.core.types import EngineUnavailableError, UserInputError
from miles.tinker.engine_load import EngineLoad, EngineLoadMonitor
from miles.tinker.rank_affinity import RankAffinity
from miles.tinker.sampler_records import SamplerRecordStore, SequenceRecord, parse_supports, prefix_hashes
from miles.utils import object_store
from miles.utils.http_utils import post
from tinker.types.topk_logprobs import MASK_LOGPROB

logger = logging.getLogger(__name__)

# A pinned engine that keeps failing hands its sample to the router after this many attempts.
_PINNED_ENGINE_ATTEMPTS = 3

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
    sampler_records: SamplerRecordStore | None = None,
    loss_fn: str | None = None,
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
    hashes = []
    if sampler_records is not None:
        hashes = [
            prefix_hashes(np.asarray(datum["tokens"][: datum["target_len"]], dtype=np.int64)) for datum in datums
        ]
    routes = None
    if routed_experts is not None:
        routes = [routed_experts.get(datum["tokens"][:-1]) for datum in datums]
    elif sampler_records is not None and sampler_records.routes:
        routes = [
            sampler_records.datum_routes(datum, datum_hashes) for datum, datum_hashes in zip(datums, hashes, strict=True)
        ]
    if routes is not None:
        missing = sum(route is None for route in routes)
        if missing:
            # Replay is all or nothing per pass; a datum the engine did not sample trains on its own routing.
            logger.warning(f"routing replay skipped: {missing}/{len(routes)} datums have no engine routes")
        else:
            train_data["rollout_routed_experts"] = [route.astype(np.int32) for route in routes]
    if loss_fn == "score_centering" and (sampler_records is None or not sampler_records.supports):
        logger.warning("score_centering without recorded sampling supports has no correction term")
    if sampler_records is not None and sampler_records.supports:
        supports = [
            sampler_records.datum_supports(datum, datum_hashes)
            for datum, datum_hashes in zip(datums, hashes, strict=True)
        ]
        _warn_unsupported_loss_positions(datums, supports)
        if any(offsets[-1] for _, offsets, _ in supports):
            train_data["rollout_sampling_mask_ids"] = [torch.from_numpy(ids) for ids, _, _ in supports]
            train_data["rollout_sampling_mask_offsets"] = [torch.from_numpy(offsets) for _, offsets, _ in supports]
            if loss_fn == "score_centering":
                train_data["rollout_sampling_mask_log_probs"] = [torch.from_numpy(values) for _, _, values in supports]
    return train_data


def _warn_unsupported_loss_positions(datums: list[dict], supports: list[tuple[np.ndarray, ...]]) -> None:
    """Loss positions without an engine support normalize over the full vocabulary; report how many."""
    num_missing = num_loss = num_datums = 0
    for datum, (_, offsets, _) in zip(datums, supports, strict=True):
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
        sampler_records: SamplerRecordStore | None = None,
        rank_affinity: RankAffinity | None = None,
    ) -> None:
        self.trainer = trainer
        self.router_url = router_url
        self.dp_size = dp_size
        self.inference_controller = inference_controller
        self.routed_experts = routed_experts
        self.num_layers = num_layers
        self.sampler_records = sampler_records
        self.rank_affinity = rank_affinity
        self.engine_load_monitor = EngineLoadMonitor(router_url)

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
            _pad_to_dp_multiple(slot_datums, self.dp_size), self.routed_experts, self.sampler_records, loss_fn
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

    async def engine_loads(self) -> list[EngineLoad]:
        return await self.engine_load_monitor.get_loads()

    async def _choose_engine(self, prompt_tokens: list[int]) -> str:
        """The engine holding this rollout's context, or the router when affinity is off or load is unknown."""
        if self.rank_affinity is None:
            return self.router_url
        try:
            loads = await self.engine_loads()
        except EngineUnavailableError as error:
            logger.warning(f"rank affinity skipped: {error}")
            return self.router_url
        if not loads:
            return self.router_url
        return self.rank_affinity.choose_engine(prompt_tokens, loads)

    async def _generate(self, prompt_tokens: list[int], request: dict, num_samples: int) -> tuple[str, list[dict]]:
        """Sample on the rollout's engine; when that engine keeps failing, the router places the request."""
        requests = [_with_sample_seed(request, index) for index in range(num_samples)]
        engine_url = await self._choose_engine(prompt_tokens)
        if engine_url != self.router_url:
            try:
                return engine_url, await asyncio.gather(
                    *[post(f"{engine_url}/generate", item, max_retries=_PINNED_ENGINE_ATTEMPTS) for item in requests]
                )
            except httpx.HTTPError as error:
                logger.warning(f"rank affinity: {engine_url} failed ({error}); the router places this sample")
        return self.router_url, await asyncio.gather(*[post(f"{self.router_url}/generate", item) for item in requests])

    async def sample(
        self, payload: dict, lora_name: str | None, lora_path: str | None = None, sequence_ids: list[str] | None = None
    ) -> dict:
        request = self._generate_request(payload, lora_name, lora_path)
        records, parent = self.sampler_records, None
        if records is not None and payload["prompt_tokens"]:
            prompt_hashes = prefix_hashes(np.asarray(payload["prompt_tokens"], dtype=np.int64))
            if records.routes:
                # the engine returns routes only past the prompt prefix an earlier sample already recorded
                parent = records.route_parent(prompt_hashes, lora_name)
                request["routed_experts_start_len"] = parent.covered_len if parent is not None else 0
        try:
            engine_url, responses = await self._generate(payload["prompt_tokens"], request, payload["num_samples"])
        except httpx.HTTPError as error:
            return {"error": str(error)}
        sequences = [_to_sequence(response) for response in responses]
        for sequence in sequences:
            if "error" in sequence:
                return sequence
        if self.rank_affinity is not None and engine_url != self.router_url:
            for sequence in sequences:
                self.rank_affinity.record(payload["prompt_tokens"], sequence["tokens"], engine_url)
        if self.routed_experts is not None:
            for sequence, response in zip(sequences, responses, strict=True):
                self._cache_routes(payload["prompt_tokens"], sequence["tokens"], response)
        if records is not None and payload["prompt_tokens"]:
            sequence_ids = sequence_ids or [f"seq-{uuid.uuid4().hex}" for _ in sequences]
            try:
                self._record_samples(
                    payload["prompt_tokens"], prompt_hashes, lora_name, parent, sequence_ids, sequences, responses, request
                )
            except (KeyError, ValueError) as error:
                return {"error": f"the engine returned invalid sampler records: {error!r}"}
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

    def _record_samples(
        self,
        prompt_tokens: list[int],
        prompt_hashes: np.ndarray,
        lora_name: str | None,
        parent: SequenceRecord | None,
        sequence_ids: list[str],
        sequences: list[dict],
        responses: list[dict],
        request: dict,
    ) -> None:
        """Record each sample; with supports, its logprobs become the support-renormalized ones the trainer scores."""
        records = self.sampler_records
        for sequence_id, sequence, response in zip(sequence_ids, sequences, responses, strict=True):
            if not sequence["tokens"]:
                continue
            meta = response["meta_info"]
            record = SequenceRecord(
                sequence_id, lora_name, len(prompt_tokens), np.asarray(sequence["tokens"], dtype=np.int32), parent=parent
            )
            if request.get("return_sampling_mask"):
                ids, offsets, log_probs, sequence["logprobs"] = parse_supports(
                    sequence["tokens"], meta["output_token_sampling_mask"], meta["output_token_sampling_logprobs"]
                )
                record.support_ids, record.support_offsets, record.support_log_probs = ids, offsets, log_probs
            if records.routes:
                record.routes = self._decode_routes(meta["routed_experts"], record)
            records.put(record, int(prompt_hashes[-1]))

    def _decode_routes(self, encoded: str, record: SequenceRecord) -> np.ndarray | None:
        """The ``[routes_start, covered_len)`` rows the engine returned, or None if their count is wrong."""
        rows = record.covered_len - record.routes_start
        routes = np.frombuffer(pybase64.b64decode(encoded.encode("ascii")), dtype=np.int32)
        if rows == 0 and record.parent is not None:
            return record.parent.routes[:0]
        if rows <= 0 or routes.size == 0 or routes.size % (rows * self.num_layers):
            logger.warning(f"routing replay: {routes.size} route values for {rows} positions; not recorded")
            return None
        routes = routes.reshape(rows, self.num_layers, -1).astype(np.int16)
        if record.parent is not None and routes.shape[1:] != record.parent.routes.shape[1:]:
            logger.warning(f"routing replay: route shape {routes.shape[1:]} differs from the parent's; not recorded")
            return None
        return routes

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
        records = self.sampler_records
        if self.routed_experts is not None or (records is not None and records.routes):
            request["return_routed_experts"] = True
        if records is not None and records.supports and _draws_from_support(sampling_params):
            request["return_sampling_mask"] = True
            request["sampling_logprobs_mode"] = "support"
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
