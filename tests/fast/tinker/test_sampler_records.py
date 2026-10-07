"""Sampler records give each datum position the routes and sampling support of the call that produced it."""

import logging

import numpy as np
import pybase64
import pytest
import torch

from miles.tinker.core.types import UserInputError
from miles.tinker.expert_load import expert_counts
from miles.tinker.runtime import MilesBackend, _batches, _build_train_data
from miles.tinker.sampler_records import SamplerRecordStore, SequenceRecord, parse_supports, prefix_hashes

NUM_LAYERS, TOPK = 2, 3
PROMPT = [1, 2, 3]
TURN_1 = [10, 11, 12]
TURN_1_SUPPORTS = [[20, 10], [11], [12, 21, 22]]
TOOL = [5, 6]
TURN_2 = [13, 14]
TURN_2_SUPPORTS = [[13, 23], [24, 14]]
DIALOGUE = PROMPT + TURN_1 + TOOL + TURN_2


class FakeEngine:
    """SGLang's contract: routes recorded once per cached (adapter, prefix) position, returned from a start.

    Supports come back in support mode: per output token its candidate ids and their log-probabilities.
    """

    def __init__(self) -> None:
        self.cache: dict[tuple, np.ndarray] = {}
        self.outputs: list[tuple[list[int], list[list[int]]]] = []
        self.requests: list[dict] = []
        self.rng = np.random.default_rng(0)

    def routes(self, lora: str | None, tokens: list[int]) -> np.ndarray:
        """Routes of every position of ``tokens``, computed the first time a prefix is seen."""
        rows = []
        for end in range(1, len(tokens) + 1):
            key = (lora, tuple(tokens[:end]))
            if key not in self.cache:
                self.cache[key] = self.rng.integers(0, 64, size=(NUM_LAYERS, TOPK), dtype=np.int32)
            rows.append(self.cache[key])
        return np.stack(rows)

    async def post(self, url: str, request: dict) -> dict:
        self.requests.append(request)
        tokens, supports = self.outputs.pop(0)
        meta = {
            "output_token_logprobs": [(-3.0, token) for token in tokens],
            "finish_reason": {"type": "stop"},
        }
        if request.get("return_routed_experts"):
            start = request.get("routed_experts_start_len", 0)
            covered = request["input_ids"] + tokens[:-1]
            routes = self.routes(request.get("lora_path"), covered)[start:]
            meta["routed_experts"] = pybase64.b64encode(routes.astype(np.int32).tobytes()).decode("ascii")
        if request.get("return_sampling_mask"):
            assert request["sampling_logprobs_mode"] == "support"
            meta["output_token_sampling_mask"] = supports
            meta["output_token_sampling_logprobs"] = [
                [-0.1 * (rank + 1) for rank in range(len(row))] for row in supports
            ]
        return {"meta_info": meta}


@pytest.fixture
def engine(monkeypatch) -> FakeEngine:
    fake = FakeEngine()
    monkeypatch.setattr("miles.tinker.runtime.post", fake.post)
    return fake


def _backend(*, supports: bool = True, routes: bool = True, max_bytes: int = 2**30) -> MilesBackend:
    store = SamplerRecordStore(max_bytes, supports=supports, routes=routes)
    return MilesBackend(None, "http://router", num_layers=NUM_LAYERS, sampler_records=store)


def _payload(prompt: list[int], num_samples: int = 1, **params) -> dict:
    return {
        "prompt_tokens": prompt,
        "sampling_params": {"max_tokens": 8, "top_p": 0.97, "top_k": 1024, **params},
        "num_samples": num_samples,
        "prompt_logprobs": False,
        "topk_prompt_logprobs": 0,
    }


async def _sample(backend, engine, prompt, outputs, lora="m@1", **params) -> dict:
    engine.outputs.extend(outputs)
    sequence_ids = [f"seq-{len(engine.requests)}-{index}" for index in range(len(outputs))]
    return await backend.sample(_payload(prompt, len(outputs), **params), lora, sequence_ids=sequence_ids)


async def _dialogue(backend, engine, lora="m@1") -> None:
    await _sample(backend, engine, PROMPT, [(TURN_1, TURN_1_SUPPORTS)], lora=lora)
    # a second sample of the first prompt that agrees with the dialogue on its first token only
    await _sample(backend, engine, PROMPT, [([10, 30], [[10, 99], [30]])], lora=lora)
    await _sample(backend, engine, PROMPT + TURN_1 + TOOL, [(TURN_2, TURN_2_SUPPORTS)], lora=lora)


def _datum(tokens: list[int], **extra) -> dict:
    return {"tokens": tokens, "target_len": len(tokens) - 1, "target_tokens": tokens[1:], **extra}


def _rows(ids, offsets) -> list[list[int]]:
    ids, offsets = np.asarray(ids), np.asarray(offsets)
    return [ids[start:end].tolist() for start, end in zip(offsets[:-1], offsets[1:], strict=True)]


def _supports(backend: MilesBackend, datum: dict) -> list[list[int]]:
    records = backend.sampler_records
    ids, offsets, _ = records.datum_supports(datum, prefix_hashes(np.asarray(datum["tokens"][:-1])))
    return _rows(ids, offsets)


class TestRouteDeltas:
    async def test_a_later_turn_requests_routes_only_past_the_recorded_prefix(self, engine):
        backend = _backend()
        await _dialogue(backend, engine)
        starts = [request["routed_experts_start_len"] for request in engine.requests]
        assert starts == [0, 0, len(PROMPT) + len(TURN_1) - 1]

    async def test_concatenated_deltas_equal_the_engine_routes_of_the_whole_datum(self, engine):
        """Routes of a cached prefix are recorded when first computed, so the chain is exact within one adapter."""
        backend = _backend()
        await _dialogue(backend, engine)
        train_data = _build_train_data([(0, _datum(DIALOGUE))], sampler_records=backend.sampler_records)
        expected = engine.routes("m@1", DIALOGUE[:-1])
        np.testing.assert_array_equal(train_data["rollout_routed_experts"][0], expected)
        record = backend.sampler_records.get("seq-2-0")
        assert len(record.routes) == len(DIALOGUE) - 1 - (len(PROMPT) + len(TURN_1) - 1)

    async def test_a_new_adapter_version_never_builds_on_an_older_one(self, engine):
        backend = _backend()
        await _dialogue(backend, engine, lora="m@1")
        turn_3 = DIALOGUE + [7]
        await _sample(backend, engine, turn_3, [([15, 16], [[15], [16]])], lora="m@2")
        assert engine.requests[-1]["routed_experts_start_len"] == 0
        tokens = turn_3 + [15, 16]
        train_data = _build_train_data([(0, _datum(tokens))], sampler_records=backend.sampler_records)
        np.testing.assert_array_equal(train_data["rollout_routed_experts"][0], engine.routes("m@2", tokens[:-1]))

    async def test_a_datum_without_complete_routes_skips_replay(self, engine, caplog):
        backend = _backend()
        await _dialogue(backend, engine)
        with caplog.at_level(logging.WARNING, logger="miles.tinker.runtime"):
            train_data = _build_train_data(
                [(0, _datum(DIALOGUE)), (0, _datum(DIALOGUE[:-1]))], sampler_records=backend.sampler_records
            )
        assert "rollout_routed_experts" not in train_data
        assert "routing replay skipped: 1/2 datums have no engine routes" in caplog.text

    async def test_evicting_a_record_evicts_every_record_built_on_it(self, engine):
        backend = _backend(supports=False)
        await _sample(backend, engine, PROMPT, [(TURN_1, TURN_1_SUPPORTS)])
        await _sample(backend, engine, PROMPT + TURN_1 + TOOL, [(TURN_2, TURN_2_SUPPORTS)])
        await _sample(backend, engine, [9, 9], [([8], [[8]])])
        records = backend.sampler_records
        root = records.get("seq-0-0")
        records.max_bytes = records.num_bytes  # the next put evicts the least recently extended chain
        await _sample(backend, engine, [9, 8], [([8], [[8]])])
        assert records.get("seq-0-0") is None and records.get("seq-1-0") is None, "the chain leaves together"
        assert records.num_bytes == sum(records.get(sid).nbytes for sid in ("seq-2-0", "seq-3-0"))
        assert root.children == [] and len(records) == 2

    async def test_extending_a_chain_keeps_it_ahead_of_older_samples(self, engine):
        backend = _backend(supports=False)
        await _sample(backend, engine, PROMPT, [(TURN_1, TURN_1_SUPPORTS)])
        await _sample(backend, engine, [9, 9], [([8], [[8]])])
        await _sample(backend, engine, PROMPT + TURN_1 + TOOL, [(TURN_2, TURN_2_SUPPORTS)])
        records = backend.sampler_records
        records.max_bytes = records.num_bytes
        await _sample(backend, engine, [9, 8], [([8], [[8]])])
        assert records.get("seq-1-0") is None, "the unrelated sample is the least recently used"
        assert records.get("seq-0-0") is not None and records.get("seq-2-0") is not None

    def test_a_parent_evicted_while_sampling_leaves_the_child_without_routes(self):
        records = SamplerRecordStore(2**20, supports=False, routes=True)
        routes = np.zeros((2, 1, 1), np.int16)
        parent = SequenceRecord("a", "m@1", 2, np.array([5, 6], dtype=np.int32), routes=routes)
        records.put(parent, int(prefix_hashes(np.array([1, 2]))[-1]))
        child = SequenceRecord("b", "m@1", 4, np.array([7], dtype=np.int32), routes=routes[:1], parent=parent)
        records._drop_descendants(parent)
        records.put(child, int(prefix_hashes(np.array([1, 2, 5, 6]))[-1]))
        assert child.parent is None and child.routes is None


class TestSupports:
    async def test_a_multi_turn_datum_takes_each_position_from_the_call_that_sampled_it(self, engine):
        backend = _backend(routes=False)
        await _dialogue(backend, engine)
        # target i scores DIALOGUE[i + 1]; prompt and tool targets keep the full vocabulary
        assert _supports(backend, _datum(DIALOGUE)) == [[], [], *TURN_1_SUPPORTS, [], [], *TURN_2_SUPPORTS]

    async def test_a_datum_cut_inside_an_output_keeps_the_sampled_prefix(self, engine):
        backend = _backend(routes=False)
        await _dialogue(backend, engine)
        assert _supports(backend, _datum(PROMPT + TURN_1[:2])) == [[], [], *TURN_1_SUPPORTS[:2]]

    async def test_coverage_stops_where_a_target_leaves_the_sampled_output(self, engine):
        backend = _backend(routes=False)
        await _dialogue(backend, engine)
        targets = DIALOGUE[1:]
        targets[3] = 77  # the label of TURN_1[1]
        rows = _supports(backend, {**_datum(DIALOGUE), "target_tokens": targets})
        assert rows[2:5] == [TURN_1_SUPPORTS[0], [], []]
        assert rows[7:] == TURN_2_SUPPORTS

    async def test_a_later_call_rescores_positions_an_earlier_output_also_covers(self, engine):
        backend = _backend(routes=False)
        await _sample(backend, engine, PROMPT, [(TURN_1, TURN_1_SUPPORTS)])
        # a continuation from inside the first output, as after an aborted sample
        await _sample(backend, engine, PROMPT + TURN_1[:1], [(TURN_1[1:], [[11, 40], [12]])])
        rows = _supports(backend, _datum(PROMPT + TURN_1))
        assert rows == [[], [], TURN_1_SUPPORTS[0], [11, 40], [12]]

    async def test_provenance_spans_name_each_position_s_sequence(self, engine):
        backend = _backend(routes=False)
        await _dialogue(backend, engine)
        provenance = (
            [],
            [
                ("prompt", "seq-0-0", 1, 2),
                ("sampled", "seq-0-0", 0, 3),
                ("prompt", "seq-2-0", 6, 2),
                ("sampled", "seq-2-0", 0, 1),
                ("sampled", "seq-unknown", 1, 1),
            ],
        )
        rows = _supports(backend, _datum(DIALOGUE, provenance=provenance))
        assert rows == [[], [], *TURN_1_SUPPORTS, [], [], TURN_2_SUPPORTS[0], []]

    async def test_provenance_is_trusted_only_where_the_target_is_the_sampled_token(self, engine):
        backend = _backend(routes=False)
        await _dialogue(backend, engine)
        provenance = ([], [("prompt", "seq-0-0", 1, 2), ("sampled", "seq-1-0", 0, 2), ("prompt", "seq-2-0", 4, 5)])
        tokens = PROMPT + TURN_1[:2] + [0, 0, 0, 0, 0]  # seq-1-0 sampled [10, 30]; the datum continues with 11
        assert _supports(backend, _datum(tokens, provenance=provenance))[2:4] == [[10, 99], []]

    async def test_a_hash_shared_by_a_shorter_prefix_is_not_a_match(self, engine, monkeypatch):
        monkeypatch.setattr("miles.tinker.runtime.prefix_hashes", lambda tokens: np.cumsum(tokens).astype(np.uint64))
        monkeypatch.setattr("miles.tinker.sampler_records.prefix_hashes", lambda t: np.cumsum(t).astype(np.uint64))
        backend = _backend(routes=False)
        await _sample(backend, engine, PROMPT, [(TURN_1, TURN_1_SUPPORTS)])
        tokens = [6, *TURN_1]  # sum(PROMPT) == 6: same hash, prompt length 1 instead of 3
        records = backend.sampler_records
        ids, offsets, _ = records.datum_supports(_datum(tokens), np.cumsum(tokens[:-1]).astype(np.uint64))
        assert _rows(ids, offsets) == [[], [], []]

    async def test_samples_return_logprobs_renormalized_within_the_support(self, engine):
        backend = _backend(routes=False)
        result = await _sample(backend, engine, PROMPT, [(TURN_1, TURN_1_SUPPORTS)])
        # the fake engine's support log-probabilities are -0.1 * (rank + 1); TURN_1 tokens rank 2nd, 1st, 1st
        assert result["sequences"][0]["logprobs"] == pytest.approx([-0.2, -0.1, -0.1])

    @pytest.mark.parametrize(
        "params, requested",
        [
            ({"top_p": 0.97, "top_k": 1024}, True),
            ({"top_p": 1.0, "top_k": 50}, True),
            ({"top_p": 1.0, "top_k": -1}, False),
            ({"temperature": 0.0}, False),
            ({"top_k": 1}, False),
        ],
    )
    def test_supports_are_requested_only_for_restricted_sampling(self, params, requested):
        request = _backend(routes=False)._generate_request(_payload(PROMPT, **params), lora_name="m@1")
        assert request.get("return_sampling_mask", False) is requested
        assert ("sampling_logprobs_mode" in request) is requested

    def test_top_p_without_a_top_k_bound_is_a_user_error(self):
        with pytest.raises(UserInputError, match="top_k bound"):
            _backend(routes=False)._generate_request(_payload(PROMPT, top_k=-1), lora_name="m@1")

    async def test_a_response_without_requested_supports_fails_the_sample(self, monkeypatch):
        async def no_supports(url, request):
            return {"meta_info": {"output_token_logprobs": [(-2.0, 10)], "finish_reason": {"type": "stop"}}}

        monkeypatch.setattr("miles.tinker.runtime.post", no_supports)
        result = await _backend(routes=False).sample(_payload(PROMPT), "m@1", sequence_ids=["s"])
        assert "invalid sampler records" in result["error"]

    def test_supports_must_contain_their_sampled_token_exactly_once(self):
        with pytest.raises(ValueError, match="exactly once"):
            parse_supports([10, 11], [[10], [12]], [[0.0], [0.0]])
        with pytest.raises(ValueError, match="aligned"):
            parse_supports([10], [[10, 20]], [[0.0]])


class TestTrainData:
    async def test_supports_ride_with_the_datums_and_padding_copies_them(self, engine):
        backend = _backend(routes=False)
        await _dialogue(backend, engine)
        datums = [(0, _datum(DIALOGUE)), (0, _datum([7, 8, 9])), (0, _datum(DIALOGUE, padding=True))]
        train_data = _build_train_data(datums, sampler_records=backend.sampler_records, loss_fn="score_centering")
        ids = train_data["rollout_sampling_mask_ids"]
        offsets = train_data["rollout_sampling_mask_offsets"]
        assert [_rows(i.numpy(), o.numpy()) for i, o in zip(ids, offsets, strict=True)] == [
            [[], [], *TURN_1_SUPPORTS, [], [], *TURN_2_SUPPORTS],
            [[], []],
            [[], [], *TURN_1_SUPPORTS, [], [], *TURN_2_SUPPORTS],
        ]
        log_probs = train_data["rollout_sampling_mask_log_probs"][0].numpy()
        np.testing.assert_allclose(log_probs, [-0.1, -0.2, -0.1, -0.1, -0.2, -0.3, -0.1, -0.2, -0.1, -0.2])

    async def test_only_score_centering_ships_the_sampler_log_probs(self, engine):
        backend = _backend(routes=False)
        await _dialogue(backend, engine)
        train_data = _build_train_data([(0, _datum(DIALOGUE))], sampler_records=backend.sampler_records, loss_fn="ppo")
        assert "rollout_sampling_mask_ids" in train_data and "rollout_sampling_mask_log_probs" not in train_data

    async def test_loss_positions_without_a_support_are_reported(self, engine, caplog):
        backend = _backend(routes=False)
        await _sample(backend, engine, PROMPT, [(TURN_1, TURN_1_SUPPORTS)])
        advantages = [0.0] * 9
        advantages[2:5] = [1.0] * 3  # TURN_1, recorded
        advantages[7:] = [1.0] * 2  # TURN_2, sampled by a call the store does not hold
        with caplog.at_level(logging.WARNING, logger="miles.tinker.runtime"):
            _build_train_data([(0, _datum(DIALOGUE, advantages=advantages))], sampler_records=backend.sampler_records)
        assert "2/5 loss positions in 1 datums have no engine support" in caplog.text


def _b64(array: np.ndarray) -> str:
    return pybase64.b64encode(array.tobytes()).decode("ascii")


class KeepingEngine(FakeEngine):
    """A request with keep_rollout_record leaves its routes and supports in the engine under its rid.

    Its response carries each output token's selected log-probability instead of its support.
    """

    def __init__(self) -> None:
        super().__init__()
        self.kept: dict[str, dict] = {}
        self.collected: list[list[str]] = []

    async def post(self, url: str, request: dict, max_retries: int = 60) -> dict:
        if url.endswith("/collect_rollout_records"):
            self.collected.append(list(request["rids"]))
            return {"records": {rid: self.kept.pop(rid) for rid in request["rids"] if rid in self.kept}}
        if not self.outputs:
            self.outputs.append(([0], [[0]]))  # a one-token greedy sample
        response = await super().post(url, request)
        if not request.get("keep_rollout_record"):
            return response
        meta = response["meta_info"]
        kept = dict.fromkeys(["routes", "routes_shape", "support_lengths", "support_ids", "support_logprobs"])
        kept["routes_start"] = request.get("routed_experts_start_len", 0)
        if "routed_experts" in meta:
            routes = np.frombuffer(pybase64.b64decode(meta.pop("routed_experts")), dtype=np.int32)
            routes = routes.reshape(-1, NUM_LAYERS, TOPK).astype(np.int16)
            kept["routes"], kept["routes_shape"] = _b64(routes), list(routes.shape)
        if "output_token_sampling_mask" in meta:
            rows, values = meta.pop("output_token_sampling_mask"), meta["output_token_sampling_logprobs"]
            kept["support_lengths"] = _b64(np.array([len(row) for row in rows], dtype=np.int32))
            kept["support_ids"] = _b64(np.array([token for row in rows for token in row], dtype=np.int32))
            kept["support_logprobs"] = _b64(np.array([value for row in values for value in row], dtype=np.float32))
            tokens = [entry[1] for entry in meta["output_token_logprobs"]]
            meta["output_token_sampling_logprobs"] = [
                row_values[row.index(token)] for row, row_values, token in zip(rows, values, tokens, strict=True)
            ]
        self.kept[request["rid"]] = kept
        return response


@pytest.fixture
def keeping_engine(monkeypatch) -> KeepingEngine:
    fake = KeepingEngine()
    monkeypatch.setattr("miles.tinker.runtime.post", fake.post)
    return fake


def _collecting_backend(*, supports: bool = True, routes: bool = True) -> MilesBackend:
    async def engine_urls() -> list[str]:
        return ["http://engine"]

    store = SamplerRecordStore(2**30, supports=supports, routes=routes, collect=True)
    return MilesBackend(None, "http://router", num_layers=NUM_LAYERS, sampler_records=store, engine_urls=engine_urls)


async def _train_data(backend: MilesBackend, datums: list[dict], loss_fn: str = "score_centering") -> dict:
    """What forward_backward hands the trainer."""
    slot_datums = [(0, datum) for datum in datums]
    if backend.sampler_records.collect:
        await backend._collect_rollout_records(datums)
    return _build_train_data(slot_datums, sampler_records=backend.sampler_records, loss_fn=loss_fn)


class TestCollection:
    async def test_turns_leave_routes_and_supports_in_the_engine(self, keeping_engine):
        backend = _collecting_backend()
        result = await _sample(backend, keeping_engine, PROMPT, [(TURN_1, TURN_1_SUPPORTS)])
        assert keeping_engine.requests[0]["rid"] == "seq-0-0" and keeping_engine.requests[0]["keep_rollout_record"]
        assert result["sequences"][0]["logprobs"] == pytest.approx([-0.2, -0.1, -0.1])
        record = backend.sampler_records.get("seq-0-0")
        assert record.routes is None and record.support_ids is None
        assert record.pending_routes and record.pending_supports
        assert list(keeping_engine.kept) == ["seq-0-0"]

    async def test_a_later_turn_still_requests_routes_past_the_recorded_prefix(self, keeping_engine):
        backend = _collecting_backend()
        await _dialogue(backend, keeping_engine)
        starts = [request["routed_experts_start_len"] for request in keeping_engine.requests]
        assert starts == [0, 0, len(PROMPT) + len(TURN_1) - 1]

    async def test_collected_records_equal_the_per_turn_ones(self, engine):
        per_turn = _backend()
        await _dialogue(per_turn, engine)
        expected = await _train_data(per_turn, [_datum(DIALOGUE)])

        keeping = KeepingEngine()
        collecting = _collecting_backend()
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr("miles.tinker.runtime.post", keeping.post)
            await _dialogue(collecting, keeping)
            collected = await _train_data(collecting, [_datum(DIALOGUE)])
        for key in ("rollout_routed_experts", "rollout_sampling_mask_ids", "rollout_sampling_mask_offsets"):
            np.testing.assert_array_equal(np.asarray(collected[key][0]), np.asarray(expected[key][0]))
        np.testing.assert_array_equal(
            collected["rollout_sampling_mask_log_probs"][0].numpy(),
            expected["rollout_sampling_mask_log_probs"][0].numpy(),
        )

    async def test_a_datum_collects_only_the_records_it_draws_on_and_only_once(self, keeping_engine):
        backend = _collecting_backend()
        await _dialogue(backend, keeping_engine)
        await _sample(backend, keeping_engine, [9, 9], [([8], [[8]])])
        await _train_data(backend, [_datum(DIALOGUE)])
        # the dialogue's turns, not the unrelated sample nor the second sample of the first prompt
        assert keeping_engine.collected == [["seq-2-0", "seq-0-0"]]
        assert set(keeping_engine.kept) == {"seq-1-0", "seq-3-0"}
        await _train_data(backend, [_datum(DIALOGUE)])
        assert len(keeping_engine.collected) == 1, "collected payloads stay with the gateway's records"
        assert backend.sampler_records.num_collected == 2

    async def test_routes_the_engine_lost_are_recomputed_and_lost_supports_counted(self, keeping_engine, caplog):
        backend = _collecting_backend()
        await _dialogue(backend, keeping_engine)
        del keeping_engine.kept["seq-0-0"]  # evicted before collection
        advantages = [0.0] * 2 + [1.0] * 3 + [0.0] * 2 + [1.0] * 2
        with caplog.at_level(logging.WARNING, logger="miles.tinker.runtime"):
            train_data = await _train_data(backend, [_datum(DIALOGUE, advantages=advantages)])
        np.testing.assert_array_equal(
            train_data["rollout_routed_experts"][0], keeping_engine.routes("m@1", DIALOGUE[:-1])
        )
        recompute = keeping_engine.requests[-1]
        assert recompute["input_ids"] == DIALOGUE[:-1] and recompute["sampling_params"]["max_new_tokens"] == 1
        records = backend.sampler_records
        assert (records.num_collected, records.num_missing, records.num_recomputed) == (1, 1, 1)
        assert records.get("seq-2-0").parent is None, "the recomputed record covers the whole prefix"
        assert "3/5 loss positions in 1 datums have no engine support" in caplog.text

    async def test_a_failed_collection_counts_as_missing(self, keeping_engine, monkeypatch):
        backend = _collecting_backend(routes=False)
        await _sample(backend, keeping_engine, PROMPT, [(TURN_1, TURN_1_SUPPORTS)])

        async def failing(url, request, max_retries=60):
            raise RuntimeError("engine down")

        monkeypatch.setattr("miles.tinker.runtime.post", failing)
        train_data = await _train_data(backend, [_datum(PROMPT + TURN_1)])
        assert "rollout_sampling_mask_ids" not in train_data
        assert backend.sampler_records.num_missing == 1

    async def test_collection_reads_every_engine(self, monkeypatch):
        engines = {"http://a": KeepingEngine(), "http://b": KeepingEngine()}
        serving = iter(engines.values())

        async def post(url, request, max_retries=60):
            engine = next(serving) if url == "http://router/generate" else engines[url.rsplit("/", 1)[0]]
            return await engine.post(url, request)

        async def engine_urls() -> list[str]:
            return list(engines)

        monkeypatch.setattr("miles.tinker.runtime.post", post)
        backend = _collecting_backend(routes=False)
        backend.engine_urls = engine_urls
        engines["http://a"].outputs.append((TURN_1, TURN_1_SUPPORTS))
        engines["http://b"].outputs.append(([8], [[8]]))
        await backend.sample(_payload(PROMPT), "m@1", sequence_ids=["first"])
        await backend.sample(_payload([9, 9]), "m@1", sequence_ids=["second"])
        train_data = await _train_data(backend, [_datum(PROMPT + TURN_1)])
        ids, offsets = train_data["rollout_sampling_mask_ids"][0], train_data["rollout_sampling_mask_offsets"][0]
        assert _rows(ids.numpy(), offsets.numpy()) == [[], [], *TURN_1_SUPPORTS]
        assert [engine.collected for engine in engines.values()] == [[["first"]], [["first"]]]

    async def test_forward_backward_trains_and_measures_expert_load_on_collected_routes(self, keeping_engine):
        backend = _collecting_backend()
        backend.moe_layers, backend.num_experts = [0, 1], 64
        await _dialogue(backend, keeping_engine)
        trained = {}

        async def fake_call_trainer(method, batch_id, train_data):
            trained.update(train_data)
            return [{"per_datum": [{"sample_index": 0, "loss": 1.0, "logprobs": torch.zeros(len(DIALOGUE) - 1)}]}]

        backend._call_trainer = fake_call_trainer
        [output] = await backend.forward_backward(1, [(0, _datum(DIALOGUE))], "cross_entropy", {})
        expected = keeping_engine.routes("m@1", DIALOGUE[:-1])
        np.testing.assert_array_equal(trained["rollout_routed_experts"][0], expected)
        np.testing.assert_array_equal(output["expert_counts"], expert_counts(expected, [0, 1], 64))

    def test_collection_calls_split_by_rows(self):
        assert _batches(["a", "b", "c", "d"], [3, 2, 9, 1], max_size=5) == [["a", "b"], ["c"], ["d"]]
