"""The gateway records each sampled token's engine support and gives every datum position the support it was drawn from."""

import logging

import numpy as np
import pytest

from miles.tinker.core.types import UserInputError
from miles.tinker.runtime import MilesBackend, SamplingSupportCache, _build_train_data

PROMPT = [1, 2, 3]
TURN_1 = [10, 11, 12]
TURN_1_SUPPORTS = [[20, 10], [11], [12, 21, 22]]
TOOL = [5, 6]
TURN_2 = [13, 14]
TURN_2_SUPPORTS = [[13, 23], [24, 14]]
DIALOGUE = PROMPT + TURN_1 + TOOL + TURN_2


def _rows(ids: np.ndarray, offsets: np.ndarray) -> list[list[int]]:
    return [ids[start:end].tolist() for start, end in zip(offsets[:-1], offsets[1:], strict=True)]


def _datum(tokens: list[int], **extra) -> dict:
    return {"tokens": tokens, "target_len": len(tokens) - 1, "target_tokens": tokens[1:], **extra}


def _dialogue_cache(max_bytes: int = 2**20) -> SamplingSupportCache:
    cache = SamplingSupportCache(max_bytes)
    cache.put(PROMPT, [(TURN_1, TURN_1_SUPPORTS)])
    # a later sample of the same prompt that agrees with the datum on its first token only
    cache.put(PROMPT, [([10, 30], [[10, 99], [30]])])
    cache.put(PROMPT + TURN_1 + TOOL, [(TURN_2, TURN_2_SUPPORTS)])
    return cache


class TestLookup:
    def test_a_multi_turn_datum_takes_each_position_from_the_call_that_sampled_it(self):
        rows = _rows(*_dialogue_cache().lookup(DIALOGUE, DIALOGUE[1:]))
        # target i scores DIALOGUE[i + 1]; prompt and tool targets keep the full vocabulary
        assert rows == [[], [], *TURN_1_SUPPORTS, [], [], *TURN_2_SUPPORTS]

    def test_a_datum_cut_inside_an_output_keeps_the_sampled_prefix(self):
        tokens = PROMPT + TURN_1[:2]
        assert _rows(*_dialogue_cache().lookup(tokens, tokens[1:])) == [[], [], *TURN_1_SUPPORTS[:2]]

    def test_a_datum_the_engine_did_not_sample_has_only_full_vocabulary_rows(self):
        tokens = [7, 8, 9, 10]
        ids, offsets = _dialogue_cache().lookup(tokens, tokens[1:])
        assert ids.size == 0 and offsets.tolist() == [0, 0, 0, 0]

    def test_coverage_stops_where_a_target_leaves_the_sampled_output(self):
        targets = DIALOGUE[1:]
        targets[3] = 77  # the label of TURN_1[1]
        rows = _rows(*_dialogue_cache().lookup(DIALOGUE, targets))
        assert rows[2:5] == [TURN_1_SUPPORTS[0], [], []]
        assert rows[7:] == TURN_2_SUPPORTS

    def test_a_later_call_rescores_positions_an_earlier_output_also_covers(self):
        cache = SamplingSupportCache(2**20)
        cache.put(PROMPT, [(TURN_1, TURN_1_SUPPORTS)])
        # a continuation from inside the first output, as after an aborted sample
        cache.put(PROMPT + TURN_1[:1], [(TURN_1[1:], [[11, 40], [12]])])
        tokens = PROMPT + TURN_1
        assert _rows(*cache.lookup(tokens, tokens[1:])) == [[], [], TURN_1_SUPPORTS[0], [11, 40], [12]]

    def test_a_prompt_hash_shared_by_a_shorter_prefix_is_not_a_match(self, monkeypatch):
        monkeypatch.setattr("miles.tinker.runtime._prefix_hashes", lambda tokens: np.cumsum(tokens).astype(np.uint64))
        cache = SamplingSupportCache(2**20)
        cache.put(PROMPT, [(TURN_1, TURN_1_SUPPORTS)])
        tokens = [6, *TURN_1]  # sum(PROMPT) == 6: same hash, prompt length 1 instead of 3
        assert _rows(*cache.lookup(tokens, tokens[1:])) == [[], [], []]

    def test_the_oldest_samples_are_evicted_first(self):
        cache = SamplingSupportCache(max_bytes=1)
        cache.put(PROMPT, [(TURN_1, TURN_1_SUPPORTS)])
        cache.put(PROMPT + TURN_1 + TOOL, [(TURN_2, TURN_2_SUPPORTS)])
        assert cache.num_bytes > 0
        assert _rows(*cache.lookup(DIALOGUE, DIALOGUE[1:])) == [[]] * 7 + TURN_2_SUPPORTS

    def test_a_support_without_its_sampled_token_is_rejected(self):
        with pytest.raises(ValueError, match="outside its sampling support"):
            SamplingSupportCache(2**20).put(PROMPT, [(TURN_1, [[10], [12], [12]])])


class TestTrainData:
    def test_supports_ride_with_the_datums_and_padding_copies_them(self):
        cache = _dialogue_cache()
        unsampled = [7, 8, 9]
        train_data = _build_train_data(
            [(0, _datum(DIALOGUE)), (0, _datum(unsampled)), (0, _datum(DIALOGUE, padding=True))],
            sampling_supports=cache,
        )
        ids = train_data["rollout_sampling_mask_ids"]
        offsets = train_data["rollout_sampling_mask_offsets"]
        assert [_rows(i.numpy(), o.numpy()) for i, o in zip(ids, offsets, strict=True)] == [
            [[], [], *TURN_1_SUPPORTS, [], [], *TURN_2_SUPPORTS],
            [[], []],
            [[], [], *TURN_1_SUPPORTS, [], [], *TURN_2_SUPPORTS],
        ]
        assert ids[0].dtype.itemsize == 4 and offsets[0].dtype.itemsize == 8

    def test_a_batch_without_supports_trains_on_the_full_vocabulary(self):
        train_data = _build_train_data([(0, _datum([7, 8, 9]))], sampling_supports=_dialogue_cache())
        assert "rollout_sampling_mask_ids" not in train_data

    def test_loss_positions_without_a_support_are_reported(self, caplog):
        advantages = [0.0] * 9
        advantages[2:5] = [1.0] * 3  # TURN_1, covered
        advantages[7:] = [1.0] * 2  # TURN_2, sampled by a call the cache no longer holds
        cache = SamplingSupportCache(2**20)
        cache.put(PROMPT, [(TURN_1, TURN_1_SUPPORTS)])
        with caplog.at_level(logging.WARNING, logger="miles.tinker.runtime"):
            _build_train_data([(0, _datum(DIALOGUE, advantages=advantages))], sampling_supports=cache)
        assert "2/5 loss positions in 1 datums have no engine support" in caplog.text

    def test_fully_covered_loss_positions_are_not_reported(self, caplog):
        weights = [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0]
        with caplog.at_level(logging.WARNING, logger="miles.tinker.runtime"):
            _build_train_data([(0, _datum(DIALOGUE, weights=weights))], sampling_supports=_dialogue_cache())
        assert "no engine support" not in caplog.text


class TestSampling:
    def _payload(self, **params) -> dict:
        return {
            "prompt_tokens": PROMPT,
            "sampling_params": {"max_tokens": 8, **params},
            "num_samples": 1,
            "prompt_logprobs": False,
            "topk_prompt_logprobs": 0,
        }

    @pytest.mark.parametrize(
        "params, requested",
        [
            ({"top_p": 0.97, "top_k": 1024}, True),
            ({"top_k": 50}, True),
            ({}, False),
            ({"temperature": 0.0, "top_p": 0.97, "top_k": 1024}, False),
            ({"top_k": 1}, False),
        ],
    )
    def test_supports_are_requested_only_for_restricted_sampling(self, params, requested):
        backend = MilesBackend(None, "http://router", sampling_supports=SamplingSupportCache(2**20))
        request = backend._generate_request(self._payload(**params), lora_name="m@1")
        assert request.get("return_sampling_mask", False) is requested
        plain = MilesBackend(None, "http://router")._generate_request(self._payload(**params), lora_name="m@1")
        assert "return_sampling_mask" not in plain

    def test_top_p_without_a_top_k_bound_is_a_user_error(self):
        backend = MilesBackend(None, "http://router", sampling_supports=SamplingSupportCache(2**20))
        with pytest.raises(UserInputError, match="top_k bound"):
            backend._generate_request(self._payload(top_p=0.97), lora_name="m@1")

    async def test_a_sample_returns_renormalized_logprobs_and_feeds_the_cache(self, monkeypatch):
        cache = SamplingSupportCache(2**20)
        backend = MilesBackend(None, "http://router", sampling_supports=cache)
        sent = []

        async def fake_post(url, request):
            sent.append(request)
            return {
                "meta_info": {
                    "output_token_logprobs": [(-2.0, token) for token in TURN_1],
                    "output_token_sampling_logprobs": [-0.5, 0.0, -1.0],
                    "output_token_sampling_mask": TURN_1_SUPPORTS,
                    "finish_reason": {"type": "stop"},
                }
            }

        monkeypatch.setattr("miles.tinker.runtime.post", fake_post)
        result = await backend.sample(self._payload(top_p=0.97, top_k=1024), lora_name="m@1")
        assert sent[0]["return_sampling_mask"] is True
        assert result["sequences"][0]["logprobs"] == [-0.5, 0.0, -1.0]
        tokens = PROMPT + TURN_1
        assert _rows(*cache.lookup(tokens, tokens[1:])) == [[], [], *TURN_1_SUPPORTS]

    async def test_a_response_without_supports_fails_the_sample(self, monkeypatch):
        backend = MilesBackend(None, "http://router", sampling_supports=SamplingSupportCache(2**20))

        async def fake_post(url, request):
            return {"meta_info": {"output_token_logprobs": [(-2.0, 10)], "finish_reason": {"type": "stop"}}}

        monkeypatch.setattr("miles.tinker.runtime.post", fake_post)
        result = await backend.sample(self._payload(top_p=0.97, top_k=1024), lora_name="m@1")
        assert "no valid sampling supports" in result["error"]
