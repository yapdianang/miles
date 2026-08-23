"""Official tinker payloads <-> backend operation payloads.

The SDK's Datum is (model_input tokens, per-token ``loss_fn_inputs`` of
length N, next-token targets); the backend's sample is (tokens, trailing
``response_length`` span, per-token channels on that span). The bridge:

    input_tokens  = concat(encoded_text chunks)          # length N
    target_tokens = loss_fn_inputs["target_tokens"]      # length N
    tokens          = input_tokens + [target_tokens[-1]] # length N + 1
    response_length = N

so the trainer's shifted logprob for response position i is exactly the
logprob of ``tokens[i+1]`` given the first i+1 tokens — the official
"logprob of target i given the input prefix" for every position where
``target_tokens[i] == input_tokens[i+1]``. Positions with a non-zero loss
contribution MUST satisfy that next-token alignment (rejected otherwise);
zero-weighted positions (canonical RL pads prompt targets with 0) are
normalized to the next input token, and their returned logprob refers to
that normalized target.

Every rejection raises ``UserInputError`` — the caller records it as a
terminal FAILED(user) operation so the client's ordinal is still consumed.
"""

import math

import numpy as np

from miles.ray.tinker_frontend import wire

SUPPORTED_LOSS_FNS = ("cross_entropy", "importance_sampling", "ppo", "gspo")

# Official loss_fn_inputs channel -> backend per-token channel.
_CHANNEL_TO_BACKEND = {
    "weights": "loss_weights",
    "advantages": "advantages",
    "logprobs": "rollout_log_probs",
}
_REQUIRED_CHANNELS = {
    "cross_entropy": ("weights",),
    "importance_sampling": ("logprobs", "advantages"),
    "ppo": ("logprobs", "advantages"),
    "gspo": ("logprobs", "advantages", "weights"),
}
# Which channel decides whether a position contributes loss (and therefore
# must be a true next-token target).
_ACTIVE_CHANNEL = {
    "cross_entropy": "weights",
    "importance_sampling": "advantages",
    "ppo": "advantages",
    "gspo": "weights",
}


class UserInputError(ValueError):
    """Typed client-payload rejection (never a server fault)."""


def _decode_proto_tensor(tensor, public_pb) -> wire.TensorData:
    dtype_by_proto = {
        public_pb.DTYPE_FLOAT32: ("float32", np.dtype(np.float32)),
        public_pb.DTYPE_INT64: ("int64", np.dtype(np.int64)),
    }
    dtype_spec = dtype_by_proto.get(tensor.dtype)
    if dtype_spec is None:
        raise UserInputError(f"unsupported protobuf tensor dtype {tensor.dtype}")
    dtype_name, numpy_dtype = dtype_spec
    encoding = tensor.WhichOneof("encoding")
    if encoding == "dense":
        data = np.frombuffer(tensor.dense, dtype=numpy_dtype).tolist()
        crow_indices = col_indices = None
    elif encoding == "sparse_csr":
        data = np.frombuffer(tensor.sparse_csr.values, dtype=numpy_dtype).tolist()
        crow_indices = np.frombuffer(tensor.sparse_csr.crow_indices, dtype=np.int64).tolist()
        col_indices = np.frombuffer(tensor.sparse_csr.col_indices, dtype=np.int64).tolist()
    else:
        raise UserInputError("protobuf tensor must contain dense or sparse_csr data")
    return wire.TensorData(
        data=data,
        dtype=dtype_name,
        shape=list(tensor.shape) or None,
        sparse_crow_indices=crow_indices,
        sparse_col_indices=col_indices,
    )


def decode_forward_backward_proto(body: bytes) -> tuple[wire.ForwardBackwardRequest | wire.ForwardRequest, bool]:
    from google.protobuf.message import DecodeError
    from tinker.proto import tinker_public_pb2 as public_pb

    request = public_pb.ForwardBackwardRequest()
    try:
        request.ParseFromString(body)
    except DecodeError as exc:
        raise UserInputError(f"invalid forward_backward protobuf: {exc}") from exc

    data = []
    for datum in request.data:
        chunks = []
        for chunk in datum.model_input:
            chunk_type = chunk.WhichOneof("chunk")
            if chunk_type != "encoded_text":
                raise UserInputError(f"protobuf model_input chunk type '{chunk_type}' is not supported in v1")
            tokens = np.frombuffer(chunk.encoded_text.tokens, dtype=np.int32).tolist()
            chunks.append(wire.ModelInputChunk(type="encoded_text", tokens=tokens))
        data.append(
            wire.Datum(
                model_input=wire.ModelInput(chunks=chunks),
                loss_fn_inputs={
                    name: _decode_proto_tensor(tensor, public_pb) for name, tensor in datum.loss_fn_inputs.items()
                },
            )
        )
    fb_input = wire.ForwardBackwardInput(
        data=data,
        loss_fn=request.loss_fn,
        loss_fn_config=dict(request.loss_fn_config) or None,
    )
    if request.forward_only:
        return (
            wire.ForwardRequest(forward_input=fb_input, model_id=request.model_id, seq_id=request.seq_id),
            True,
        )
    return (
        wire.ForwardBackwardRequest(
            forward_backward_input=fb_input,
            model_id=request.model_id,
            seq_id=request.seq_id,
        ),
        False,
    )


def _encode_batched_tensor(tensors: list[dict], public_pb):
    dtype_by_name = {
        "float32": (public_pb.DTYPE_FLOAT32, np.dtype(np.float32)),
        "int64": (public_pb.DTYPE_INT64, np.dtype(np.int64)),
    }
    dtype_name = tensors[0]["dtype"]
    dtype_spec = dtype_by_name.get(dtype_name)
    if dtype_spec is None or any(tensor["dtype"] != dtype_name for tensor in tensors):
        raise RuntimeError("forward_backward result tensors must use one supported dtype per field")
    proto_dtype, numpy_dtype = dtype_spec
    arrays = [np.asarray(tensor["data"], dtype=numpy_dtype) for tensor in tensors]
    trailing_shape = list(tensors[0].get("shape") or [])[1:]
    if any(list(tensor.get("shape") or [])[1:] != trailing_shape for tensor in tensors):
        raise RuntimeError("forward_backward result tensors must have one trailing shape per field")

    batched = public_pb.BatchedTensor(dtype=proto_dtype, trailing_shape=trailing_shape)
    batched.data = b"".join(array.tobytes() for array in arrays)
    byte_offsets = np.asarray([0, *np.cumsum([array.nbytes for array in arrays])], dtype=np.int64)
    batched.offsets = byte_offsets.tobytes()
    return batched


def encode_future_response_proto(body: dict) -> bytes | None:
    from tinker.proto import tinker_public_pb2 as public_pb

    if body.get("type") == "forward_backward":
        response = public_pb.ForwardBackwardOutput(loss_fn_output_type=body["loss_fn_output_type"])
        response.metrics.update({name: float(value) for name, value in body.get("metrics", {}).items()})
        outputs = body.get("loss_fn_outputs", [])
        if outputs:
            fields = set(outputs[0])
            if any(set(output) != fields for output in outputs):
                raise RuntimeError("forward_backward result rows must contain the same fields")
            record = response.loss_fn_outputs.add(type_tag=body["loss_fn_output_type"], num_datums=len(outputs))
            for name in fields:
                record.fields[name].CopyFrom(_encode_batched_tensor([output[name] for output in outputs], public_pb))
        return response.SerializeToString()

    if body.get("type") == "sample":
        response = public_pb.SampleResponse(prompt_cache_hit_tokens=body.get("prompt_cache_hit_tokens", 0))
        stop_reason = {
            "stop": public_pb.STOP_REASON_STOP,
            "length": public_pb.STOP_REASON_LENGTH,
        }
        for sequence in body["sequences"]:
            sequence_proto = response.sequences.add(
                stop_reason=stop_reason[sequence["stop_reason"]],
                tokens=np.asarray(sequence["tokens"], dtype=np.int32).tobytes(),
            )
            if sequence.get("logprobs") is not None:
                sequence_proto.logprobs = np.asarray(sequence["logprobs"], dtype=np.float32).tobytes()
        if body.get("prompt_logprobs") is not None:
            prompt_logprobs = [np.nan if value is None else value for value in body["prompt_logprobs"]]
            response.prompt_logprobs = np.asarray(prompt_logprobs, dtype=np.float32).tobytes()
        return response.SerializeToString()

    return None


def _decode_1d(name: str, where: str, tensor: wire.TensorData, expect_len: int, integer: bool) -> list:
    if tensor.sparse_crow_indices is not None or tensor.sparse_col_indices is not None:
        raise UserInputError(f"{where}: sparse (CSR) '{name}' is not supported in v1 — send dense 1-D tensors")
    if tensor.shape is not None and (len(tensor.shape) != 1 or tensor.shape[0] != len(tensor.data)):
        raise UserInputError(
            f"{where}: '{name}' must be 1-D with shape matching its data "
            f"(got shape={tensor.shape}, len={len(tensor.data)}) — nested/top-K targets are not supported in v1"
        )
    if len(tensor.data) != expect_len:
        raise UserInputError(
            f"{where}: '{name}' must have one value per input token (got {len(tensor.data)}, want {expect_len})"
        )
    values = []
    for value in tensor.data:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise UserInputError(f"{where}: '{name}' must contain only finite numbers")
        if integer:
            if isinstance(value, float) and not value.is_integer():
                raise UserInputError(f"{where}: '{name}' must contain integer token ids")
            value = int(value)
            if value < 0:
                # No tokenizer has negative ids; vocab UPPER bounds are the
                # engine's to enforce (the frontend never loads the tokenizer).
                raise UserInputError(f"{where}: '{name}' token ids must be non-negative (got {value})")
            values.append(value)
        else:
            value = float(value)
            if not math.isfinite(value):
                raise UserInputError(f"{where}: '{name}' must contain only finite numbers")
            values.append(value)
    return values


def _input_tokens(where: str, model_input: wire.ModelInput) -> list[int]:
    tokens: list[int] = []
    for chunk in model_input.chunks:
        if chunk.type != "encoded_text":
            raise UserInputError(f"{where}: model_input chunk type '{chunk.type}' is not supported in v1 (text-only)")
        tokens.extend(chunk.tokens)
    if not tokens or not all(isinstance(t, int) and not isinstance(t, bool) for t in tokens):
        raise UserInputError(f"{where}: model_input must carry at least one encoded-text token")
    if any(t < 0 for t in tokens):
        raise UserInputError(f"{where}: model_input token ids must be non-negative")
    return tokens


def datum_to_sample(index: int, datum: wire.Datum, loss_fn: str) -> dict:
    where = f"data[{index}]"
    input_tokens = _input_tokens(where, datum.model_input)
    n = len(input_tokens)

    known = {"target_tokens", *_CHANNEL_TO_BACKEND}
    if unknown := sorted(set(datum.loss_fn_inputs) - known):
        raise UserInputError(f"{where}: unsupported loss_fn_inputs {unknown}; v1 accepts {sorted(known)}")
    if "target_tokens" not in datum.loss_fn_inputs:
        raise UserInputError(f"{where}: loss_fn_inputs must include 'target_tokens'")
    for required in _REQUIRED_CHANNELS[loss_fn]:
        if required not in datum.loss_fn_inputs:
            raise UserInputError(f"{where}: loss_fn '{loss_fn}' requires loss_fn_inputs['{required}']")

    targets = _decode_1d("target_tokens", where, datum.loss_fn_inputs["target_tokens"], n, integer=True)
    channels = {
        official: _decode_1d(official, where, tensor, n, integer=False)
        for official, tensor in datum.loss_fn_inputs.items()
        if official != "target_tokens"
    }

    # Positions that contribute loss must be true next-token targets; the
    # rest (canonical RL zero-weights its prompt span) are normalized to the
    # next input token, which is what their returned logprob refers to.
    active = channels[_ACTIVE_CHANNEL[loss_fn]]
    for i in range(n - 1):
        if active[i] != 0.0 and targets[i] != input_tokens[i + 1]:
            raise UserInputError(
                f"{where}: target_tokens[{i}]={targets[i]} has non-zero loss weight but is not the next input "
                f"token ({input_tokens[i + 1]}); v1 serves next-token targets only"
            )

    sample = {
        "tokens": input_tokens + [targets[-1]],
        "response_length": n,
        "loss_mask": [1] * n,
    }
    for official, backend_channel in _CHANNEL_TO_BACKEND.items():
        if official in channels:
            sample[backend_channel] = channels[official]
    return sample


def fb_input_to_payload(fb_input: wire.ForwardBackwardInput) -> dict:
    """Backend payload for one forward_backward/forward operation. The loss
    spec rides along for forward too: the trainer ignores it structurally
    (no gradients), and result translation recomputes the loss metrics from
    it (the backend attaches metrics only to forward_backward results)."""
    if fb_input.loss_fn not in SUPPORTED_LOSS_FNS:
        raise UserInputError(
            f"loss_fn '{fb_input.loss_fn}' is not supported in v1; supported: {', '.join(SUPPORTED_LOSS_FNS)}"
        )
    if not fb_input.data:
        raise UserInputError("forward_backward needs at least one datum")
    loss: dict = {"loss_fn": fb_input.loss_fn}
    if fb_input.loss_fn_config is not None:
        loss["loss_fn_config"] = dict(fb_input.loss_fn_config)
    return {
        "samples": [datum_to_sample(i, datum, fb_input.loss_fn) for i, datum in enumerate(fb_input.data)],
        "loss": loss,
    }


def adam_params_to_payload(adam: wire.AdamParams) -> dict:
    return {"adam_params": adam.model_dump()}


# ---------------- results: backend operation -> SDK terminal JSON ----------------


def fb_result_to_response(result: dict, payload: dict | None = None) -> dict:
    """ForwardBackwardOutput JSON. logprobs arrive in the operation's datum
    order, one row per datum, one value per input token. ``payload`` (the
    operation's request) triggers a metrics recompute for forward results,
    which the backend completes without metrics."""
    logprobs = result.get("logprobs") or []
    metrics = result.get("metrics")
    if metrics is None and payload is not None:
        from miles.ray.multi_lora.backend import operation_result_metrics

        metrics = operation_result_metrics(payload, logprobs)
    return {
        "type": "forward_backward",
        "loss_fn_output_type": "ArrayRecord",
        "loss_fn_outputs": [{"logprobs": {"data": row, "dtype": "float32", "shape": [len(row)]}} for row in logprobs],
        "metrics": metrics or {},
    }


def optim_result_to_response(result: dict) -> dict:
    metrics = {
        key: float(value)
        for key, value in (result or {}).items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    return {"type": "optim_step", "metrics": metrics}


def save_weights_result_to_response(tinker_path: str) -> dict:
    return {"type": "save_weights", "path": tinker_path}


def load_weights_result_to_response(tinker_path: str, model_id: str) -> dict:
    return {"type": "load_weights", "path": tinker_path, "model_id": model_id}


def sampler_publish_result_to_response(sampling_session_id: str) -> dict:
    return {"type": "save_weights_for_sampler", "path": None, "sampling_session_id": sampling_session_id}


# ---------------- sampling: SDK request <-> sglang router ----------------

_FINISH_TO_STOP_REASON = {"stop": "stop", "length": "length"}


def sampling_params_to_sglang(params: wire.SamplingParams) -> dict:
    """Per-request sglang sampling_params. ``seed`` is handled by the caller
    (each fanned-out sample i gets ``sampling_seed = seed + i``: deterministic
    per request, still diverse across num_samples)."""
    if params.max_tokens is None or params.max_tokens < 1:
        raise UserInputError("sampling_params.max_tokens is required (>= 1) in v1")
    if not math.isfinite(params.temperature) or params.temperature < 0:
        raise UserInputError("sampling_params.temperature must be a non-negative finite number")
    if not math.isfinite(params.top_p) or not 0 < params.top_p <= 1:
        raise UserInputError("sampling_params.top_p must be a finite number in (0, 1]")
    if params.top_k != -1 and params.top_k < 1:
        raise UserInputError("sampling_params.top_k must be -1 or at least 1")
    if params.seed is not None and not -(2**63) <= params.seed < 2**63:
        raise UserInputError("sampling_params.seed must fit in a signed 64-bit integer")
    sglang_params: dict = {
        "max_new_tokens": params.max_tokens,
        "temperature": params.temperature,
        "top_p": params.top_p,
        "top_k": params.top_k,
    }
    stop = params.stop
    if stop is not None:
        if isinstance(stop, str):
            sglang_params["stop"] = [stop]
        elif all(isinstance(s, str) for s in stop):
            sglang_params["stop"] = list(stop)
        elif all(isinstance(s, int) and not isinstance(s, bool) for s in stop):
            if any(s < 0 for s in stop):
                raise UserInputError("sampling_params.stop token ids must be non-negative")
            sglang_params["stop_token_ids"] = list(stop)
        else:
            raise UserInputError("sampling_params.stop must be a string, list of strings, or list of token ids")
    return sglang_params


def generation_to_sequence(generation: dict) -> dict:
    """One sglang /generate response -> one SampledSequence JSON."""
    meta = generation.get("meta_info") or {}
    finish = (meta.get("finish_reason") or {}).get("type")
    stop_reason = _FINISH_TO_STOP_REASON.get(finish)
    if stop_reason is None:
        raise RuntimeError(f"generation finished with '{finish}'")
    token_logprobs = meta.get("output_token_logprobs") or []
    return {
        "stop_reason": stop_reason,
        "tokens": [int(entry[1]) for entry in token_logprobs],
        "logprobs": [float(entry[0]) for entry in token_logprobs],
    }


def prompt_logprobs_from_generation(generation: dict, prompt_len: int) -> list[float | None]:
    """meta_info.input_token_logprobs (logprob_start_len=0) -> one float-or-None per prompt token."""
    entries = (generation.get("meta_info") or {}).get("input_token_logprobs")
    if not entries:
        raise RuntimeError("the engine returned no input_token_logprobs for a prompt_logprobs request")
    if len(entries) != prompt_len:
        raise RuntimeError(f"the engine returned {len(entries)} prompt logprobs for {prompt_len} prompt tokens")
    # The first entry has no context, so sglang reports None there; keep it.
    return [None if entry[0] is None else float(entry[0]) for entry in entries]


def sequences_to_sample_response(sequences: list[dict], prompt_logprobs: list[float | None] | None = None) -> dict:
    return {
        "type": "sample",
        "sequences": sequences,
        "prompt_logprobs": prompt_logprobs,
        "topk_prompt_logprobs": None,
        "prompt_cache_hit_tokens": 0,
    }
