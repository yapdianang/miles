"""Translate Tinker SDK protobuf requests and results, including compressed requests."""

import numpy as np

from miles.tinker.core.types import UserInputError
from miles.tinker.server.encoding import build_datum, forward_backward_metrics
from tinker.proto import tinker_public_pb2 as public_pb

PROTO_CONTENT_TYPE = "application/x-protobuf"

_PROTO_DTYPE_TO_NUMPY = {
    public_pb.DTYPE_FLOAT32: np.dtype(np.float32),
    public_pb.DTYPE_INT64: np.dtype(np.int64),
    public_pb.DTYPE_INT32: np.dtype(np.int32),
}
_STOP_REASON_TO_PROTO = {
    "stop": public_pb.STOP_REASON_STOP,
    "length": public_pb.STOP_REASON_LENGTH,
}


def maybe_decompress(body: bytes, content_encoding: str | None) -> bytes:
    if content_encoding == "zstd":
        import zstandard

        return zstandard.ZstdDecompressor().decompress(body)
    return body


def decode_forward_backward_request(body: bytes) -> tuple[str, dict]:
    """ForwardBackwardRequest proto -> (op, internal payload)."""
    message = public_pb.ForwardBackwardRequest()
    message.ParseFromString(body)
    op = "forward_only" if message.forward_only else "forward_backward"
    try:
        return op, _decode_forward_backward(message)
    except (UserInputError, KeyError, TypeError, ValueError, IndexError) as error:
        return op, {"model_id": message.model_id, "seq_id": message.seq_id, "validation_error": str(error)}


def _decode_forward_backward(message) -> dict:
    datums = []
    for index, datum in enumerate(message.data):
        tokens: list[int] = []
        for chunk in datum.model_input:
            if chunk.WhichOneof("chunk") != "encoded_text":
                raise UserInputError(f"unsupported model_input chunk type: {chunk.WhichOneof('chunk')}")
            tokens.extend(np.frombuffer(chunk.encoded_text.tokens, dtype=np.int32).tolist())
        inputs = {name: _decode_tensor(name, tensor) for name, tensor in datum.loss_fn_inputs.items()}
        decoded_datum = build_datum(tokens, inputs, index)
        if datum.model_input_spans or datum.loss_fn_input_spans:
            decoded_datum["provenance"] = _decode_provenance(datum, index, len(tokens), decoded_datum["target_len"])
        datums.append(decoded_datum)

    loss_fn_config = dict(message.loss_fn_config)
    # Tinker SDK's v2 protobuf config supports both numeric and string values.
    if message.loss_fn_config_v2:
        loss_fn_config = {}
        for name, value in message.loss_fn_config_v2.items():
            kind = value.WhichOneof("value")
            if kind is None:
                raise UserInputError(f"loss_fn_config[{name!r}]: missing number or text value")
            loss_fn_config[name] = getattr(value, kind)

    decoded = {
        "model_id": message.model_id,
        "seq_id": message.seq_id,
        "datums": datums,
        "loss_fn": message.loss_fn,
        "loss_fn_config": loss_fn_config,
    }
    return decoded


def _decode_provenance(datum, index: int, num_inputs: int, num_targets: int) -> tuple[list, list]:
    """Both span lists as ``(kind, sequence_id, offset, length)`` runs, each tiling its field."""
    tilings = []
    for name, spans, size in (
        ("model_input", datum.model_input_spans, num_inputs),
        ("loss_fn_inputs", datum.loss_fn_input_spans, num_targets),
    ):
        runs = []
        for span in spans:
            kind = span.WhichOneof("span")
            run = getattr(span, kind) if kind is not None else None
            if run is None or run.length < 1 or run.offset < 0 or not run.sequence_id:
                raise UserInputError(f"datum {index}: invalid {name} provenance span")
            runs.append(("sampled" if kind == "sampled_tokens" else "prompt", run.sequence_id, run.offset, run.length))
        if sum(run[3] for run in runs) != size:
            raise UserInputError(f"datum {index}: {name} provenance spans must tile its {size} positions")
        tilings.append(runs)
    return tilings[0], tilings[1]


def _decode_tensor(name: str, tensor) -> list:
    np_dtype = _PROTO_DTYPE_TO_NUMPY.get(tensor.dtype)
    if np_dtype is None:
        raise UserInputError(f"loss_fn_inputs[{name!r}]: unsupported tensor dtype {tensor.dtype}")
    if len(tensor.shape) > 1:
        raise UserInputError(f"loss_fn_inputs[{name!r}]: shape {list(tensor.shape)} is not supported (1-D only)")
    encoding = tensor.WhichOneof("encoding")
    if encoding == "dense":
        return np.frombuffer(tensor.dense, dtype=np_dtype).tolist()
    if encoding == "sparse_csr":
        (length,) = tensor.shape
        dense = np.zeros(length, dtype=np_dtype)
        cols = np.frombuffer(tensor.sparse_csr.col_indices, dtype=np.int64)
        dense[cols] = np.frombuffer(tensor.sparse_csr.values, dtype=np_dtype)
        return dense.tolist()
    raise UserInputError(f"loss_fn_inputs[{name!r}]: tensor without data")


# -------- result encoding --------


def encode_forward_backward_output(result: dict) -> bytes:
    message = public_pb.ForwardBackwardOutput()
    message.loss_fn_output_type = "ArrayRecord"
    outputs = result["outputs"]
    message.metrics.update(forward_backward_metrics(outputs))

    record = message.loss_fn_outputs.add()
    record.num_datums = len(outputs)
    for field, arrays in (
        ("loss:sum", [np.asarray([output["loss"]], dtype=np.float32) for output in outputs]),
        ("logprobs", [np.asarray(output["logprobs"], dtype=np.float32) for output in outputs]),
    ):
        batched = record.fields[field]
        batched.dtype = public_pb.DTYPE_FLOAT32
        offsets = np.zeros(len(arrays) + 1, dtype=np.int64)
        np.cumsum([array.nbytes for array in arrays], out=offsets[1:])
        batched.offsets = offsets.tobytes()
        batched.data = b"".join(array.tobytes() for array in arrays)
        # flat per-datum arrays: empty trailing_shape decodes as shape [n]
    return message.SerializeToString()


def encode_sample_response(result: dict) -> bytes:
    message = public_pb.SampleResponse()
    for sequence in result["sequences"]:
        out = message.sequences.add()
        out.stop_reason = _STOP_REASON_TO_PROTO[sequence["stop_reason"]]
        out.tokens = np.asarray(sequence["tokens"], dtype=np.int32).tobytes()
        if sequence.get("logprobs") is not None:
            out.logprobs = np.asarray(sequence["logprobs"], dtype=np.float32).tobytes()
    if result.get("prompt_logprobs") is not None:
        message.prompt_logprobs = np.asarray(result["prompt_logprobs"], dtype=np.float32).tobytes()
    if result.get("topk_prompt_logprobs") is not None:
        topk = result["topk_prompt_logprobs"]
        token_ids = np.asarray(topk["token_ids"], dtype=np.int32)
        message.topk_prompt_logprobs.length, message.topk_prompt_logprobs.k = token_ids.shape
        message.topk_prompt_logprobs.token_ids = token_ids.tobytes()
        message.topk_prompt_logprobs.logprobs = np.asarray(topk["logprobs"], dtype=np.float32).tobytes()
    return message.SerializeToString()


PROTO_ENCODERS = {
    "forward_backward": encode_forward_backward_output,
    "forward_only": encode_forward_backward_output,
    "sample": encode_sample_response,
}
