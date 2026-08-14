---
title: Qwen3.8
description: Launch recipe for Qwen3.8-27B on the shared Qwen GDN implementation.
---

[Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) is a dense hybrid
Gated-Delta-Net model with the same training geometry as Qwen3.6-27B: 64
layers, hidden size 5120, 24 query heads, 4 KV heads, head dimension 256, and
vocabulary size 248320. Its Hugging Face `model_type` remains `qwen3_5`, so
Miles reuses the existing Qwen3.5 model spec and Megatron-Bridge mapping.

## Launch

Download and convert the checkpoint using the Qwen3.5 bridge, then select the
dedicated recipe:

```bash
hf download Qwen/Qwen3.8-27B --local-dir /root/models/Qwen3.8-27B
python scripts/run_qwen3_dense.py --model-name Qwen3.8-27B
```

The recipe uses TP4, sequence parallelism, a dynamic token budget of 8192 per
trainer GPU, one GPU per SGLang engine, and CPU optimizer offload on one 8-GPU
node. The dedicated `qwen3.8-27B` model type aliases the verified
`qwen3.6-27B` Megatron arguments rather than duplicating architecture flags.

For a Tinker-compatible shared service, pass both the checkpoint and model
type to the service launcher:

```bash
python examples/tinker_backend/run_tinker_backend.py serve \
  --hf-checkpoint /root/models/Qwen3.8-27B \
  --megatron-model-type qwen3.8-27B \
  --tp 4 \
  --extra-args "--tinker-frontend"
```
