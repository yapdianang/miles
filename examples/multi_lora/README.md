# Multi-LoRA Tinker Gateway

> **Read the docs:** [Multi-LoRA training](https://miles.radixark.com/docs/advanced/lora#multi-lora-training).

- `serve_qwen3_30b_a3b_tinker.py`: prepare Qwen3-30B-A3B and launch the gateway.
- `serve_qwen_dense_tinker.py`: launch the canonical three-node, 64K dense-Qwen workloads.
- `serve_glm5_3_flash_tinker.py`: launch the canonical three-node, 64K GLM-5.3-Flash workload.
- `run_multi_tenant_example.py`: check marker memorization for one client or adapter isolation across concurrent tenants.

## TCLI runtime-workload contract

Trajectory reaches every engine through the same Tinker-compatible API. Keep the
transport and implementation as separate selectors:

```yaml
backend: tcli
training_engine: slime
```

`backend: tcli` selects the service boundary. `training_engine` selects the
implementation behind it: `skyrl` for a SkyRL runtime workload or `slime` for
this Miles runtime workload. A Miles service should publish both
`training_engine: slime` and the more specific
`training_backend: miles-megatron-multilora` labels so the TCLI scheduler never
routes a Slime/Miles-only experiment to a SkyRL endpoint with the same model
name.

The canonical B300 presets are disaggregated and expose port `10613`:

| Model | Training | Sampling | Parallelism | Context | LoRA scope |
| --- | ---: | ---: | --- | ---: | --- |
| `Qwen/Qwen3.5-4B` | 2 nodes / 16 GPUs | 1 node / 8 GPUs | TP2, PP2 | 65,504 | attention + MLP |
| `Qwen/Qwen3.6-27B` | 2 nodes / 16 GPUs | 1 node / 8 GPUs | TP4, PP2 | 65,504 | attention + MLP |
| `Qwen/Qwen3.8-27B` | 2 nodes / 16 GPUs | 1 node / 8 GPUs | TP4, PP2 | 65,504 | attention + MLP |
| `zai-org/GLM-5.3-Flash` | 2 nodes / 16 GPUs | 1 node / 8 GPUs | TP8, PP2, EP8 | 65,504 | MLP/MoE only |

GLM attention is intentionally excluded: SGLang cannot apply LoRA to the
model's fused KDA attention operators yet. All presets use rank 32, dynamic
microbatching, and activation recomputation. The Qwen presets use one-GPU
SGLang engines; the GLM preset uses its canonical eight-GPU inference layout.

## Layout

One 8-GPU node, disaggregated (multi-LoRA forbids `--colocate`):

- 4 training GPUs: TP2 for the dense layers, EP4 for the 128 routed experts.
- 4 sampling GPUs: two SGLang engines of 2 GPUs each, serving adapter versions by name.
- 4 adapter slots (`--multi-lora-n-adapters`), rank up to 32, covering attention
  (`linear_qkv`, `linear_proj`), the per-expert MoE projections (`linear_fc1`, `linear_fc2`),
  and the output layer (`output_layer`) so the cookbook's default `train_unembed=True` is servable.

The example enables attention, MLP, and output-head training. Client SDK flags
must match the server's selected groups. See [LoRA target selection](../../docs/advanced/lora.md#hf-target-source-of-truth)
for `--target-modules attn,mlp,unembed`; use `attn,mlp` to disable output-head
training. Tinker accepts only these group names and rejects `--exclude-modules`.

## Run

The gateway implements the `tinker==0.26.2` wire schema (newer SDKs renamed protobuf fields); install that exact version on the serving node and the client:

```bash
pip install "tinker==0.26.2"
```

Start the gateway:

```bash
python examples/multi_lora/serve_qwen3_30b_a3b_tinker.py prepare   # once per node
python examples/multi_lora/serve_qwen3_30b_a3b_tinker.py serve     # Tinker API on :10613
```

For a canonical three-node 64K service, start an external three-node Ray
cluster, set `MILES_SCRIPT_EXTERNAL_RAY=1` on its head, and choose one preset:

```bash
uv run python examples/multi_lora/serve_qwen_dense_tinker.py serve \
  --base-model Qwen/Qwen3.5-4B

uv run python examples/multi_lora/serve_qwen_dense_tinker.py serve \
  --base-model Qwen/Qwen3.6-27B

uv run python examples/multi_lora/serve_qwen_dense_tinker.py serve \
  --base-model Qwen/Qwen3.8-27B

uv run python examples/multi_lora/serve_glm5_3_flash_tinker.py serve
```

The client always uses the TCLI-provided base URL; it does not need to know
which training engine implements the endpoint:

```python
import tinker

service = tinker.ServiceClient(base_url=BASE_URL, api_key="tau")
training = service.create_lora_training_client(
    base_model="Qwen/Qwen3.6-27B",
    rank=32,
    train_attn=True,
    train_mlp=True,
    train_unembed=False,
)
```

Run the ten-step train/stale-sampler/resync gate before Tau:

```bash
uv run python examples/multi_lora/run_train_sampler_sync_test.py \
  --base-url "$BASE_URL" \
  --base-model Qwen/Qwen3.6-27B \
  --steps 10 \
  --match-p90 0.1
```

Checkpoints default to `<output_dir>/checkpoints/<run_id>`; use `--save-dir` to choose another root.

Install `tinker` on the client, then run the marker checks:

```bash
# one client: train, save for sampler, sample back the marker
python examples/multi_lora/run_multi_tenant_example.py --base-model /root/models/Qwen3-30B-A3B --mode single

# four tenants training concurrently on the same prompt with different markers;
# passing means the adapters stayed isolated end to end
python examples/multi_lora/run_multi_tenant_example.py --base-model /root/models/Qwen3-30B-A3B --mode multi --clients 4
```

## MiMo-V2.6-Flash dev loop and gates

`serve_mimo_v26_flash_tinker.py prepare` builds the engine and trainer checkpoints that are missing
(its docstring lists the commands). `tools/mimo_dev.py` runs the service on one B300 dev pod from
`tools/mimo_dev_pod.yaml`; `restart`, `stop` and `gate` send this checkout's copy of the tools to the
pod, and `sync` copies this checkout over the image's Miles checkout:

```bash
python examples/multi_lora/tools/mimo_dev.py up mimo-dev --node <node> --image <miles-mimo-tinker image>
python examples/multi_lora/tools/mimo_dev.py sync mimo-dev        # only to test local changes the image lacks
python examples/multi_lora/tools/mimo_dev.py restart mimo-dev [serve args, e.g. --sglang-mem-fraction-static 0.85]
python examples/multi_lora/tools/mimo_dev.py gate mimo-dev <gate> [gate args]
kubectl cp replay.json.gz mimo-dev:/root/replay.json.gz             # workload for the --workload gates
```

`tools/mimo_gates.py <gate> --base-url <gateway>` (`mimo_dev.py gate` adds `--base-url`) prints one
JSON summary on stdout and one line on stderr; a gate with a bar exits 1 when it fails. d is the
trainer − sampler log-prob of a sampled token, and k3 = mean(exp(d) − d − 1).

| Gate | Measures | Bar |
| --- | --- | --- |
| `parity` | k3 and mean \|d\| before and after one LoRA update at top-p 0.97 / top-k 1024 | k3 ≤ 0.001 before and after |
| `kl-decompose --workload W` | k3, mean \|d\| and mean d of prefill → repeated prefill, decode → prefill, prefill → trainer and decode → trainer at top-p 1.0 | decode → trainer k3 ≤ 0.0015 |
| `parity-by-length --workload W` | top-p 0.97 k3 and mean \|d\| per prompt-length bucket | none |
| `long-train --workload W` | forward and forward_backward time on the longest contexts | completes |
| `replay --workload W [--copies N --concurrency C --env-delay S --engine-log L]` | trajectories/min, generated tok/s and turn latency; with `--engine-log`, the engine's prefix-cache hit rate during the replay | none |
| `engine-stats [LOG ...]` | prefix-cache hit rate, decode tok/s, queue and DFlash accept length from SGLang batch log lines | none |

A replay workload is a JSON (or `.json.gz`) list of trajectories; each trajectory is a list of
turns `{"prompt": [token ids], "output_len": n}`. The service log is `/tmp/mimo-dev/service.log`
in the pod. To compare launcher arms, run `restart` with each arm's serve args and then the gates.

## Supported inputs

Training accepts text with 1-D loss inputs. 2-D soft targets, including SDFT,
are not supported. Sampling requires a `/sampler_weights/` path returned by
`save_weights_for_sampler()`; `/weights/` training checkpoints cannot be sampled directly.

## Failure handling

A terminal failure of `forward_backward`, `optim_step`, or `load_state` ends
training for that model, including commands already queued behind it.
This includes content validation failures with a valid model and sequence.
Create a new model and restore a saved checkpoint to continue; completed
futures and published checkpoints keep their results.

Known request-local failures of `forward` or sampling leave model training
available. Checkpoint load/save execution failures, including filesystem errors,
invalidate the shared trainer cell and stop the server.
Saving sampler weights commits an immutable directory;
it does not call the inference engines. Sampling loads that snapshot from disk
on demand, including after cache eviction. An engine load failure fails the
sampling request; it leaves the snapshot and training state intact. Unknown
trainer execution failures invalidate the shared trainer cell and stop the server.

This gateway provides failure isolation, not automatic training recovery.
Checkpoints persist; futures, deduplication, and unsaved accumulation do not
survive a server restart.

## Sampler snapshots

Training and inference must use the same base checkpoint. Tinker engines load
that frozen base at startup and serve without trainer weight updates; dummy
loading and `update_weights: true` are rejected. Ordinary full-model and
single-LoRA training continue to use the existing weight updater.

`--tinker-checkpoint-root` must be on storage shared by the trainers, gateway,
and every inference engine. All trainer ranks participate in adapter gathering;
rank 0 writes the tensors and config, then `META.json` after all ranks finish.
Existing sampler versions cannot be overwritten. Saving between `forward_backward`
and `optim_step` neither applies nor discards pending gradients.

Training checkpoint saves and loads are serialized within the gateway. Overwriting
deletes the previous checkpoint before writing the new one; a failed overwrite does
not preserve the previous checkpoint. Checkpoints are ordinary directories;
overwriting does not retain hidden versions.
