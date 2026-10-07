# DFlash drafter fine-tuning on RL rollouts

MiMo-V2.6 (section 6.4) fine-tunes its DFlash drafter on early RL rollout logs, resampled to the RL training
distribution, so that acceptance follows the RL policy. These stages do the same for the drafter shipped in the
MiMo-V2.6-Flash-RL checkpoint (`dflash/`).

Export the rollouts where gcloud can read the Trajectory Service, then run the rest on one 8-GPU B300 node in the
`miles-mimo-tinker` image from `/root/miles`:

```bash
python -m examples.multi_lora.dflash_rl.export_rollouts --xids 1063168 1063227 1063268 1063329 \
    --num-train 300 --num-heldout 100 --out rollouts
python -m examples.multi_lora.dflash_rl.run_pipeline --rollouts /mnt/shared-volume/dflash-rl/rollouts \
    --work /mnt/shared-volume/dflash-rl
```

`run_pipeline` runs the parity gate first and stops if it fails: the shipped drafter's offline walk accept length
(`eval_offline`, from `drafter.py`) must match the engine's (`bench_engine`) on the same held-out rollouts. It then
extracts training hidden states (`extract_hidden`: TP4 DFLASH engines with a forward hook), fine-tunes
(`train_drafter`), evaluates offline, benches shipped vs RL drafter x block 6/8 x BF16/FP8 draft, and writes
`summary.json`. Each stage also runs alone; see its module docstring.

Serve the result with `serve_mimo_v26_flash_tinker.py serve --dflash-drafter <work>/drafter --dflash-block-size 6`
(add `--extra-args "--sglang-speculative-draft-model-quantization fp8"` for an FP8 draft).
