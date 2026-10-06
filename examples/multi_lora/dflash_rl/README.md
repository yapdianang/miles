# DFlash drafter fine-tuning on RL rollouts

MiMo-V2.6 (section 6.4) fine-tunes its DFlash drafter on early RL rollout logs, resampled to the RL training
distribution, so that acceptance follows the RL policy. These stages do the same for the drafter shipped in the
MiMo-V2.6-Flash-RL checkpoint (`dflash/`). Stages (b) to (d) run on one 8-GPU B300 node in the
`miles-mimo-tinker` image, from `/root/miles`.

```bash
CKPT=/data/model-cache/mimo-v26/MiMo-V2.6-Flash-RL-w4a16-linear
WORK=/data/dflash-rl
# (a) Trajectory Service -> rollouts resampled to the trained task mix (needs gcloud)
python -m examples.multi_lora.dflash_rl.export_rollouts --xids <xid> ... --num-train 1000 --num-heldout 64 --out $WORK/rollouts
# (b) target hidden states from two TP4 DFLASH engines with a forward hook
python -m examples.multi_lora.dflash_rl.extract_hidden --hf-checkpoint $CKPT --rollouts $WORK/rollouts/train.jsonl --out $WORK/hidden/train
python -m examples.multi_lora.dflash_rl.extract_hidden --hf-checkpoint $CKPT --rollouts $WORK/rollouts/heldout.jsonl --out $WORK/hidden/heldout
# (c, e) fine-tune; writes a drafter directory for --speculative-draft-model-path
torchrun --nproc-per-node 8 -m examples.multi_lora.dflash_rl.train_drafter --target-checkpoint $CKPT --drafter $CKPT/dflash --data $WORK/hidden/train --out $WORK/drafter
# (d) offline and engine acceptance, block 6 vs 8, BF16 vs FP8 draft
python -m examples.multi_lora.dflash_rl.eval_offline --target-checkpoint $CKPT --data $WORK/hidden/heldout --drafter shipped=$CKPT/dflash --drafter rl=$WORK/drafter
python -m examples.multi_lora.dflash_rl.bench_engine --hf-checkpoint $CKPT --replay $WORK/rollouts/heldout_replay.json --out $WORK/bench --drafter shipped=$CKPT/dflash --drafter rl=$WORK/drafter
```

Serve the result with `serve_mimo_v26_flash_tinker.py serve --dflash-drafter $WORK/drafter --dflash-block-size 6`
(add `--extra-args "--sglang-speculative-draft-model-quantization fp8"` for an FP8 draft).

`drafter.py` reimplements SGLang's DFLASH draft forward for this checkpoint (bidirectional block, 1024-token
sliding window over the context, attention sinks, value scale, partial RoPE). Check it before training: the
offline walk accept length of the shipped drafter must match the engine's accept length for the same block size.
