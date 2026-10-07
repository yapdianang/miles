"""Stage (a): export RL rollouts from the Trajectory Service, resampled to the RL training task mix.

The rollout logs are the completed train-split samples of ``--xids``, optionally only those sampled at policy step
``<= --max-policy-step`` (MiMo-V2.6 section 6.4 fine-tunes on early RL logs). The pool is the collected samples, or
every ENV_DONE sample when fewer than ``--num-train + --num-heldout`` were collected. The RL training distribution
is the task mix of the samples the trainer used: collected and not rejected (rejection drops zero-variance groups);
a run stopped before its first update has none, and the pool's own mix stands in. A rollout of task t gets weight
``trained_share(t) / pooled_count(t)``, and rollouts are drawn without replacement with Efraimidis-Spirakis keys.
A rollout is its last stored step: the full token sequence, with ``token_masks`` 1 on every sampled token.

Writes ``train.jsonl`` and ``heldout.jsonl`` ({key, xid, task_id, policy_step, tokens, loss_mask}) and
``summary.json``. Needs gcloud with Spanner and GCS read.

python -m examples.multi_lora.dflash_rl.export_rollouts --xids 1063168 1063227 1063268 1063329 \\
    --num-train 300 --num-heldout 100 --out /data/dflash-rl/rollouts
"""

import argparse
import gzip
import json
import math
import random
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def select_rollouts(rows: list[dict], *, num: int, max_policy_step: int | None, seed: int) -> list[dict]:
    """``num`` logged rollouts drawn without replacement so the task mix matches the trained samples' mix."""
    rows = [
        row
        for row in rows
        if max_policy_step is None or (row["policy_step"] is not None and row["policy_step"] <= max_policy_step)
    ]
    pool = [row for row in rows if row["collected"]]
    if len(pool) < num:
        pool = [row for row in rows if row["env_done"]]
    trained = Counter(row["task_id"] for row in rows if row["trained"]) or Counter(row["task_id"] for row in pool)
    pool = [row for row in pool if trained[row["task_id"]]]
    pooled = Counter(row["task_id"] for row in pool)
    total = sum(trained.values())
    rng = random.Random(seed)
    # log(u) / weight: the log of the Efraimidis-Spirakis key u ** (1 / weight), which underflows for rare tasks.
    keys = [math.log(1.0 - rng.random()) * pooled[row["task_id"]] * total / trained[row["task_id"]] for row in pool]
    order = sorted(range(len(pool)), key=keys.__getitem__, reverse=True)
    return [pool[index] for index in order[:num]]


def rollout_from_step(step: dict) -> dict:
    tokens, loss_mask = step["tokens"], step["token_masks"]
    if len(tokens) != len(loss_mask) or not any(loss_mask):
        raise ValueError(f"step has {len(tokens)} tokens, {len(loss_mask)} masks and {sum(loss_mask)} sampled")
    return {"tokens": tokens, "loss_mask": loss_mask}


def _sql(args, query: str) -> list[list]:
    command = [args.gcloud, "spanner", "databases", "execute-sql", args.database, f"--instance={args.instance}"]
    command += [f"--project={args.project}", "--format=json", f"--sql={query}"]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(result.stdout).get("rows", [])


def _logged_rows(args) -> list[dict]:
    if not all(xid.isdigit() for xid in args.xids):
        raise ValueError(f"XIDs must be digits: {args.xids}")
    xids = ", ".join(f"'{xid}'" for xid in args.xids)
    rows = _sql(
        args,
        "SELECT s.tid, s.xid, s.task_id, s.rollout_policy_step, s.is_collected, s.rejection_reason IS NULL, "
        "s.termination_reason, "
        "(SELECT st.step_blob_uri FROM steps st WHERE st.trajectory_id = s.tid ORDER BY st.step_index DESC LIMIT 1) "
        f"FROM samples s WHERE s.xid IN ({xids}) AND s.split = 'train' AND s.status = 'completed' "
        "AND s.tid IS NOT NULL",
    )
    return [
        {
            "key": tid,
            "xid": xid,
            "task_id": task,
            "policy_step": None if step is None else int(step),
            "collected": collected,
            "trained": collected and not_rejected,
            "env_done": termination == "TERMINATION_REASON_ENV_DONE",
            "uri": uri,
        }
        for tid, xid, task, step, collected, not_rejected, termination, uri in rows
        if uri is not None
    ]


def _load_rollout(args, row: dict) -> dict:
    blob = subprocess.run([args.gcloud, "storage", "cat", row["uri"]], check=True, capture_output=True).stdout
    step = json.loads(gzip.decompress(blob) if blob[:2] == b"\x1f\x8b" else blob)
    return {key: row[key] for key in ("key", "xid", "task_id", "policy_step")} | rollout_from_step(step)


def _write_jsonl(path: Path, rollouts: list[dict]) -> None:
    with open(path, "w") as file:
        for rollout in rollouts:
            file.write(json.dumps(rollout) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--xids", nargs="+", required=True)
    parser.add_argument("--num-train", type=int, required=True)
    parser.add_argument("--num-heldout", type=int, default=64)
    parser.add_argument("--max-policy-step", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gcloud", default="gcloud")
    parser.add_argument("--project", default="useful-memory-477923-v7")
    parser.add_argument("--instance", default="trajectory-spanner")
    parser.add_argument("--database", default="trajectories")
    args = parser.parse_args(argv)

    rows = _logged_rows(args)
    chosen = select_rollouts(
        rows, num=args.num_train + args.num_heldout, max_policy_step=args.max_policy_step, seed=args.seed
    )
    with ThreadPoolExecutor(32) as pool:
        rollouts = list(pool.map(lambda row: _load_rollout(args, row), chosen))
    heldout, train = rollouts[: args.num_heldout], rollouts[args.num_heldout :]
    args.out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(args.out / "train.jsonl", train)
    _write_jsonl(args.out / "heldout.jsonl", heldout)
    summary = {
        "xids": args.xids,
        "logged": len(rows),
        "collected": sum(row["collected"] for row in rows),
        "trained": sum(row["trained"] for row in rows),
        "env_done": sum(row["env_done"] for row in rows),
        "chosen_policy_steps": Counter(row["policy_step"] for row in chosen),
        "trained_task_mix": Counter(row["task_id"] for row in rows if row["trained"]),
        "chosen_task_mix": Counter(row["task_id"] for row in chosen),
        "train": {"rollouts": len(train), "sampled_tokens": sum(sum(r["loss_mask"]) for r in train)},
        "heldout": {"rollouts": len(heldout), "sampled_tokens": sum(sum(r["loss_mask"]) for r in heldout)},
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps({key: summary[key] for key in ("logged", "collected", "trained", "env_done", "train", "heldout")})
    )


if __name__ == "__main__":
    main()
