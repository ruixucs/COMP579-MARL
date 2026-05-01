"""Aggregate step-level probabilistic-attack training results.

Scans
    eir_mappo/results/<env>/[<map>/]mappo_advt_belief/attack_prob_*/<seed>/run<n>/
for each run, reads config.json (to confirm attack_prob) and the TensorBoard
event file (to extract eval-return scalars), then:
    1) writes <out>/attack_prob_summary.csv  with one row per run
    2) writes <out>/attack_prob_curve.png    with mean ± std return vs attack_prob

This is COMP579 project tooling for the probabilistic-adversary experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable


PROB_DIR_RE = re.compile(r"^attack_prob_(?P<prob>[\d.]+)$")


def find_run_dirs(results_root: Path, env: str, map_name: str | None) -> list[Path]:
    """Return all run_dir paths for attack_prob_* experiments."""
    if env in {"toy", "rendezvous", "pursuit", "navigation", "cover"}:
        base = results_root / env / "mappo_advt_belief"
    elif map_name:
        base = results_root / env / map_name / "mappo_advt_belief"
    else:
        raise ValueError(f"env={env} requires --map")
    if not base.exists():
        return []
    runs: list[Path] = []
    for exp_dir in sorted(base.iterdir()):
        if not exp_dir.is_dir() or not PROB_DIR_RE.match(exp_dir.name):
            continue
        for seed_dir in sorted(exp_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
            for run_dir in sorted(seed_dir.glob("run*")):
                if (run_dir / "config.json").exists():
                    runs.append(run_dir)
    return runs


def read_attack_prob_from_config(run_dir: Path) -> float:
    with open(run_dir / "config.json") as f:
        cfg = json.load(f)
    return float(cfg["algo_args"]["algo"]["attack_prob"])


def read_seed_from_path(run_dir: Path) -> int:
    return int(run_dir.parent.name)


def load_scalar_from_tb(run_dir: Path, tag_candidates: Iterable[str]) -> tuple[str | None, float | None]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as e:
        raise ImportError("tensorboard is required for aggregation; pip install tensorboard") from e
    log_dir = run_dir / "logs"
    if not log_dir.exists():
        return None, None
    ea = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})
    ea.Reload()
    available = set(ea.Tags().get("scalars", []))
    for tag in tag_candidates:
        if tag in available:
            events = ea.Scalars(tag)
            if events:
                return tag, float(events[-1].value)
    return None, None


def collect_rows(run_dirs: Iterable[Path], tag_candidates: Iterable[str]) -> list[dict]:
    tag_candidates = list(tag_candidates)
    rows = []
    for run_dir in run_dirs:
        attack_prob = read_attack_prob_from_config(run_dir)
        seed = read_seed_from_path(run_dir)
        tag, value = load_scalar_from_tb(run_dir, tag_candidates)
        rows.append({
            "run_dir": str(run_dir),
            "seed": seed,
            "attack_prob": attack_prob,
            "metric_tag": tag or "",
            "metric_value": value if value is not None else float("nan"),
        })
    return rows


def group_by_attack_prob(rows: list[dict]) -> dict[float, list[float]]:
    grouped: dict[float, list[float]] = defaultdict(list)
    for r in rows:
        v = r["metric_value"]
        if v is None or (isinstance(v, float) and v != v):
            continue
        grouped[r["attack_prob"]].append(float(v))
    return dict(grouped)


DEFAULT_TAG_CANDIDATES = (
    "env/eval_return_mean",
    "env/eval_average_episode_rewards",
    "env/eval_episode_reward",
    "env/return_mean",
    "env/episode_reward",
)


def write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        print(f"[warn] no rows to write at {out_path}")
        return
    fieldnames = ["run_dir", "seed", "attack_prob", "metric_tag", "metric_value"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"[ok] wrote {len(rows)} rows -> {out_path}")


def plot_curve(rows: list[dict], out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[warn] matplotlib not installed; skip plot. (pip install matplotlib)")
        return
    grouped = group_by_attack_prob(rows)
    if not grouped:
        print(f"[warn] nothing to plot at {out_path}")
        return
    probs = sorted(grouped)
    means, stds, ns = [], [], []
    for p in probs:
        vals = grouped[p]
        means.append(statistics.fmean(vals))
        stds.append(statistics.pstdev(vals) if len(vals) > 1 else 0.0)
        ns.append(len(vals))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(probs, means, yerr=stds, marker="o", capsize=4)
    for p, m, n in zip(probs, means, ns):
        ax.annotate(f"n={n}", (p, m), textcoords="offset points", xytext=(6, 4))
    ax.set_xlabel("Step-level attack probability (attack_prob)")
    ax.set_ylabel("Final-eval mean return")
    ax.set_title("Probabilistic adversary: defender quality vs attack_prob")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[ok] wrote curve -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Aggregate COMP579 attack_prob experiment results.")
    parser.add_argument("--results-root", default="eir_mappo/results", type=Path)
    parser.add_argument("--env", required=True)
    parser.add_argument("--map", dest="map_name", default=None,
                        help="Required for SMAC/SMACv2/LBF; leave empty for toy/rendezvous/etc.")
    parser.add_argument("--out-dir", default="analysis/attack_prob", type=Path)
    parser.add_argument("--tag", action="append", default=None,
                        help="TB scalar tag to read (can repeat). Default: try common eval-return tags.")
    args = parser.parse_args()

    tag_candidates = args.tag if args.tag else list(DEFAULT_TAG_CANDIDATES)
    runs = find_run_dirs(args.results_root, args.env, args.map_name)
    if not runs:
        print(f"[warn] no attack_prob_* runs found under {args.results_root}/{args.env}/{args.map_name or ''}")
        return
    print(f"[info] found {len(runs)} runs")
    rows = collect_rows(runs, tag_candidates)
    used_tags = {r["metric_tag"] for r in rows if r["metric_tag"]}
    if not used_tags:
        print(f"[warn] none of the candidate tags matched. Tried: {tag_candidates}")
        print("       Inspect a run's TB tags with:")
        print(f"       python -c \"from tensorboard.backend.event_processing.event_accumulator import EventAccumulator;"
              f" e=EventAccumulator('{runs[0]}/logs'); e.Reload(); print(e.Tags()['scalars'])\"")
    else:
        print(f"[info] matched tags: {sorted(used_tags)}")

    write_csv(rows, args.out_dir / "attack_prob_summary.csv")
    plot_curve(rows, args.out_dir / "attack_prob_curve.png")


if __name__ == "__main__":
    main()
