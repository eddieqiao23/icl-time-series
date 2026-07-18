#!/usr/bin/env python3
"""Create the tracked checkpoint-readiness inventory."""

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from common.checkpoints import discover_checkpoints, write_inventory  # noqa: E402


def write_readiness(records, output_dir: Path, target_step: int) -> None:
    rows = []
    for K in (2, 3):
        for noise_std in (0.0, 0.2):
            for T in (2, 3, 4, 5):
                matches = [r for r in records if r.config_valid and r.checkpoint_readable
                           and (r.T, r.K, r.N, r.noise_std) == (T, K, 50, noise_std)]
                best = max(matches, key=lambda r: r.train_step or -1) if matches else None
                step = best.train_step if best else None
                status = "complete" if step is not None and step >= target_step else (
                    "partial" if best else "missing"
                )
                rows.append({
                    "T": T, "K": K, "N": 50, "noise_std": noise_std,
                    "status": status, "best_step": step or "",
                    "remaining_steps": max(target_step - step, 0) if step is not None else "",
                    "run_name": best.run_name if best else "",
                    "run_dir": f"{best.run_name}/{best.run_id}" if best else "",
                })
    path = output_dir / "readiness.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=HERE / "artifacts" / "summaries")
    parser.add_argument("--metadata-only", action="store_true",
                        help="Inspect configs and file presence without loading model weights.")
    parser.add_argument("--target-step", type=int, default=500_000)
    args = parser.parse_args()

    records = discover_checkpoints(args.models_root, read_checkpoints=not args.metadata_only)
    write_inventory(records, args.output_dir)
    write_readiness(records, args.output_dir, args.target_step)
    valid = sum(r.config_valid and r.checkpoint_readable for r in records)
    at_500k = sum(r.config_valid and r.checkpoint_readable and (r.train_step or 0) >= 500_000 for r in records)
    print(f"Found {len(records)} MLR runs: {valid} valid, {at_500k} at >=500k steps")
    print(f"Wrote {args.output_dir / 'checkpoints.csv'}")
    print(f"Wrote {args.output_dir / 'readiness.csv'}")


if __name__ == "__main__":
    main()
