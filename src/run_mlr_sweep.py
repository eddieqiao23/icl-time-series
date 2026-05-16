"""
Driver for the Mixture-of-Linear-Regression (MLR) sweep.

Sweep grid:
    T (batch_size_per_task)        ∈ {2, 3, 4, 5}     # in-batch pairs
    N (num_batches_per_sample)     ∈ {30, 40, 50}     # context length (# batches)
    K (num_mixture_models)         ∈ {2, 3}           # mixture components
    d (regressor_dim)              = 4                # fixed

Per cell:
    token dim D = T*(d+1)+d   ∈ {14, 19, 24, 29}
    n_positions = N           ∈ {30, 40, 50}

Layout:
    sweeps/mlr/configs/<name>.yaml
    sweeps/mlr/sbatch/<name>.sbatch
    sweeps/mlr/logs/<name>.out
    sweeps/mlr/manifest.csv
    models/mlr/<name>/<uuid>/        (trained artifacts)

Usage (from src/):
    python run_mlr_sweep.py prepare                       # generate configs+sbatch+manifest
    python run_mlr_sweep.py submit --only T2_K2_N30       # submit one config
    python run_mlr_sweep.py submit                        # submit all (skips already-submitted)
    python run_mlr_sweep.py status                        # job state via sacct
"""

import argparse
import csv
import os
import subprocess
import sys
import uuid
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
SWEEP_DIR = REPO_ROOT / "sweeps" / "mlr"
CONFIG_DIR = SWEEP_DIR / "configs"
SBATCH_DIR = SWEEP_DIR / "sbatch"
LOG_DIR = SWEEP_DIR / "logs"
MANIFEST_PATH = SWEEP_DIR / "manifest.csv"
MODEL_ROOT_REL = "../models/mlr"  # relative to src/

WANDB_PROJECT = "mlr_sweep"

# Sweep grid
T_VALUES = [2, 3, 4, 5]
N_VALUES = [50]
K_VALUES = [2, 3]
NOISE_VALUES = [("noiseless", 0.0), ("noisy", 0.2)]
D_REGRESSOR = 4   # d
TRAIN_STEPS = 20001
BATCH_SIZE = 64
LEARNING_RATE = 0.0001

# SLURM
SBATCH_PARTITION = "mit_normal_gpu"
SBATCH_TIME = "05:00:00"
SBATCH_CPUS = 4
SBATCH_MEM = "16G"
SBATCH_GPU = "1"


def token_dim(T):
    return T * (D_REGRESSOR + 1) + D_REGRESSOR


def grid():
    for K in K_VALUES:
        for T in T_VALUES:
            for N in N_VALUES:
                for noise_label, noise_std in NOISE_VALUES:
                    name = f"T{T}_K{K}_N{N}_{noise_label}"
                    yield name, T, K, N, noise_label, noise_std


def build_config(name, T, K, N, noise_std, run_id):
    D = token_dim(T)
    return {
        "model": {
            "family": "gpt2",
            "n_embd": 128,
            "n_layer": 6,
            "n_head": 4,
            "n_dims": D,
            "n_positions": N,
        },
        "training": {
            "task": "linear_regression_mixture",
            "data": "linear_regression_mixture",
            "task_kwargs": {
                "num_mixture_models": K,
                "num_batches_per_sample": N,
                "batch_size_per_task": T,
                "regressor_dim": D_REGRESSOR,
                "normalize_coeffs": True,
                "regenerate_pool": True,
                "noise_std": noise_std,
            },
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "save_every_steps": 1000,
            "keep_every_steps": 100000,
            "train_steps": TRAIN_STEPS,
            "resume_id": run_id,
            "curriculum": {
                "dims":   {"start": D, "end": D, "inc": 1, "interval": 2000},
                "points": {"start": N, "end": N, "inc": 1, "interval": 2000},
            },
        },
        "out_dir": f"{MODEL_ROOT_REL}/{name}",
        "wandb": {
            "project": WANDB_PROJECT,
            "entity": None,
            "name": name,
            "notes": f"MLR sweep: T={T}, K={K}, N={N}, d={D_REGRESSOR}, D={D}, noise_std={noise_std}",
            "log_every_steps": 100,
        },
        "test_run": False,
    }


def build_sbatch(name, config_path, log_path):
    return f"""#!/bin/bash
#SBATCH -J mlr_{name}
#SBATCH -p {SBATCH_PARTITION}
#SBATCH -t {SBATCH_TIME}
#SBATCH -c {SBATCH_CPUS}
#SBATCH --mem={SBATCH_MEM}
#SBATCH --gres=gpu:{SBATCH_GPU}
#SBATCH -o {log_path}
#SBATCH -e {log_path}

set -eo pipefail

module load miniforge/24.3.0-0
eval "$(conda shell.bash hook)"
conda activate in-context-learning

cd {SRC_DIR}

echo "===== mlr sweep: {name} ====="
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'none')"
echo "Config: {config_path}"
echo

time srun python train.py --config {config_path}
"""


def cmd_prepare(args):
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    SBATCH_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    existing_uuids = {}
    if MANIFEST_PATH.exists():
        with MANIFEST_PATH.open() as f:
            for row in csv.DictReader(f):
                existing_uuids[row["name"]] = row["run_id"]
    # Pre-noise sweep used names like T5_K2_N50 (no _noiseless / _noisy suffix);
    # those runs were effectively noiseless, so remap their uuids onto the new
    # _noiseless names where we don't already have one.
    for legacy_name, run_id in list(existing_uuids.items()):
        new_name = f"{legacy_name}_noiseless"
        if new_name not in existing_uuids and not legacy_name.endswith(("_noiseless", "_noisy")):
            existing_uuids[new_name] = run_id

    rows = []
    for name, T, K, N, noise_label, noise_std in grid():
        run_id = existing_uuids.get(name, str(uuid.uuid4()))

        cfg = build_config(name, T, K, N, noise_std, run_id)
        cfg_path = CONFIG_DIR / f"{name}.yaml"
        with cfg_path.open("w") as f:
            yaml.dump(cfg, f, sort_keys=False, default_flow_style=False)

        log_path = LOG_DIR / f"{name}.out"
        sbatch_text = build_sbatch(name, cfg_path, log_path)
        sbatch_path = SBATCH_DIR / f"{name}.sbatch"
        sbatch_path.write_text(sbatch_text)
        sbatch_path.chmod(0o755)

        model_dir = REPO_ROOT / "models" / "mlr" / name / run_id

        rows.append({
            "name": name,
            "run_id": run_id,
            "T": T,
            "K": K,
            "N": N,
            "noise_std": noise_std,
            "d": D_REGRESSOR,
            "D": token_dim(T),
            "config_path": str(cfg_path),
            "sbatch_path": str(sbatch_path),
            "log_path": str(log_path),
            "model_dir": str(model_dir),
            "wandb_project": WANDB_PROJECT,
            "wandb_run_name": name,
            "slurm_job_id": "",
        })

    fieldnames = list(rows[0].keys())
    with MANIFEST_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} configs to {CONFIG_DIR}")
    print(f"Wrote {len(rows)} sbatch scripts to {SBATCH_DIR}")
    print(f"Manifest: {MANIFEST_PATH}")
    print()
    print("Sweep grid:")
    for r in rows:
        print(f"  {r['name']:28s}  T={r['T']} K={r['K']} N={r['N']} noise={r['noise_std']}  D={r['D']:2d}  uuid={r['run_id']}")


def cmd_submit(args):
    if not MANIFEST_PATH.exists():
        sys.exit("No manifest yet — run `prepare` first.")

    with MANIFEST_PATH.open() as f:
        rows = list(csv.DictReader(f))

    if args.only:
        wanted = set(args.only)
        rows_to_submit = [r for r in rows if r["name"] in wanted]
        if not rows_to_submit:
            sys.exit(f"No matching configs for --only {args.only}. Options: {[r['name'] for r in rows]}")
    else:
        rows_to_submit = rows

    submitted = 0
    skipped = 0
    for r in rows_to_submit:
        if r["slurm_job_id"] and not args.resubmit:
            print(f"  skip {r['name']:28s}  (already submitted as job {r['slurm_job_id']})")
            skipped += 1
            continue
        result = subprocess.run(
            ["sbatch", "--parsable", r["sbatch_path"]],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"  FAIL {r['name']:28s}  {result.stderr.strip()}")
            continue
        job_id = result.stdout.strip().split(";")[0]
        r["slurm_job_id"] = job_id
        print(f"  ok   {r['name']:28s}  job {job_id}")
        submitted += 1

    # Rewrite manifest
    fieldnames = list(rows[0].keys())
    with MANIFEST_PATH.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    print()
    print(f"Submitted: {submitted}  Skipped: {skipped}  Total in manifest: {len(rows)}")
    print(f"Logs will appear under: {LOG_DIR}")
    print(f"wandb runs: https://wandb.ai/<your-entity>/{WANDB_PROJECT}")


def cmd_status(args):
    if not MANIFEST_PATH.exists():
        sys.exit("No manifest yet — run `prepare` first.")

    with MANIFEST_PATH.open() as f:
        rows = list(csv.DictReader(f))

    job_ids = [r["slurm_job_id"] for r in rows if r["slurm_job_id"]]
    if not job_ids:
        print("No jobs submitted yet.")
        return

    result = subprocess.run(
        ["sacct", "-j", ",".join(job_ids),
         "--format=JobID,JobName,State,Elapsed,ExitCode", "-n", "-P", "-X"],
        capture_output=True, text=True,
    )
    states = {}
    for line in result.stdout.splitlines():
        parts = line.split("|")
        if len(parts) >= 5:
            states[parts[0]] = parts

    print(f"{'name':28s}  {'job_id':>10s}  {'state':12s}  {'elapsed':>10s}  exit")
    print("-" * 75)
    for r in rows:
        jid = r["slurm_job_id"]
        if not jid:
            print(f"{r['name']:28s}  {'(none)':>10s}  {'(not submitted)':12s}")
            continue
        s = states.get(jid)
        if not s:
            print(f"{r['name']:28s}  {jid:>10s}  (no sacct record)")
            continue
        _, _, state, elapsed, exit_code = s[:5]
        print(f"{r['name']:28s}  {jid:>10s}  {state:12s}  {elapsed:>10s}  {exit_code}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("prepare", help="generate configs, sbatch scripts, manifest")

    p_submit = sub.add_parser("submit", help="sbatch jobs (all, or --only NAME [NAME...])")
    p_submit.add_argument("--only", nargs="+", default=None,
                          help="submit only these config names (e.g. T2_K2_N50_noisy)")
    p_submit.add_argument("--resubmit", action="store_true", help="resubmit even if a job id is recorded")

    sub.add_parser("status", help="show job states via sacct")

    args = parser.parse_args()
    {"prepare": cmd_prepare, "submit": cmd_submit, "status": cmd_status}[args.cmd](args)


if __name__ == "__main__":
    main()
