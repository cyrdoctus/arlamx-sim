"""YAML sweep runner with a CSV ledger. No LLM.

Each job calls ``train.train_one`` or ``train_v12.train_run``. Completed names
(metrics.json present, or already in the ledger) are skipped unless ``--force``.

    PYTHONPATH=python python -m arlamx_v2 sweep --config python/configs/sweep_example.yaml
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import time
from pathlib import Path

import yaml

from arlamx_v2.paths import OUTPUTS


def _as_list(x):
    if x is None:
        return [None]
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _job_name(base, algo, controller, physics, seed, extra=""):
    parts = [base]
    if algo:
        parts.append(str(algo).upper())
    if controller:
        parts.append(str(controller))
    if physics and str(physics) not in ("standard", "default"):
        parts.append(str(physics).replace("/", "_"))
    if seed is not None:
        parts.append(f"s{seed}")
    if extra:
        parts.append(extra)
    return "_".join(parts)


def expand_jobs(cfg):
    """Build a list of job dicts from a product or an explicit arms list."""
    if cfg.get("arms"):
        jobs = []
        for i, arm in enumerate(cfg["arms"]):
            job = {
                "name": arm.get("name") or f"{cfg.get('name', 'sweep')}_arm{i}",
                "backend": arm.get("backend", cfg.get("backend", "train")),
                "variant": arm.get("variant", cfg.get("variant", "v8a")),
                "algo": arm.get("algo", cfg.get("algo", "ppo")),
                "controller": arm.get("controller", cfg.get("controller")),
                "physics": arm.get("physics", cfg.get("physics", "standard")),
                "gsi": arm.get("gsi", cfg.get("gsi")),
                "seed": int(arm.get("seed", cfg.get("seed", 42))),
                "timesteps": int(arm.get("timesteps", cfg.get("timesteps", 100_000))),
                "n_envs": int(arm.get("n_envs", cfg.get("n_envs", 8))),
                "arch": arm.get("arch", cfg.get("arch", "4x16")),
                "env_kw": dict(arm.get("env_kw") or {}),
                "ppo_kw": dict(arm.get("ppo_kw") or {}),
                "v12_kw": dict(arm.get("v12_kw") or {}),
            }
            jobs.append(job)
        return jobs

    algos = _as_list(cfg.get("algo", "ppo"))
    controllers = _as_list(cfg.get("controller"))
    physics = _as_list(cfg.get("physics", "standard"))
    seeds = _as_list(cfg.get("seeds", cfg.get("seed", 42)))
    base = cfg.get("name", "sweep")
    jobs = []
    for algo, ctrl, phys, seed in itertools.product(algos, controllers, physics, seeds):
        name = _job_name(base, algo, ctrl, phys, seed)
        jobs.append({
            "name": name,
            "backend": cfg.get("backend", "train"),
            "variant": cfg.get("variant", "v8a"),
            "algo": algo,
            "controller": ctrl,
            "physics": phys,
            "gsi": cfg.get("gsi"),
            "seed": int(seed),
            "timesteps": int(cfg.get("timesteps", 100_000)),
            "n_envs": int(cfg.get("n_envs", 8)),
            "arch": cfg.get("arch", "4x16"),
            "env_kw": {},
            "ppo_kw": dict(cfg.get("ppo_kw") or {}),
            "v12_kw": dict(cfg.get("v12_kw") or {}),
        })
    return jobs


def _parse_arch(arch):
    if isinstance(arch, (list, tuple)):
        return list(arch)
    layers, width = (int(x) for x in str(arch).lower().split("x"))
    return [width] * layers


def job_dest(job, out_root):
    """Directory that actually receives metrics.json for this backend."""
    if str(job.get("backend", "train")) == "train_v12":
        from arlamx_v2.train_v12 import OUT
        return Path(OUT) / "runs" / job["name"]
    return Path(out_root) / job["name"]


def _ledger_path(out_root):
    return Path(out_root) / "SWEEP_LEDGER.csv"


def _ledger_names(path):
    if not path.is_file():
        return set()
    with open(path, newline="") as fh:
        return {row["name"] for row in csv.DictReader(fh) if row.get("name")}


def _append_ledger(path, row, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.is_file()
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        if new:
            w.writeheader()
        w.writerow(row)


def run_job(job, out_root, force=False):
    out_root = Path(out_root)
    dest = job_dest(job, out_root)
    if not force and (dest / "metrics.json").is_file():
        print(f"[sweep] skip {job['name']} (metrics.json)", flush=True)
        return json.loads((dest / "metrics.json").read_text())

    env_kw = dict(job.get("env_kw") or {})
    if job.get("controller"):
        env_kw["controller"] = job["controller"]
    if job.get("physics"):
        env_kw["physics"] = job["physics"]
    if job.get("gsi"):
        env_kw["gsi"] = job["gsi"]

    backend = str(job.get("backend", "train"))
    t0 = time.time()
    if backend == "train_v12":
        from arlamx_v2.train_v12 import train_run
        kw = dict(job.get("v12_kw") or {})
        metrics = train_run(
            job["name"],
            kw.pop("w_slew", 0.5),
            int(job["timesteps"]),
            int(job["n_envs"]),
            int(job["seed"]),
            physics=kw.pop("physics", job.get("physics", "standard")),
            controller=kw.pop("controller", job.get("controller")),
            gsi=kw.pop("gsi", job.get("gsi")),
            **kw,
        )
    else:
        from arlamx_v2.train import train_one
        metrics = train_one(
            job["variant"],
            str(job["algo"]).lower(),
            int(job["timesteps"]),
            int(job["n_envs"]),
            int(job["seed"]),
            job["name"],
            out_root,
            ppo_kw=job.get("ppo_kw"),
            env_kw=env_kw,
            arch=_parse_arch(job.get("arch", "4x16")),
        )
    if metrics is None:
        metrics = {}
    metrics.setdefault("name", job["name"])
    metrics.setdefault("wall_s", time.time() - t0)
    return metrics


def run_sweep(cfg, force=False):
    out_root = Path(cfg.get("out") or (OUTPUTS / "training"))
    ledger = Path(cfg.get("ledger") or _ledger_path(out_root))
    done = set() if force else _ledger_names(ledger)
    fields = ("name", "algo", "seed", "controller", "physics", "gsi",
              "wall_s", "path")
    results = []
    for job in expand_jobs(cfg):
        if not force and job["name"] in done:
            print(f"[sweep] skip {job['name']} (ledger)", flush=True)
            continue
        metrics = run_job(job, out_root, force=force)
        row = {
            "name": job["name"],
            "algo": job.get("algo"),
            "seed": job.get("seed"),
            "controller": job.get("controller"),
            "physics": job.get("physics"),
            "gsi": job.get("gsi") or "",
            "wall_s": metrics.get("wall_s", ""),
            "path": str(job_dest(job, out_root)),
        }
        _append_ledger(ledger, row, fields)
        results.append(row)
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description="YAML sweep + ledger resume")
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--force", action="store_true",
                    help="re-run arms even if metrics.json / ledger say done")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text()) or {}
    rows = run_sweep(cfg, force=args.force)
    print(json.dumps(rows, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
