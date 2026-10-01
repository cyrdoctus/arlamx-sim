#!/usr/bin/env python
"""ARLAMX front door: train, sweep, decay, snapshots, quantize, sim/eval/plot/legacy tools, test, verify."""
from __future__ import annotations

import argparse
import os
import runpy
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "python"))
CONFIG = ROOT / "config"

TOOLS = {
    "sim": {
        "decay_run": "prescribed min/max-drag decay (main.py decay reads orbit.yaml)",
        "aoa_decay": "fixed angle-of-attack decay with lift/drag logging",
        "lift_drag_study": "free-molecular lift/drag vs angle of attack",
        "srp_assess": "can SRP raise/hold the orbit",
        "srp_f107_sweep": "F10.7 sweep of an SRP-aware attitude strategy",
        "optics_run": "sail optical presets: force + 1-orbit energy",
        "advisor_run": "closed-loop heuristic / MPC campaign",
        "validate_attitudes": "simplified plates vs STL projected areas",
        "plot_simplify": "input mesh vs simplified plates (3D)",
        "geometry": "mesh (.stl/.obj/.step) -> simplified plates (.geom)",
    },
    "eval": {
        "mc_decay": "Monte-Carlo decay 500->300 km (v8/v10 archive models)",
        "mc_v12": "SC_v12 lifetime MC vs MPC (archive)",
        "mc_four": "four-task MC SC_v13 vs MPC (archive)",
        "damage_eval": "SC_v11 damage verification (archive)",
        "analyze_ood": "OOD / weather-spike / brownout analysis",
        "eval_v4": "SC_v4 bake-off evaluation",
        "check_mpc_ref": "MPC_v1 vs v2 reference checks",
        "inference_budget": "STM32U575 inference cost",
    },
    "plot": {
        "plot_v8": "SC_v8/v9 conference figures (archive)",
        "plot_v12": "SC_v12-v14 figures + animation (archive)",
        "plot_v14fix": "talk-set figures (archive)",
        "plot_production": "SC_v7 bake-off figures (archive)",
        "plot_gradient70": "70x70 gradient figures (archive)",
        "showcase_plots": "showcase figure set (archive)",
        "slides": "two conference-slide figures (archive)",
        "ipc_animation": "3D disturbance-response animation",
        "report_v8": "SC_v8/v9 master report (archive)",
        "build_mpc_report": "MPC / heuristic PDF (archive)",
    },
    "legacy": {
        "bench_v8": "SC_v8-v11 campaign stages (writes outputs/.old/v8)",
        "campaign": "SC_v7 IPC tuning campaign (writes outputs/.old/campaign)",
        "tune_mpc": "MPC_v2 tuning sprint (writes outputs/.old/v8)",
        "gradient70": "70x70 gradient sweeps (archive)",
    },
}

NO_FLAGS = {"inference_budget", "plot_v14fix", "plot_production", "plot_gradient70",
            "showcase_plots", "slides", "ipc_animation", "report_v8", "build_mpc_report",
            "gradient70"}

HELP = """ARLAMX  python main.py <command> [flags]

  train      train one model from config/network.yaml + orbit.yaml + train.yaml
             flags: --network F --orbit F --train F --from-snapshot S --set k=v
                    --algo --arch LxW --timesteps --n-envs --seed --variant
                    --backend --physics --gsi --controller --name
  sweep      several trainings: --config config/sweep.yaml [--force]
  decay      decay simulation from config/orbit.yaml (simulation:)  [--kind min|max|both]
  snapshots  list outputs/snapshots (id, status, backend, variant, steps, wall)
  quantize   INT8 actor copy: --in MODEL.zip --out OUT.zip [--algo]
  sim|eval|plot|legacy <tool> [tool flags]   (tools below; <tool> -h for flags)
  run <module> [flags]                       any python/arlamx_v2 module main
  test [pytest flags]                        pytest tests -q
  verify                                     independent physics cross-check
"""


def print_list():
    print(HELP)
    for group, tools in TOOLS.items():
        print(f"  [{group}]")
        for name, purpose in tools.items():
            print(f"    {name:<20} {purpose}{'  [no flags, runs at once]' if name in NO_FLAGS else ''}")
        print()
    print("Settings: config/network.yaml  config/orbit.yaml  config/train.yaml  config/plant/*.yaml")
    print("Outputs:  outputs/snapshots  outputs/models  outputs/results  (outputs/.old = pre-v2.7)")


def forward(module, argv):
    old = sys.argv
    sys.argv = [module, *argv]
    try:
        runpy.run_module(f"arlamx_v2.{module}", run_name="__main__", alter_sys=True)
    except SystemExit as e:
        return e.code or 0
    finally:
        sys.argv = old
    return 0


def add_settings_flags(ap):
    ap.add_argument("--network", type=Path, default=CONFIG / "network.yaml")
    ap.add_argument("--orbit", type=Path, default=CONFIG / "orbit.yaml")
    ap.add_argument("--train", type=Path, default=CONFIG / "train.yaml")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="dotted override, e.g. network.ppo.learning_rate=1e-4")


def cmd_train(argv):
    from arlamx_v2 import session as S
    ap = argparse.ArgumentParser(prog="main.py train")
    add_settings_flags(ap)
    ap.add_argument("--from-snapshot", type=Path, default=None,
                    help="replay: take network/orbit/train/plant from this snapshot")
    ap.add_argument("--algo", choices=("ppo", "sac", "td3"))
    ap.add_argument("--arch", help="LAYERSxWIDTH, e.g. 4x16")
    ap.add_argument("--timesteps", type=int)
    ap.add_argument("--n-envs", type=int)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--variant")
    ap.add_argument("--backend", choices=S.BACKENDS)
    ap.add_argument("--physics")
    ap.add_argument("--gsi", choices=("sentman", "cll"))
    ap.add_argument("--controller", choices=("mrp", "quaternion"))
    ap.add_argument("--name")
    ap.add_argument("--out", type=Path, default=None, help="output root (default outputs/)")
    a = ap.parse_args(argv)
    settings = S.load_settings(a.network, a.orbit, a.train, snapshot=a.from_snapshot)
    sets = [S.parse_set(x) for x in a.set]
    if a.arch:
        layers, width = (int(x) for x in a.arch.lower().split("x"))
        sets += [("network.layers", layers), ("network.width", width)]
    for flag, key in (("algo", "network.algo"), ("timesteps", "network.timesteps"),
                      ("n_envs", "network.n_envs"), ("seed", "network.seed"),
                      ("variant", "train.variant"), ("backend", "train.backend"),
                      ("physics", "train.physics"), ("gsi", "train.gsi"),
                      ("controller", "train.controller"), ("name", "train.name")):
        if getattr(a, flag) is not None:
            sets.append((key, getattr(a, flag)))
    if a.from_snapshot and any(k.startswith("train.") and k.split(".")[1] in
                               ("variant", "backend", "physics", "gsi", "reward_w")
                               for k, _ in sets):
        settings.pop("frozen", None)
    for k, v in sets:
        S.set_path(settings, k, v)
    S.run(settings, out_root=a.out or ROOT / "outputs",
          command="python main.py train " + " ".join(argv))
    return 0


def cmd_sweep(argv):
    import yaml
    from arlamx_v2.sweep import run_sweep
    ap = argparse.ArgumentParser(prog="main.py sweep")
    ap.add_argument("--config", type=Path, default=CONFIG / "sweep.yaml")
    ap.add_argument("--network", type=Path, default=CONFIG / "network.yaml")
    ap.add_argument("--orbit", type=Path, default=CONFIG / "orbit.yaml")
    ap.add_argument("--train", type=Path, default=CONFIG / "train.yaml")
    ap.add_argument("--force", action="store_true", help="re-run arms already done")
    ap.add_argument("--out", type=Path, default=None, help="output root (default outputs/)")
    a = ap.parse_args(argv)
    cfg = yaml.safe_load(a.config.read_text()) or {}
    rows = run_sweep(cfg, force=a.force, out_root=a.out or ROOT / "outputs",
                     files={"network": a.network, "orbit": a.orbit, "train": a.train})
    for r in rows:
        print(f"{r['label']:<40} {r.get('status', ''):<8} {r.get('id', '')}")
    return 0 if all(r.get("status") == "done" for r in rows) else 1


def cmd_decay(argv):
    from arlamx_v2 import session as S
    ap = argparse.ArgumentParser(prog="main.py decay")
    ap.add_argument("--orbit", type=Path, default=CONFIG / "orbit.yaml")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="e.g. orbit.simulation.alt0_km=450")
    ap.add_argument("--kind", choices=("min", "max", "both"), default="both")
    ap.add_argument("--physics", default="standard")
    ap.add_argument("--gsi", choices=("sentman", "cll"))
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    settings = {"network": {}, "orbit": S.read_yaml(a.orbit), "train": {}}
    for k, v in (S.parse_set(x) for x in a.set):
        S.set_path(settings, k, v)
    sim = settings["orbit"].get("simulation") or {}
    out = a.out or ROOT / "outputs" / "results" / "decay" / (
        f"{datetime.now():%m-%d-%H-%M}_{a.kind}_{Path(str(a.physics)).stem}")
    S.write_yaml({"simulation": sim, "kind": a.kind, "physics": a.physics, "gsi": a.gsi,
                  "command": "python main.py decay " + " ".join(argv)}, out / "settings.yaml")
    fw = ["--kind", a.kind, "--physics", str(a.physics), "--out", str(out)]
    for key, flag in (("alt0_km", "--alt0"), ("alt_stop_km", "--alt-stop"), ("inc_deg", "--inc"),
                      ("ecc", "--ecc"), ("f107", "--f107"), ("ap", "--ap"), ("mass_kg", "--mass"),
                      ("dt_s", "--dt"), ("max_days", "--max-days"), ("geom", "--geom")):
        if sim.get(key) is not None:
            fw += [flag, str(sim[key])]
    if a.gsi:
        fw += ["--gsi", a.gsi]
    return forward("decay_run", fw)


def cmd_snapshots(argv):
    from arlamx_v2.session import list_snapshots
    rows = list_snapshots(ROOT / "outputs")
    if not rows:
        print("no snapshots in outputs/snapshots")
    for r in rows:
        print(f"{r['id']:<36} {str(r['status']):<11} {str(r['backend']):<6} "
              f"{str(r['variant']):<5} {str(r['timesteps']):>9} steps  {r['wall_s']} s")
    return 0


def cmd_tool(group, argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(f"python main.py {group} <tool> [flags]\n")
        for name, purpose in TOOLS[group].items():
            print(f"  {name:<20} {purpose}")
        return 0
    tool, rest = argv[0], argv[1:]
    if tool not in TOOLS[group]:
        raise SystemExit(f"unknown {group} tool {tool!r}; python main.py {group} -h")
    if tool in NO_FLAGS and any(x in ("-h", "--help") for x in rest):
        print(f"{tool}: {TOOLS[group][tool]}\nTakes no flags; `python main.py {group} {tool}` runs it now.")
        return 0
    return forward(tool, rest)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("list", "help", "-h", "--help"):
        print_list()
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "train":
        return cmd_train(rest)
    if cmd == "sweep":
        return cmd_sweep(rest)
    if cmd == "decay":
        return cmd_decay(rest)
    if cmd == "snapshots":
        return cmd_snapshots(rest)
    if cmd == "quantize":
        return forward("quantize", rest)
    if cmd in TOOLS:
        return cmd_tool(cmd, rest)
    if cmd == "run":
        if not rest:
            raise SystemExit("python main.py run <module> [flags]")
        return forward(rest[0], rest[1:])
    if cmd == "test":
        return subprocess.call([sys.executable, "-m", "pytest", str(ROOT / "tests"), "-q", *rest],
                               env={**os.environ, "PYTHONPATH": str(ROOT / "python")})
    if cmd == "verify":
        return subprocess.call([sys.executable, str(ROOT / "tests" / "validation" / "verify_physics.py")],
                               env={**os.environ, "PYTHONPATH": str(ROOT / "python")})
    raise SystemExit(f"unknown command {cmd!r}. python main.py list")


if __name__ == "__main__":
    raise SystemExit(main())
