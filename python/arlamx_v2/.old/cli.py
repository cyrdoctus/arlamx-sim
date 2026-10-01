"""Single command surface for ARLAMX.

    PYTHONPATH=python python -m arlamx_v2 list
    PYTHONPATH=python python -m arlamx_v2 train --variant v8a --algo ppo --physics high
    PYTHONPATH=python python -m arlamx_v2 sweep --config python/configs/sweep_example.yaml

Existing ``python -m arlamx_v2.train`` (and other module mains) stay callable.
"""
from __future__ import annotations

import runpy
import sys
from textwrap import dedent

# name, group, one-line purpose, example
CATALOG = [
    ("list", "meta", "Print this catalog", "python -m arlamx_v2 list"),
    ("help", "meta", "Reprint this catalog (flags: python -m arlamx_v2 <cmd> -h)",
     "python -m arlamx_v2 help"),
    ("train", "train",
     "PPO/SAC/TD3 trainer (FP32). --physics --gsi --controller (unset = YAML)",
     "python -m arlamx_v2 train --variant v8a --algo ppo --physics high"),
    ("sweep", "sweep",
     "YAML product / ledger runner (no LLM)",
     "python -m arlamx_v2 sweep --config python/configs/sweep_example.yaml"),
    ("quantize", "eval",
     "Dynamic INT8 of actor Linear layers (not used in train)",
     "python -m arlamx_v2 quantize --in PATH --out PATH"),
    ("eval mc_decay", "eval",
     "Monte-Carlo decay (v8/v10 policies)",
     "python -m arlamx_v2 eval mc_decay --runs 24"),
    ("eval mc_v12", "eval",
     "SC_v12 lifetime MC vs MPC",
     "python -m arlamx_v2 eval mc_v12 --runs 24"),
    ("decay", "physics",
     "Prescribed min/max-drag decay",
     "python -m arlamx_v2 decay --physics standard --kind both"),
]

# Modules that already have main(); listed as aliases, not rewritten.
ALIASES = {
    "train_v12": "arlamx_v2.train_v12",
    "train_v4": "arlamx_v2.train_v4",
    "train_v56": "arlamx_v2.train_v56",
    "aoa_decay": "arlamx_v2.aoa_decay",
    "lift_drag_study": "arlamx_v2.lift_drag_study",
    "srp_assess": "arlamx_v2.srp_assess",
    "validate_attitudes": "arlamx_v2.validate_attitudes",
    "bench_v8": "arlamx_v2.bench_v8",
    "mc_decay": "arlamx_v2.mc_decay",
    "mc_v12": "arlamx_v2.mc_v12",
    "mc_four": "arlamx_v2.mc_four",
    "plot_v8": "arlamx_v2.plot_v8",
    "plot_v12": "arlamx_v2.plot_v12",
    "campaign": "arlamx_v2.campaign",
    "quantize": "arlamx_v2.quantize",
    "decay_run": "arlamx_v2.decay_run",
}


def _print_list():
    print("ARLAMX commands  (python -m arlamx_v2 <cmd>)\n")
    groups = []
    seen = set()
    for _name, group, _purp, _ex in CATALOG:
        if group not in seen:
            seen.add(group)
            groups.append(group)
    for group in groups:
        print(f"  [{group}]")
        for name, g, purpose, example in CATALOG:
            if g != group:
                continue
            print(f"    {name:<16} {purpose}")
            print(f"                     e.g. {example}")
        print()
    print("  [aliases]  forwarded to python -m arlamx_v2.<mod>")
    for alias, mod in sorted(ALIASES.items()):
        if alias in ("quantize",):
            continue
        print(f"    {alias:<16} {mod}")
    print("\nPer-command flags: python -m arlamx_v2 <cmd> -h")
    print("Physics presets:   fast | standard | high | path/to.yaml")


def _forward_module(modname, argv):
    old = sys.argv
    sys.argv = [f"{modname}", *argv]
    try:
        runpy.run_module(modname, run_name="__main__", alter_sys=True)
    finally:
        sys.argv = old


def _cmd_train(argv):
    from arlamx_v2.train import main as train_main
    old = sys.argv
    sys.argv = ["arlamx_v2.train", *argv]
    try:
        return train_main()
    finally:
        sys.argv = old


def _cmd_sweep(argv):
    from arlamx_v2.sweep import main as sweep_main
    old = sys.argv
    sys.argv = ["arlamx_v2.sweep", *argv]
    try:
        return sweep_main()
    finally:
        sys.argv = old


def _cmd_quantize(argv):
    from arlamx_v2.quantize import main as qmain
    old = sys.argv
    sys.argv = ["arlamx_v2.quantize", *argv]
    try:
        return qmain()
    finally:
        sys.argv = old


def _cmd_decay(argv):
    from arlamx_v2.decay_run import main as dmain
    old = sys.argv
    sys.argv = ["arlamx_v2.decay_run", *argv]
    try:
        return dmain()
    finally:
        sys.argv = old


def _cmd_eval(argv):
    if not argv or argv[0] in ("-h", "--help"):
        print(dedent("""\
            usage: python -m arlamx_v2 eval {mc_decay,mc_v12} [flags]

            Thin wrappers of the existing eval mains. Remaining flags are
            forwarded unchanged.
        """))
        return 0
    which, rest = argv[0], argv[1:]
    if which == "mc_decay":
        return _forward_module("arlamx_v2.mc_decay", rest)
    if which == "mc_v12":
        return _forward_module("arlamx_v2.mc_v12", rest)
    raise SystemExit(f"unknown eval subcommand {which!r} (mc_decay|mc_v12)")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("list", "help", "-h", "--help"):
        _print_list()
        if argv and argv[0] == "help":
            print()
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "train":
        return _cmd_train(rest) or 0
    if cmd == "sweep":
        return _cmd_sweep(rest) or 0
    if cmd == "quantize":
        return _cmd_quantize(rest) or 0
    if cmd == "decay":
        return _cmd_decay(rest) or 0
    if cmd == "eval":
        return _cmd_eval(rest) or 0
    if cmd in ALIASES:
        return _forward_module(ALIASES[cmd], rest) or 0
    # Tolerate python -m arlamx_v2 arlamx_v2.train style.
    if cmd.startswith("arlamx_v2."):
        return _forward_module(cmd, rest) or 0
    raise SystemExit(
        f"unknown command {cmd!r}. Run: python -m arlamx_v2 list")


if __name__ == "__main__":
    raise SystemExit(main())
