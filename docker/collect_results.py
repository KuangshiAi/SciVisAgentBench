#!/usr/bin/env python3
"""
Copy agent outputs from a sanitized sandbox back into the real cases tree.

After the agent runs in the container, each case's outputs live at
``<sandbox>/<case>/results/<agent_mode>/``. This copies them into the real
``<cases-dir>/<case>/results/<agent_mode>/`` so the normal evaluation pipeline
(which needs the ground truth under ``GS/``) can score them on the host:

    python -m benchmark.evaluation_framework.run_evaluation \
        --agent <agent> --config <config.json> \
        --yaml <cases.yaml> --cases <cases-dir> \
        --eval-only --agent-mode <agent_mode>

Only the ``results/`` subtree is copied back — never anything that could change
the ground truth.

Example
-------
    python docker/collect_results.py \
        --sandbox sandbox/paraview \
        --cases-dir SciVisAgentBench-tasks/paraview \
        --agent-mode docker_claude_code_exp1
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sandbox", required=True,
                    help="Sandbox directory the container wrote into")
    ap.add_argument("--cases-dir", required=True,
                    help="Real cases directory to copy results back into")
    ap.add_argument("--agent-mode", default=None,
                    help="Only copy this results/<agent_mode> subdir (default: all)")
    ap.add_argument("--case", default=None,
                    help="Only copy this single case's results (default: all cases)")
    ap.add_argument("--dry-run", action="store_true",
                    help="List what would be copied without copying")
    args = ap.parse_args()

    sandbox = Path(args.sandbox).resolve()
    cases_dir = Path(args.cases_dir).resolve()

    if not sandbox.is_dir():
        print(f"ERROR: sandbox not found: {sandbox}", file=sys.stderr)
        return 2
    if not cases_dir.is_dir():
        print(f"ERROR: cases-dir not found: {cases_dir}", file=sys.stderr)
        return 2

    # Find every results/<mode> directory the agent produced in the sandbox, at
    # ANY depth, and mirror it back preserving the sandbox-relative path:
    #   paraview:  <case>/results/<mode>
    #   bioimage:  eval_visualization_tasks/<case>/results/<mode>
    copied = 0
    for results_dir in sorted(sandbox.rglob("results")):
        if not results_dir.is_dir():
            continue
        rel = results_dir.relative_to(sandbox)          # e.g. <case>/results
        if list(rel.parts).count("results") > 1:
            continue                                     # skip a results/ nested in another
        case = results_dir.parent.name                   # <case> for the --case filter
        if args.case and case != args.case:
            continue
        mode_dirs = ([results_dir / args.agent_mode] if args.agent_mode
                     else sorted(p for p in results_dir.iterdir() if p.is_dir()))
        for mode_dir in mode_dirs:
            if not mode_dir.is_dir():
                if args.agent_mode:
                    print(f"  (skip) {case}: no results/{args.agent_mode}")
                continue
            dest = cases_dir / rel / mode_dir.name        # preserve relative path
            print(f"  {mode_dir}  ->  {dest}")
            if not args.dry_run:
                dest.mkdir(parents=True, exist_ok=True)
                # Copy file-by-file so existing sibling result-modes are kept.
                for item in mode_dir.rglob("*"):
                    rel = item.relative_to(mode_dir)
                    target = dest / rel
                    if item.is_dir():
                        target.mkdir(parents=True, exist_ok=True)
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(item, target)
            copied += 1

    if copied == 0:
        print("⚠ No results/<mode> directories found in the sandbox.")
        return 1
    verb = "Would copy" if args.dry_run else "Copied"
    print(f"✓ {verb} {copied} result set(s) back into {cases_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
