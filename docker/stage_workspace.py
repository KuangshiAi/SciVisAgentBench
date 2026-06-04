#!/usr/bin/env python3
"""
Stage a *sanitized* copy of a benchmark cases directory for an agent to work in.

The agent runs inside the Docker container with this sanitized tree mounted at
/workspace. It contains everything the agent legitimately needs (the task
prompt and the input ``data/`` for each case) but **none** of the ground truth
or evaluation labels:

  * ``GS/``                  ground-truth state files, images and eval scripts
  * ``visualization_goals.txt``  the per-case evaluation rubric
  * ``results/`` ``test_results/`` ``evaluation_results/``  prior run artifacts
  * ``*_gs.*`` / ``gs_*``    stray ground-truth files outside ``GS/``

After the container exits, copy the agent's outputs back into the real tree with
``collect_results.py`` and run evaluation on the host (which *does* have GS).

Examples
--------
    # Stage the whole paraview category into ./sandbox/paraview
    python docker/stage_workspace.py \
        --cases-dir SciVisAgentBench-tasks/paraview \
        --out sandbox/paraview

    # Stage only two cases
    python docker/stage_workspace.py \
        --cases-dir SciVisAgentBench-tasks/paraview \
        --out sandbox/paraview --cases vortex foot
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

# Directory names that are stripped at any depth.
EXCLUDE_DIRS = {
    "GS",                  # ground truth (state, images, eval scripts)
    "results",            # agent outputs from previous runs
    "test_results",       # execution metadata from previous runs
    "evaluation_results", # scores from previous runs
    "__pycache__",
    ".cache",
    ".git",
    ".ipynb_checkpoints",
}

# File names / glob patterns that are stripped at any depth (defense in depth —
# most ground truth already lives under GS/, but stray copies are caught here).
EXCLUDE_FILES = {
    "visualization_goals.txt",  # per-case evaluation rubric (a label)
}
EXCLUDE_PATTERNS = [
    "*_gs.*",          # foo_gs.png, foo_gs.pvsm, foo_gs.py, foo_gs.vti, screenshot_gs.sh, ...
    "gs_*",            # gs_front_view.png, gs_side_view.png, ...
    "*_eval.py",       # ground-truth evaluation scripts (e.g. cylinder_eval.py)
    "*_from_gs.py",    # helpers that derive answers from ground truth (e.g. get_tf_from_gs.py)
    "*_verbose_*.log", # prior agent run transcripts (claude_code_verbose_*, codex_cli_verbose_*)
    "eval_*",          # eval target/answer files (e.g. bioimage eval_*_target_*.txt)
    "*[Ss]coring.py",  # suite-level scoring algorithms (e.g. topology/topologyScoring.py)
]
# NOTE: deliberately NOT excluding "*ground_truth*": some cases take a file named
# *_ground_truth.* as a legitimate INPUT (e.g. paraview/materials reads
# materials_ground_truth.vtr into a side-by-side comparison). The real answer for
# those lives under GS/, which is excluded as a directory.

# Patterns the post-copy audit treats as a ground-truth leak (must be empty).
LEAK_PATTERNS = ["*_gs.*", "gs_*", "*_eval.py", "eval_*", "*[Ss]coring.py",
                 "visualization_goals.txt"]


def _is_excluded_file(name: str) -> bool:
    if name in EXCLUDE_FILES:
        return True
    return any(fnmatch.fnmatch(name, pat) for pat in EXCLUDE_PATTERNS)


def _ignore(dir_path: str, names: list[str]) -> set[str]:
    """shutil.copytree ignore-callback: drop excluded dirs and files."""
    ignored: set[str] = set()
    base = Path(dir_path)
    for name in names:
        full = base / name
        if full.is_dir():
            if name in EXCLUDE_DIRS:
                ignored.add(name)
        elif _is_excluded_file(name):
            ignored.add(name)
    return ignored


# Files smaller than this are skipped by the content-dedup pass: no meaningful
# ground-truth answer is a handful of bytes, and it avoids pathological hash
# collisions on tiny/empty files.
_MIN_DEDUP_BYTES = 64


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _gs_content_index(src_root: Path) -> dict[int, set[str]]:
    """size -> {md5, ...} for every file under a GS/ directory in the source.

    Keyed by size so the dedup pass can skip hashing staged files whose size
    matches no ground-truth file (the common case for large input volumes)."""
    index: dict[int, set[str]] = {}
    for gs_dir in src_root.rglob("GS"):
        if not gs_dir.is_dir():
            continue
        for p in gs_dir.rglob("*"):
            if not p.is_file():
                continue
            try:
                size = p.stat().st_size
            except OSError:
                continue
            if size < _MIN_DEDUP_BYTES:
                continue
            index.setdefault(size, set()).add(_md5(p))
    return index


def _strip_gs_content_duplicates(src_root: Path, out_dir: Path) -> list[str]:
    """Remove staged files whose *content* equals a ground-truth file.

    Catches answers copied into the sandbox under innocuous names that no naming
    rule would flag — e.g. bioimage_data/data/dataset_001/dataset_001.png is
    byte-identical to GS/dataset_001.png."""
    gs_index = _gs_content_index(src_root)
    if not gs_index:
        return []
    removed: list[str] = []
    for p in out_dir.rglob("*"):
        if not p.is_file():
            continue
        try:
            size = p.stat().st_size
        except OSError:
            continue
        if size in gs_index and size >= _MIN_DEDUP_BYTES and _md5(p) in gs_index[size]:
            removed.append(str(p))
            p.unlink()
    return removed


def _audit(out_dir: Path) -> list[str]:
    """Return any files under out_dir that look like leaked ground truth."""
    leaks: list[str] = []
    for path in out_dir.rglob("*"):
        if path.is_dir():
            if path.name in EXCLUDE_DIRS:
                leaks.append(str(path) + "  (excluded dir present!)")
            continue
        if any(fnmatch.fnmatch(path.name, pat) for pat in LEAK_PATTERNS):
            leaks.append(str(path))
    return leaks


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases-dir", required=True,
                    help="Source cases directory, e.g. SciVisAgentBench-tasks/paraview")
    ap.add_argument("--out", required=True,
                    help="Destination sandbox directory (will mirror cases-dir)")
    ap.add_argument("--cases", nargs="*", default=None,
                    help="Optional subset of case names to stage (default: all)")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite the destination if it already exists")
    args = ap.parse_args()

    cases_dir = Path(args.cases_dir).resolve()
    out_dir = Path(args.out).resolve()

    if not cases_dir.is_dir():
        print(f"ERROR: cases-dir not found: {cases_dir}", file=sys.stderr)
        return 2

    if out_dir.exists():
        if not args.force:
            print(f"ERROR: destination exists: {out_dir}\n"
                  f"       pass --force to overwrite.", file=sys.stderr)
            return 2
        shutil.rmtree(out_dir)

    # Decide which top-level entries to stage.
    if args.cases:
        entries = [cases_dir / c for c in args.cases]
        missing = [str(e) for e in entries if not e.exists()]
        if missing:
            print("ERROR: requested cases not found:\n  " + "\n  ".join(missing),
                  file=sys.stderr)
            return 2
    else:
        # Everything in the category except excluded dirs / files.
        entries = sorted(p for p in cases_dir.iterdir()
                         if not (p.is_dir() and p.name in EXCLUDE_DIRS)
                         and not (p.is_file() and _is_excluded_file(p.name)))

    out_dir.mkdir(parents=True, exist_ok=True)
    staged_cases: list[str] = []
    for entry in entries:
        dest = out_dir / entry.name
        if entry.is_dir():
            shutil.copytree(entry, dest, ignore=_ignore, symlinks=False)
            staged_cases.append(entry.name)
        else:
            shutil.copy2(entry, dest)

    # Content-level defense: drop any staged file byte-identical to a GS file
    # (answers duplicated under innocuous names that no naming rule catches).
    content_dupes = _strip_gs_content_duplicates(cases_dir, out_dir)

    # Audit: refuse to produce a sandbox that contains ground truth.
    leaks = _audit(out_dir)
    if leaks:
        print("ERROR: ground-truth leak detected in sandbox — aborting:",
              file=sys.stderr)
        for leak in leaks:
            print(f"  {leak}", file=sys.stderr)
        shutil.rmtree(out_dir)
        return 3

    manifest = {
        "source_cases_dir": str(cases_dir),
        "sandbox_dir": str(out_dir),
        "staged_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "staged_cases": staged_cases,
        "excluded_dirs": sorted(EXCLUDE_DIRS),
        "excluded_files": sorted(EXCLUDE_FILES),
        "excluded_patterns": EXCLUDE_PATTERNS,
        "content_duplicates_removed": content_dupes,
        "ground_truth_included": False,
        "note": ("Sanitized for agent execution. No GS/ ground truth, eval "
                 "scripts, or visualization_goals rubric included."),
    }
    (out_dir / "STAGING_MANIFEST.json").write_text(json.dumps(manifest, indent=2))

    print(f"✓ Staged {len(staged_cases)} case(s) into: {out_dir}")
    if content_dupes:
        print(f"✓ Removed {len(content_dupes)} GS content-duplicate(s) from sandbox:")
        for d in content_dupes:
            print(f"    {Path(d).relative_to(out_dir)}")
    print(f"✓ Ground-truth audit passed (no GS / labels present).")
    print(f"  Mount this directory at /workspace inside the container.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
