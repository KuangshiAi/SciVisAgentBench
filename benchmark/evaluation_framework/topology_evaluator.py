"""
Rule-Based Topology Evaluator

Handles evaluation of topology tasks using custom eval scripts
defined in the YAML test cases with type: rule_based assertions.

Each assertion specifies:
  - eval_script: Path to a Python script containing the eval function
  - eval_function: Name of the function to call
  - gs_file: Ground truth file(s) (string or list)
  - rs_file: Result file(s) (string or list, optional - derived from gs_file if missing)
"""

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def _resolve_paths(
    file_spec: Union[str, List[str]],
    base_dir: str,
    agent_mode: str,
) -> List[str]:
    """
    Resolve file paths relative to base_dir and replace {agent_mode} placeholder.

    Args:
        file_spec: A single path string or list of path strings.
        base_dir: The base directory (tasks dir) to resolve relative paths against.
        agent_mode: The agent mode string to substitute for {agent_mode}.

    Returns:
        List of resolved absolute path strings.
    """
    if isinstance(file_spec, str):
        file_spec = [file_spec]

    resolved = []
    for p in file_spec:
        p = p.replace("{agent_mode}", agent_mode)
        full = Path(base_dir) / p
        resolved.append(str(full.resolve()))
    return resolved


def _derive_rs_from_gs(
    gs_files: List[str],
    agent_mode: str,
) -> List[str]:
    """
    Derive result file paths from ground-truth paths by replacing /GS/ with /results/{agent_mode}/.
    """
    rs_files = []
    for gs in gs_files:
        rs = gs.replace("/GS/", f"/results/{agent_mode}/")
        rs_files.append(rs)
    return rs_files


def _load_eval_function(script_path: str, function_name: str):
    """
    Dynamically load a Python function from a script file.

    Args:
        script_path: Absolute path to the Python script.
        function_name: Name of the function to load.

    Returns:
        The loaded function object.
    """
    spec = importlib.util.spec_from_file_location("eval_module", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {script_path}")

    module = importlib.util.module_from_spec(spec)

    # Add the script's directory to sys.path so it can import siblings (e.g. topologyScoring)
    script_dir = str(Path(script_path).parent)
    tasks_dir = str(Path(script_path).parent.parent.parent)  # up from dataset/GS/ to topology/
    original_path = sys.path[:]
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    if tasks_dir not in sys.path:
        sys.path.insert(0, tasks_dir)

    try:
        spec.loader.exec_module(module)
    finally:
        sys.path = original_path

    func = getattr(module, function_name, None)
    if func is None:
        raise AttributeError(
            f"Function '{function_name}' not found in {script_path}"
        )
    return func


def run_rule_based_evaluation(
    assertion: Dict[str, Any],
    data_dir: str,
    agent_mode: str,
) -> Dict[str, Any]:
    """
    Execute a single rule_based assertion and return the evaluation result.

    Args:
        assertion: Dictionary from the YAML assert entry, containing:
            - eval_script: relative path to eval Python script
            - eval_function: name of function to call
            - gs_file: ground-truth file(s)
            - rs_file: (optional) result file(s)
        data_dir: Base directory for resolving relative paths (the topology tasks dir).
        agent_mode: Agent mode string (e.g. "mcp").

    Returns:
        Dictionary with keys:
            - score: float (0-10 scale from eval function)
            - max_score: 10
            - eval_script: script used
            - eval_function: function called
            - gs_files: resolved ground-truth paths
            - rs_files: resolved result paths
            - status: "completed" or "error"
            - error: error message if any
    """
    result = {
        "score": 0,
        "max_score": 10,
        "eval_script": assertion.get("eval_script", ""),
        "eval_function": assertion.get("eval_function", ""),
        "gs_files": [],
        "rs_files": [],
        "status": "pending",
        "error": None,
    }

    try:
        # Resolve eval script path
        eval_script_rel = assertion["eval_script"]
        eval_script_path = str((Path(data_dir) / eval_script_rel).resolve())
        result["eval_script"] = eval_script_path

        # Resolve ground-truth files
        gs_spec = assertion.get("gs_file", [])
        gs_files = _resolve_paths(gs_spec, data_dir, agent_mode)
        result["gs_files"] = gs_files

        # Resolve result files
        rs_spec = assertion.get("rs_file")
        if rs_spec:
            rs_files = _resolve_paths(rs_spec, data_dir, agent_mode)
        else:
            # Derive from gs paths
            rs_files = _derive_rs_from_gs(gs_files, agent_mode)
        result["rs_files"] = rs_files

        # Check that all files exist
        missing = []
        for f in gs_files:
            if not Path(f).exists():
                missing.append(f"GS: {f}")
        for f in rs_files:
            if not Path(f).exists():
                missing.append(f"RS: {f}")
        if missing:
            result["status"] = "error"
            result["error"] = f"Missing files: {', '.join(missing)}"
            return result

        # Load the eval function
        eval_func = _load_eval_function(eval_script_path, assertion["eval_function"])

        # Call with interleaved args: gs1, gs2, ..., rs1, rs2, ...
        # (matching the convention used by the topology eval scripts)
        args = gs_files + rs_files
        score = eval_func(*args)

        result["score"] = float(score)
        result["status"] = "completed"

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    return result


def evaluate_rule_based_assertions(
    assertions: List[Dict[str, Any]],
    data_dir: str,
    agent_mode: str,
) -> Dict[str, Any]:
    """
    Evaluate all rule_based assertions for a test case.

    Args:
        assertions: List of rule_based assertion dicts from YAML.
        data_dir: Base directory for resolving paths.
        agent_mode: Agent mode string.

    Returns:
        Dictionary with overall evaluation results:
            - status: "completed" or "error"
            - assertion_results: list of individual results
            - score: average score across assertions (0-10)
            - max_score: 10
    """
    results = []
    total_score = 0.0

    for assertion in assertions:
        res = run_rule_based_evaluation(assertion, data_dir, agent_mode)
        results.append(res)
        total_score += res["score"]

    n = len(results) if results else 1
    avg_score = total_score / n

    overall_status = "completed"
    for r in results:
        if r["status"] == "error":
            overall_status = "partial_error"
            break

    return {
        "eval_type": "rule_based",
        "status": overall_status,
        "assertion_results": results,
        "score": avg_score,
        "max_score": 10,
        "num_assertions": len(results),
    }
