import re
import json
import os
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional
from openai import OpenAI
import tiktoken

HERE      = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
print(f"REPO_ROOT: {REPO_ROOT}")
BATCH_DIR = REPO_ROOT / "cases"

# ————————————————
# 2) TOKEN COUNTER
# ————————————————
class TokenCounter:
    """Utility class for counting tokens using GPT-4o tokenizer."""
    
    def __init__(self):
        """Initialize the tokenizer for GPT-4o."""
        try:
            self.tokenizer = tiktoken.encoding_for_model("gpt-4o")
        except Exception as e:
            print(f"Warning: Failed to initialize tokenizer: {e}")
            self.tokenizer = None
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in the given text."""
        if self.tokenizer is None:
            return 0
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            print(f"Warning: Failed to count tokens: {e}")
            return 0

# ————————————————
# 3) PRICING INFO
# ————————————————
# Default pricing for GPT-4o (can be overridden via config if needed)
DEFAULT_PRICING = {
    "input_per_1m_tokens": "$2.50",
    "output_per_1m_tokens": "$10.00"
}

def calculate_cost(input_tokens: int, output_tokens: int, pricing_info: Optional[Dict] = None) -> Optional[Dict]:
    """Calculate cost based on token usage and pricing info."""
    if pricing_info is None:
        pricing_info = DEFAULT_PRICING
    
    try:
        # Parse pricing strings (e.g., "$10.00" -> 10.00)
        input_price_str = pricing_info.get("input_per_1m_tokens", "").replace("$", "")
        output_price_str = pricing_info.get("output_per_1m_tokens", "").replace("$", "")
        
        input_price_per_1m = float(input_price_str)
        output_price_per_1m = float(output_price_str)
        
        # Calculate costs
        input_cost = (input_tokens / 1_000_000) * input_price_per_1m
        output_cost = (output_tokens / 1_000_000) * output_price_per_1m
        total_cost = input_cost + output_cost
        
        return {
            "input_cost": round(input_cost, 6),
            "output_cost": round(output_cost, 6),
            "total_cost": round(total_cost, 6),
            "currency": "USD",
            "pricing_model": {
                "input_per_1m_tokens": pricing_info["input_per_1m_tokens"],
                "output_per_1m_tokens": pricing_info["output_per_1m_tokens"]
            }
        }
    except (ValueError, KeyError) as e:
        print(f"Warning: Could not calculate cost: {e}")
        return None

# ————————————————
# 4) OPENAI CLIENT
# ————————————————
openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# ————————————————
# 5) PARAVIEW EXECUTION HELPERS
# ————————————————
def execute_paraview_code(code: str) -> str:
    """Execute ParaView Python code directly using pvpython"""
    try:
        # Create a temporary file with the code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_script = f.name
        
        # Try to find pvpython in common locations
        pvpython_paths = [
            '/Applications/ParaView-5.13.3.app/Contents/bin/pvpython',  # macOS
        ]
        
        pvpython_cmd = None
        for path in pvpython_paths:
            try:
                result = subprocess.run([path, '--version'], capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    pvpython_cmd = path
                    break
            except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
                continue
        
        if not pvpython_cmd:
            return "Error: pvpython not found. Please install ParaView or add pvpython to your PATH."
        
        # Execute the script
        result = subprocess.run([pvpython_cmd, temp_script], 
                              capture_output=True, text=True, timeout=120)
        
        # Clean up temp file
        Path(temp_script).unlink()
        
        if result.returncode == 0:
            return result.stdout if result.stdout else "Execution completed successfully"
        else:
            return f"Error executing ParaView script:\n{result.stderr}"
            
    except Exception as e:
        return f"Error: {str(e)}"

# ————————————————
# 6) LLM & UTILS
# ————————————————
def extract_python(text: str) -> str:
    m = re.search(r"```python(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()

def extract_errors(output: str) -> list:
    return [ln for ln in output.splitlines() if "Error" in ln or "Traceback" in ln]

def save_script(code: str, case: str, case_dir: Path):
    generated_dir = Path(os.path.join(case_dir, "results", "pvpython"))
    generated_dir.mkdir(parents=True, exist_ok=True)
    fn = generated_dir / f"{case}.py"
    fn.write_text(code, encoding="utf-8")
    print(f"    • script saved → {fn.name}")

def save_test_result(case_path: Path, result: Dict):
    """Save the result of a test case to pvpython subdirectory."""
    # Create test_results/pvpython subdirectory
    test_results_dir = case_path / "test_results" / "pvpython"
    test_results_dir.mkdir(parents=True, exist_ok=True)
    
    result_file = test_results_dir / f"test_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    try:
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"  ✓ result saved to: {result_file}")
        
        # Print token usage summary if available
        if "token_usage" in result:
            token_usage = result["token_usage"]
            print(f"  → Token usage - Input: {token_usage['input_tokens']}, Output: {token_usage['output_tokens']}, Total: {token_usage['total_tokens']}")
        
        # Print cost summary if available
        if "cost_info" in result:
            cost_info = result["cost_info"]
            print(f"  → Cost - Input: ${cost_info['input_cost']:.6f}, Output: ${cost_info['output_cost']:.6f}, Total: ${cost_info['total_cost']:.6f}")
            
    except Exception as e:
        print(f"  ✘ Failed to save result: {e}")

# ————————————————
# 7) RUN ONE CASE (with iterative fixes)
# ————————————————
def run_case(case_path: Path) -> Dict:
    case = case_path.name
    print(f"\n=== {case} ===")

    # Initialize token counter
    token_counter = TokenCounter()
    
    # Initialize result structure
    start_time = datetime.now()
    result = {
        "case_name": case,
        "status": "running",
        "start_time": start_time.isoformat(),
        "task_description": "",
        "response": "",
        "error": None,
        "token_usage": {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0
        }
    }

    # Look for .raw files in the data/ subdirectory
    data_dir = case_path / "data"
    raw_file = next(data_dir.glob("*.raw"), None) if data_dir.exists() else None
    
    # Look for task_description.txt instead of task_description.txt
    task_file = case_path / "task_description.txt"
    
    # Look for other .txt files in data/ subdirectory (like bonsai.txt)
    attr_file = None
    if data_dir.exists():
        attr_file = next((p for p in data_dir.glob("*.txt")), None)

    if not raw_file or not task_file.exists():
        print(f"  ✘ missing .raw file in data/ or task_description.txt")
        print(f"    Raw file: {raw_file}")
        print(f"    Task file: {task_file}")
        result["status"] = "error"
        result["error"] = f"Missing .raw file in data/ or task_description.txt. Raw file: {raw_file}, Task file: {task_file}"
        end_time = datetime.now()
        result["end_time"] = end_time.isoformat()
        result["duration"] = (end_time - start_time).total_seconds()
        save_test_result(case_path, result)
        return result

    # 1) get task description
    def get_task_description(case_path: Path) -> str:
        """Read and return the task description."""
        task_description_file = case_path / "task_description.txt"
        
        if not task_description_file.exists():
            raise FileNotFoundError(f"Task description not found: {task_description_file}")
        
        working_dir = case_path.parent
        
        # Read original task description
        with open(task_description_file, 'r', encoding='utf-8') as f:
            original_task = f.read().strip()
        
        # Prepend working directory information
        working_dir_info = f'Your agent_mode is "pvpython", use it when saving results. Your working directory is "{working_dir}", and you should have access to it. In the following prompts, we will use relative path with respect to your working path. But remember, when you load or save any file, always stick to absolute path.'
        
        return f"{working_dir_info}\n\n{original_task}"
    
    try:
        prompt = get_task_description(case_path)
        result["task_description"] = prompt
        
        # Count input tokens
        input_tokens = token_counter.count_tokens(prompt)
        result["token_usage"]["input_tokens"] = input_tokens
        result["token_usage"]["total_tokens"] = input_tokens
        
        print("  ✓ task description loaded")
    except (FileNotFoundError, ValueError) as e:
        print(f"  ✘ {e}")
        result["status"] = "error"
        result["error"] = str(e)
        end_time = datetime.now()
        result["end_time"] = end_time.isoformat()
        result["duration"] = (end_time - start_time).total_seconds()
        save_test_result(case_path, result)
        return result

    # 2) generate script
    sr = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role":"system","content":(
                "You are a code assistant. Output only a self-contained Python script for ParaView:\n"
                "- Call ResetSession() once\n"
                "- Import from paraview.simple only needed symbols\n"
                "- Read raw via ImageReader(...) and configure DataScalarType, DataByteOrder, DataExtent, DataSpacing\n"
                "- UpdatePipeline(), Show(...), Representation='Volume'\n"
                "- Configure GetColorTransferFunction and GetOpacityTransferFunction\n"
                "- SaveState(state_path) at end"
            )},
            {"role":"user","content":prompt}
        ]
    )
    
    # Count output tokens and update token usage
    response_content = sr.choices[0].message.content
    output_tokens = token_counter.count_tokens(response_content)
    result["token_usage"]["output_tokens"] += output_tokens
    result["token_usage"]["total_tokens"] = result["token_usage"]["input_tokens"] + result["token_usage"]["output_tokens"]
    
    # Calculate cost
    cost_info = calculate_cost(result["token_usage"]["input_tokens"], result["token_usage"]["output_tokens"])
    if cost_info:
        result["cost_info"] = cost_info
    
    code = extract_python(response_content)
    code = "from paraview.simple import ResetSession\nResetSession()\n" + code
    
    # Store the response
    result["response"] = response_content

    # 3) syntax check
    try:
        compile(code, "<string>", "exec")
        print("  ✓ syntax")
    except SyntaxError as e:
        print("  ✘ syntax error:", e)
        result["status"] = "error"
        result["error"] = f"Syntax error: {e}"
        end_time = datetime.now()
        result["end_time"] = end_time.isoformat()
        result["duration"] = (end_time - start_time).total_seconds()
        save_test_result(case_path, result)
        return result

    # 4) save initial script
    save_script(code, case, case_path)

    # 5) execution + iterative fix
    max_attempts = 5
    out = execute_paraview_code(code)
    errors = extract_errors(out)
    attempt = 1
    while errors and attempt < max_attempts:
        attempt += 1
        print(f"  → fix attempt {attempt} errors: {errors}")
        fix = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role":"system","content":"You are a code fixer. Return only corrected Python code."},
                {"role":"user","content":(
                    f"Script:\n```python\n{code}\n```\nErrors:\n" + "\n".join(errors)
                )}
            ]
        )
        
        # Count tokens for fix attempts
        fix_input = f"Script:\n```python\n{code}\n```\nErrors:\n" + "\n".join(errors)
        fix_response = fix.choices[0].message.content
        fix_input_tokens = token_counter.count_tokens(fix_input)
        fix_output_tokens = token_counter.count_tokens(fix_response)
        
        # Update token usage
        result["token_usage"]["input_tokens"] += fix_input_tokens
        result["token_usage"]["output_tokens"] += fix_output_tokens
        result["token_usage"]["total_tokens"] = result["token_usage"]["input_tokens"] + result["token_usage"]["output_tokens"]
        
        # Update cost
        cost_info = calculate_cost(result["token_usage"]["input_tokens"], result["token_usage"]["output_tokens"])
        if cost_info:
            result["cost_info"] = cost_info
        
        # Append to response
        result["response"] += f"\n\nFix attempt {attempt}:\n{fix_response}"
        
        code = extract_python(fix_response)
        if not code.startswith("from paraview.simple import ResetSession"):
            code = "from paraview.simple import ResetSession\nResetSession()\n" + code
        try:
            compile(code, "<string>", "exec")
        except SyntaxError as e:
            print("  ✘ syntax error after fix:", e)
            result["status"] = "error"
            result["error"] = f"Syntax error after fix: {e}"
            end_time = datetime.now()
            result["end_time"] = end_time.isoformat()
            result["duration"] = (end_time - start_time).total_seconds()
            save_test_result(case_path, result)
            return result
        out = execute_paraview_code(code)
        errors = extract_errors(out)

    if errors:
        print("  ✘ final errors:", errors)
        result["status"] = "error"
        result["error"] = f"Final errors: {errors}"
        end_time = datetime.now()
        result["end_time"] = end_time.isoformat()
        result["duration"] = (end_time - start_time).total_seconds()
        save_test_result(case_path, result)
        return result

    # 6) verify state file
    state_dir = Path(os.path.join(case_path, "results", "pvpython"))
    state_dir.mkdir(parents=True, exist_ok=True)
    pvsm = state_dir / f"{case}.pvsm"
    if pvsm.exists():
        print("  ✓ state saved")
        result["status"] = "completed"
        end_time = datetime.now()
        result["end_time"] = end_time.isoformat()
        result["duration"] = (end_time - start_time).total_seconds()
        save_test_result(case_path, result)
        return result
    else:
        print("  ✘ missing state file")
        result["status"] = "error"
        result["error"] = "Missing state file"
        end_time = datetime.now()
        result["end_time"] = end_time.isoformat()
        result["duration"] = (end_time - start_time).total_seconds()
        save_test_result(case_path, result)
        return result

# ————————————————
# 8) MAIN
# ————————————————
def main():
    results = {}
    all_results = []
    start_time = datetime.now()
    
    for case_dir in sorted(BATCH_DIR.iterdir()):
        if case_dir.is_dir():
            result = run_case(case_dir)
            results[case_dir.name] = result["status"] == "completed"
            all_results.append(result)

    end_time = datetime.now()
    
    # Calculate summary statistics
    completed = sum(1 for r in all_results if r["status"] == "completed")
    errors = sum(1 for r in all_results if r["status"] == "error")
    total_input_tokens = sum(r.get("token_usage", {}).get("input_tokens", 0) for r in all_results)
    total_output_tokens = sum(r.get("token_usage", {}).get("output_tokens", 0) for r in all_results)
    total_tokens = total_input_tokens + total_output_tokens
    
    # Calculate total cost
    total_cost_info = calculate_cost(total_input_tokens, total_output_tokens)
    
    print("\n--- Summary ---")
    for case, ok in results.items():
        print(f"{case}: {'PASS' if ok else 'FAIL'}")
    
    print(f"\nOverall Statistics:")
    print(f"Total cases: {len(all_results)}")
    print(f"Completed: {completed}")
    print(f"Errors: {errors}")
    print(f"Duration: {(end_time - start_time).total_seconds():.2f} seconds")
    print(f"Token usage - Input: {total_input_tokens}, Output: {total_output_tokens}, Total: {total_tokens}")
    if total_cost_info:
        print(f"Total cost - Input: ${total_cost_info['input_cost']:.6f}, Output: ${total_cost_info['output_cost']:.6f}, Total: ${total_cost_info['total_cost']:.6f}")
    
    # Save overall summary
    summary_dir = REPO_ROOT / "test_results" / "pvpython"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_file = summary_dir / f"test_run_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    summary_data = {
        "results": all_results,
        "summary": {
            "total": len(all_results),
            "completed": completed,
            "errors": errors,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration": (end_time - start_time).total_seconds(),
            "token_usage": {
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "total_tokens": total_tokens
            }
        }
    }
    
    if total_cost_info:
        summary_data["summary"]["cost_summary"] = total_cost_info
    
    try:
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print("TEST RUN SUMMARY")
        print(f"{'='*60}")
        print(f"Total test cases: {summary_data['summary']['total']}")
        print(f"Completed: {summary_data['summary']['completed']}")
        print(f"Errors: {summary_data['summary']['errors']}")
        print(f"Total duration: {summary_data['summary']['duration']:.2f} seconds")
        print(f"Token usage - Input: {summary_data['summary']['token_usage']['total_input_tokens']}, Output: {summary_data['summary']['token_usage']['total_output_tokens']}, Total: {summary_data['summary']['token_usage']['total_tokens']}")
        if "cost_summary" in summary_data["summary"]:
            cost_summary = summary_data["summary"]["cost_summary"]
            print(f"Total cost - Input: ${cost_summary['input_cost']:.6f}, Output: ${cost_summary['output_cost']:.6f}, Total: ${cost_summary['total_cost']:.6f}")
        print(f"Overall results saved to: {summary_file}")
    except Exception as e:
        print(f"Failed to save summary: {e}")

if __name__ == '__main__':
    main()