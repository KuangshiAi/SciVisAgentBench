import re
import json
import os
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List
from openai import OpenAI
import tiktoken
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Multi-provider client will be imported when needed

HERE      = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
print(f"REPO_ROOT: {REPO_ROOT}")
BATCH_DIR = REPO_ROOT / "cases"

# ————————————————
# 1) VECTOR DATABASE SETUP
# ————————————————
class ParaViewRAG:
    """Retrieval-Augmented Generation system for ParaView operations."""
    
    def __init__(self):
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            print(f"  ! Failed to load SentenceTransformer: {e}")
            self.model = None
        self.index = None
        self.metadata_lookup = []
        self.setup_operations_db()
    
    def setup_operations_db(self):
        """Setup the operations database from operations.json."""
        if self.model is None:
            print("  ! RAG disabled - SentenceTransformer not available")
            return
            
        operations_file = HERE / "operations.json"
        faiss_index_file = HERE / "paraview_operations_faiss.index"
        metadata_file = HERE / "metadata_lookup.pkl"
        
        # Try to load existing index
        try:
            if faiss_index_file.exists() and metadata_file.exists():
                import faiss
                self.index = faiss.read_index(str(faiss_index_file))
                with open(metadata_file, "rb") as f:
                    self.metadata_lookup = pickle.load(f)
                print("  ✓ Loaded existing FAISS index")
                return
        except Exception as e:
            print(f"  ! Failed to load existing index: {e}")
        
        # Create new index if operations.json exists
        if operations_file.exists():
            try:
                import faiss
                with open(operations_file, "r") as f:
                    operations_json = json.load(f)
                
                # Create embeddings
                d = self.model.get_sentence_embedding_dimension()
                self.index = faiss.IndexFlatL2(d)
                
                for op in operations_json:
                    text = op["name"] + " " + op["description"] + " " + op["code_snippet"]
                    embedding = self.model.encode(text, convert_to_numpy=True).astype(np.float32)
                    self.index.add(embedding.reshape(1, -1))
                    self.metadata_lookup.append(op)
                
                # Save index
                faiss.write_index(self.index, str(faiss_index_file))
                with open(metadata_file, "wb") as f:
                    pickle.dump(self.metadata_lookup, f)
                
                print(f"  ✓ Created FAISS index with {len(operations_json)} operations")
            except Exception as e:
                print(f"  ! Failed to create operations index: {e}")
                self.index = None
        else:
            print("  ! No operations.json found, RAG disabled")
    
    def search_similar_operations(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """Search for similar operations based on query text."""
        if self.index is None or self.index.ntotal == 0 or self.model is None:
            return []
        
        try:
            import faiss
            query_embedding = self.model.encode(query_text, convert_to_numpy=True).astype(np.float32)
            top_k = min(top_k, self.index.ntotal)
            distances, indices = self.index.search(query_embedding.reshape(1, -1), top_k)
            
            matches = []
            for idx in indices[0]:
                if idx < len(self.metadata_lookup):
                    matches.append(self.metadata_lookup[idx])
            return matches
        except Exception as e:
            print(f"  ! RAG search failed: {e}")
            return []

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
# 4) MULTI-PROVIDER CLIENT
# ————————————————
# Global client - will be initialized in main() based on config
multi_provider_client = None

# ————————————————
# 5) ENHANCED SYSTEM PROMPT WITH RAG
# ————————————————
def create_enhanced_system_prompt(rag_system: ParaViewRAG, task_description: str) -> str:
    """Create enhanced system prompt matching notebook implementation."""
    
    # Load operations.json directly like the notebook
    operations_file = HERE / "operations.json"
    operations_json = []
    
    if operations_file.exists():
        try:
            with open(operations_file, "r") as f:
                operations_json = json.load(f)
        except Exception as e:
            print(f"Failed to load operations.json: {e}")
    
    # Create exact system prompt from notebook
    system_prompt = f'''You are a highly accurate code assistant specializing in 3D visualization scripting (e.g., ParaView, VTK). Your task is to read and execute the user's prompt line by line, ensuring that all operations, camera angles, views, rendering, and screenshots are handled correctly.

Execution Rules:
Process the Prompt Line-by-Line

Read and execute each instruction in order without skipping or merging steps.
If an operation depends on a previous step, ensure proper sequencing.
Camera and Viewing Directions

Object Creation and Rendering
Unless the user specifically instructs you to not show a data source, please show any data source after it has been loaded or created.

Apply background settings before rendering.
If a white background is needed for screenshots, ensure it is set before rendering.
Save screenshots immediately after rendering, before moving to the next step.
Ensure filenames or saving locations match the user's intent.

Camera and Viewing Directions
If a specific camera direction or position is given by the user adjust the camera accordingly.
If the user does not specify how to zoom the camera, zoom the camera to fit the active rendered objects as the last operation in the script. Also, zoom the camera to fit the active rendered objects immediately before saving a screenshot. Call ResetCamera() on the render view object so that the camera will be zoomed to fit.
If the user manually specifies a camera zoom level, follow their instructions and do not insert extra calls to 'renderView.ResetCamera();layout = CreateLayout(name='Layout')layout.AssignView(0, renderView)'.

Use provided operation templates as references.
Maintain correct syntax, function calls, and parameters.
Code Quality & Best Practices

Ensure modular, readable, and structured code.
Add comments to explain significant steps.
Avoid redundant operations and ensure compatibility with visualization libraries.
Primary Goal:
Generate a precise, structured, and error-free script that accurately follows the user's instructions, handling camera angles, views, rendering, and screenshots correctly. If any ambiguity exists, infer the most logical approach based on best practices. Follow Example Operations \\n{operations_json}'''
    
    return system_prompt

# ————————————————
# 6) PARAVIEW EXECUTION HELPERS
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
def extract_python_code(script_content: str, filename: str) -> str:
    """Extract Python code from script content and save to file - matches notebook implementation."""
    # Extract code between ```python and ```
    python_pattern = r'```python\s*(.*?)\s*```'
    matches = re.findall(python_pattern, script_content, re.DOTALL)
    
    if matches:
        # Take the first (or largest) code block
        code = matches[0] if len(matches) == 1 else max(matches, key=len)
        
        # Clean up the code
        code = code.strip()
        
        # Save to file
        file_path = f"{filename}.py"
        with open(file_path, 'w') as f:
            f.write(code)
        
        return file_path
    else:
        # If no code blocks found, treat entire content as code
        with open(f"{filename}.py", 'w') as f:
            f.write(script_content)
        return f"{filename}.py"

def extract_error_messages(stderr_output: str) -> str:
    """Extract error messages from stderr output - matches notebook implementation."""
    if not stderr_output:
        return ""
    
    # Filter out common non-error messages
    lines = stderr_output.split('\n')
    error_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Skip common ParaView info messages
        skip_patterns = [
            "Warning: In",
            "vtkOpenGL",
            "QStandardPaths:",
            "qt.qpa.fonts:",
            "Created ThermoData"
        ]
        
        should_skip = False
        for pattern in skip_patterns:
            if pattern in line:
                should_skip = True
                break
        
        if not should_skip and any(error_word in line for error_word in ["Error", "Traceback", "Exception", "ERROR"]):
            error_lines.append(line)
    
    return '\n'.join(error_lines) if error_lines else ""

def extract_python(text: str) -> str:
    m = re.search(r"```python(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()

def extract_errors(output: str) -> list:
    """Extract error messages from ParaView output, similar to notebook approach."""
    error_lines = []
    lines = output.splitlines()
    
    for i, line in enumerate(lines):
        # Look for various error patterns
        if any(pattern in line for pattern in ["Error", "Traceback", "Exception", "ERROR", "AttributeError", "NameError", "ValueError", "TypeError"]):
            error_lines.append(line.strip())
            
            # Also include following lines that are part of the traceback
            j = i + 1
            while j < len(lines) and (lines[j].startswith('  ') or lines[j].startswith('    ') or 'File "' in lines[j]):
                error_lines.append(lines[j].strip())
                j += 1
                if j >= i + 10:  # Limit to avoid too much output
                    break
    
    return error_lines

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
def run_case(case_path: Path, client) -> Dict:
    case = case_path.name
    print(f"\n=== {case} ===")

    # Initialize token counter and RAG system
    token_counter = TokenCounter()
    rag_system = ParaViewRAG()
    
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

    # 2) generate script with enhanced system prompt
    enhanced_system_prompt = create_enhanced_system_prompt(rag_system, prompt)
    
    sr = client.create_completion(
        messages=[
            {"role":"system","content": enhanced_system_prompt},
            {"role":"user","content":prompt}
        ]
    )
    
    # Count output tokens and update token usage
    response_content = client.get_response_content(sr)
    token_usage = client.get_token_usage(sr)
    result["token_usage"]["output_tokens"] += token_usage["output_tokens"]
    result["token_usage"]["total_tokens"] = result["token_usage"]["input_tokens"] + result["token_usage"]["output_tokens"]
    
    # Calculate cost
    cost_info = calculate_cost(result["token_usage"]["input_tokens"], result["token_usage"]["output_tokens"], client.get_pricing_info())
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

    # 5) execution + iterative fix with enhanced feedback
    max_attempts = 5
    out = execute_paraview_code(code)
    errors = extract_errors(out)
    attempt = 1
    
    # Build conversation history for better context
    conversation_history = [
        {"role": "system", "content": enhanced_system_prompt},
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": f"```python\n{code}\n```"}
    ]
    
    while errors and attempt < max_attempts:
        attempt += 1
        print(f"  → fix attempt {attempt} errors: {errors}")
        
        # Create detailed error feedback
        error_feedback = f"""
I tried running the following Python script and encountered errors:

**Error Messages:**
{chr(10).join(errors)}

**Original Script:**
```python
{code}
```

Can you help me fix the issues and provide a corrected version of the script? 
Please make sure the new script runs correctly without errors.
"""
        
        # Add error feedback to conversation
        conversation_history.append({"role": "user", "content": error_feedback})
        
        fix = client.create_completion(messages=conversation_history)
        
        # Count tokens for fix attempts
        fix_response = client.get_response_content(fix)
        fix_input_tokens = token_counter.count_tokens(error_feedback)
        fix_output_tokens = token_counter.count_tokens(fix_response)
        
        # Update token usage
        result["token_usage"]["input_tokens"] += fix_input_tokens
        result["token_usage"]["output_tokens"] += fix_output_tokens
        result["token_usage"]["total_tokens"] = result["token_usage"]["input_tokens"] + result["token_usage"]["output_tokens"]
        
        # Update cost
        cost_info = calculate_cost(result["token_usage"]["input_tokens"], result["token_usage"]["output_tokens"], client.get_pricing_info())
        if cost_info:
            result["cost_info"] = cost_info
        
        # Append to response and conversation history
        result["response"] += f"\n\nFix attempt {attempt}:\n{fix_response}"
        conversation_history.append({"role": "assistant", "content": fix_response})
        
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
def main(config_path: str = None):
    """Main function with optional config path."""
    # Import MultiProviderClient
    from multi_provider_client import MultiProviderClient
    
    # Initialize client based on config
    if config_path and os.path.exists(config_path):
        print(f"Using configuration from: {config_path}")
        client = MultiProviderClient.from_config_file(config_path)
    else:
        # Fallback to OpenAI with environment variable
        print("Using default OpenAI configuration")
        client = MultiProviderClient(
            provider="openai",
            model="gpt-4o",
            api_key=os.getenv("OPENAI_API_KEY")
        )
    
    results = {}
    all_results = []
    start_time = datetime.now()
    
    for case_dir in sorted(BATCH_DIR.iterdir()):
        if case_dir.is_dir():
            result = run_case(case_dir, client)
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
    import argparse
    
    parser = argparse.ArgumentParser(description="Run ChatVis test cases")
    parser.add_argument("--config", "-c", 
                       help="Path to the configuration JSON file")
    
    args = parser.parse_args()
    main(args.config)