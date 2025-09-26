#!/usr/bin/env python3
"""
MCP JSON Logger for SciVisAgentBench

Enhanced version of the MCP logger that saves structured JSON logs with timestamps
and message ordering for benchmark evaluation.

This logger sits between the MCP server and LLM agent, capturing all communication
in a structured format suitable for analysis.
"""

import sys
import subprocess
import threading
import argparse
import os
import json
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# --- Configuration ---
BENCHMARK_DIR = Path(__file__).parent.parent
BASE_LOGS_DIR = BENCHMARK_DIR / "communication_logs"
BASE_LOGS_DIR.mkdir(exist_ok=True)

# Check for test case specific session naming
test_case_name = os.environ.get("SCIVISAGENT_TEST_CASE", "")
if test_case_name:
    # Generate timestamped session directory with test case name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
    SESSION_DIR = BASE_LOGS_DIR / f"session_{test_case_name}_{timestamp}"
else:
    # Generate timestamped session directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
    SESSION_DIR = BASE_LOGS_DIR / f"session_{timestamp}"

SESSION_DIR.mkdir(exist_ok=True)

# Create subdirectories for this session
SCREENSHOTS_DIR = SESSION_DIR / "screenshots"
SCREENSHOTS_DIR.mkdir(exist_ok=True)

# Generate log filename in the session directory
LOG_FILE = SESSION_DIR / f"mcp_communication.json"
# --- End Configuration ---

class MCPCommunicationLogger:
    """
    Structured logger for MCP communications with automatic screenshot capture
    """
    
    def __init__(self, log_file: Path, screenshots_dir: Path, session_dir: Path):
        self.log_file = log_file
        self.screenshots_dir = screenshots_dir
        self.session_dir = session_dir
        test_case_name = os.environ.get("SCIVISAGENT_TEST_CASE", "")
        self.session_data = {
            "session_id": f"mcp_session_{timestamp}",
            "test_case_name": test_case_name,
            "session_directory": str(session_dir),
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "total_messages": 0,
            "agent_to_server_count": 0,
            "server_to_agent_count": 0,
            "stderr_count": 0,
            "screenshot_count": 0,
            "screenshots": [],
            "tool_calls_detected": 0,
            "communications": [],
            "metadata": {
                "logger_version": "1.0.0",
                "format": "json",
                "encoding": "utf-8",
                "test_case_name": test_case_name,
                "session_directory": str(session_dir),
                "screenshots_directory": str(screenshots_dir)
            }
        }
        self.message_counter = 0
        self.lock = threading.Lock()
        self.target_process = None
        self.pending_tool_calls = {}  # Track pending tool calls by ID
        
        # Initialize the log file
        self._save_session()
    
    def log_communication(self, 
                         direction: str, 
                         content: str, 
                         message_type: str = "data",
                         raw_bytes: int = 0) -> None:
        """
        Log a communication event
        
        Args:
            direction: "agent_to_server", "server_to_agent", or "stderr"
            content: The message content (decoded string)
            message_type: Type of message ("data", "error", "eof", etc.)
            raw_bytes: Number of raw bytes in the original message
        """
        with self.lock:
            # Check if this is a screenshot response
            if self._is_screenshot_response(content, direction):
                # Extract screenshot ID to determine if it's agent-requested
                try:
                    parsed = json.loads(content)
                    screenshot_id = str(parsed["id"])
                    is_agent_requested = self._is_agent_requested_screenshot(screenshot_id)
                    
                    if is_agent_requested:
                        # For agent-requested screenshots, log the communication AND save the image
                        content = self._handle_agent_screenshot_response(content, screenshot_id)
                        # Continue with normal logging below
                    else:
                        # For auto-screenshots, just save the image without logging
                        self._handle_screenshot_response(content)
                        return  # Don't log auto-screenshot responses
                        
                except json.JSONDecodeError:
                    # If we can't parse it, treat as regular communication
                    pass
            
            self.message_counter += 1
            
            communication_entry = {
                "sequence_number": self.message_counter,
                "timestamp": datetime.now().isoformat(),
                "direction": direction,
                "message_type": message_type,
                "content": content,
                "content_length": len(content),
                "raw_bytes": raw_bytes,
                "is_json": self._is_json_content(content),
                "is_mcp_message": self._is_mcp_message(content)
            }
            
            # Add special handling for agent screenshot responses
            if (self._is_json_content(content) and 
                direction == "server_to_agent" and 
                "<BASE64_IMAGE_DATA_" in content):
                communication_entry["is_agent_screenshot_response"] = True
                communication_entry["screenshot_saved"] = True
            
            # Add JSON parsing if it's JSON content
            if communication_entry["is_json"]:
                try:
                    parsed_json = json.loads(content)
                    communication_entry["parsed_json"] = parsed_json
                    
                    # Check for tool calls and responses
                    self._handle_mcp_message(parsed_json, direction)
                    
                except json.JSONDecodeError:
                    communication_entry["json_parse_error"] = True
            
            self.session_data["communications"].append(communication_entry)
            self.session_data["total_messages"] = self.message_counter
            
            # Update counters
            if direction == "agent_to_server":
                self.session_data["agent_to_server_count"] += 1
            elif direction == "server_to_agent":
                self.session_data["server_to_agent_count"] += 1
            elif direction == "stderr":
                self.session_data["stderr_count"] += 1
            
            # Save periodically (every 10 messages) and on important messages
            if (self.message_counter % 10 == 0 or 
                communication_entry["is_mcp_message"] or 
                message_type == "error"):
                self._save_session()
    
    def _handle_mcp_message(self, parsed_json: Dict[str, Any], direction: str) -> None:
        """Handle MCP protocol messages and trigger screenshots when appropriate"""
        if not isinstance(parsed_json, dict):
            return
            
        # Handle tool call requests (from agent to server)
        if (direction == "agent_to_server" and 
            parsed_json.get("method") == "tools/call" and 
            "id" in parsed_json):
            
            tool_call_id = parsed_json["id"]
            tool_name = parsed_json.get("params", {}).get("name", "unknown")
            
            self.pending_tool_calls[tool_call_id] = {
                "tool_name": tool_name,
                "timestamp": datetime.now().isoformat(),
                "params": parsed_json.get("params", {}),
                "is_agent_screenshot": tool_name == "get_screenshot"  # Mark agent-requested screenshots
            }
            
            self.session_data["tool_calls_detected"] += 1
            
        # Handle tool call responses (from server to agent)
        elif (direction == "server_to_agent" and 
              "id" in parsed_json and 
              parsed_json["id"] in self.pending_tool_calls):
            
            tool_call_id = parsed_json["id"]
            tool_info = self.pending_tool_calls.pop(tool_call_id)
            tool_name = tool_info["tool_name"]
            is_agent_screenshot = tool_info.get("is_agent_screenshot", False)
            
            # For agent-requested screenshots, we still process them normally but mark them
            if is_agent_screenshot:
                # This will be handled by _is_screenshot_response but marked as agent-requested
                pass
            
            # Check if this is a successful response and not a screenshot call or benchmark call
            elif ("result" in parsed_json and 
                  tool_name != "get_screenshot" and
                  not tool_name.startswith("benchmark_")):
                
                # Trigger auto-screenshot capture after a short delay
                threading.Timer(0.5, self._capture_screenshot, 
                              args=(tool_call_id, tool_name)).start()

    def _is_screenshot_response(self, content: str, direction: str) -> bool:
        """Check if content is a screenshot response that should be saved as PNG"""
        if direction != "server_to_agent" or not self._is_json_content(content):
            return False
        
        try:
            parsed = json.loads(content)
            if (isinstance(parsed, dict) and 
                "result" in parsed and 
                "id" in parsed):
                
                # Check if the result contains image data (this is the key indicator)
                result = parsed.get("result", {})
                if isinstance(result, dict):
                    content_array = result.get("content", [])
                    if isinstance(content_array, list) and len(content_array) > 0:
                        first_content = content_array[0]
                        if (isinstance(first_content, dict) and 
                            first_content.get("type") == "image" and 
                            "data" in first_content):
                            # Additional check: ensure data looks like base64 image data
                            data = first_content.get("data", "")
                            if isinstance(data, str) and len(data) > 100:  # Base64 images are quite long
                                # Check for common image format signatures in base64
                                if (data.startswith("iVBORw0KGgo") or  # PNG
                                    data.startswith("/9j/") or          # JPEG
                                    data.startswith("R0lGOD") or        # GIF
                                    data.startswith("data:image")):     # Data URL
                                    return True
        except json.JSONDecodeError:
            pass
        
        return False

    def _is_agent_requested_screenshot(self, screenshot_id: str) -> bool:
        """Check if a screenshot response is from an agent request (not auto-triggered)"""
        # Check if this ID corresponds to an agent-requested screenshot
        # Agent screenshots have numeric IDs, auto-screenshots have prefixed IDs
        return not str(screenshot_id).startswith("screenshot_")

    def _handle_agent_screenshot_response(self, content: str, screenshot_id: str) -> str:
        """Handle agent-requested screenshot response by saving image and updating communication entry"""
        try:
            parsed = json.loads(content)
            
            # Extract image data and save as PNG
            result = parsed.get("result", {})
            content_array = result.get("content", [])
            if content_array and len(content_array) > 0:
                image_data = content_array[0].get("data", "")
                
                if image_data:
                    # Save as PNG file
                    png_filename = self._save_screenshot_as_png(screenshot_id, image_data)
                    
                    # Record this as an agent-requested screenshot
                    screenshot_entry = {
                        "screenshot_request_id": screenshot_id,
                        "actual_response_id": screenshot_id,
                        "timestamp": datetime.now().isoformat(),
                        "request_sent": True,
                        "response_received": True,
                        "response_timestamp": datetime.now().isoformat(),
                        "saved_as_png": True,
                        "type": "agent_screenshot",
                        "png_file": png_filename,
                        "triggered_by": "agent_request"
                    }
                    self.session_data["screenshots"].append(screenshot_entry)
                    self.session_data["screenshot_count"] += 1
                    
                    # Create a modified content for logging (without the large base64 data)
                    modified_content = content.replace(image_data, f"<BASE64_IMAGE_DATA_{len(image_data)}_BYTES_SAVED_AS_{png_filename}>")
                    
                    # Update the content to be logged
                    return modified_content
                    
        except Exception as e:
            self.log_error(f"Failed to handle agent screenshot response: {e}", f"screenshot_id={screenshot_id}")
        
        # If something went wrong, return original content
        return content

    def _handle_screenshot_response(self, content: str) -> None:
        """Handle auto-screenshot response by saving it as PNG file (no logging)"""
        try:
            parsed = json.loads(content)
            screenshot_id = str(parsed["id"])
            
            # Extract image data
            result = parsed.get("result", {})
            content_array = result.get("content", [])
            if content_array and len(content_array) > 0:
                image_data = content_array[0].get("data", "")
                
                if image_data:
                    # Save as PNG file
                    self._save_screenshot_as_png(screenshot_id, image_data)
                    
                    # Update the corresponding screenshot entry in session data
                    # Try to find by screenshot_request_id first, then by any pending screenshot
                    screenshot_updated = False
                    for screenshot_entry in self.session_data["screenshots"]:
                        if (screenshot_entry.get("screenshot_request_id") == screenshot_id or
                            (not screenshot_entry.get("response_received", False) and 
                             screenshot_entry.get("request_sent", False))):
                            screenshot_entry["response_received"] = True
                            screenshot_entry["response_timestamp"] = datetime.now().isoformat()
                            screenshot_entry["saved_as_png"] = True
                            screenshot_entry["actual_response_id"] = screenshot_id
                            screenshot_updated = True
                            break
                    
                    # If no matching entry found, create a new one
                    if not screenshot_updated:
                        screenshot_entry = {
                            "triggered_by_tool_call_id": "unknown",
                            "triggered_by_tool_name": "unknown", 
                            "screenshot_request_id": screenshot_id,
                            "actual_response_id": screenshot_id,
                            "timestamp": datetime.now().isoformat(),
                            "request_sent": True,
                            "response_received": True,
                            "response_timestamp": datetime.now().isoformat(),
                            "saved_as_png": True,
                            "type": "auto_screenshot"
                        }
                        self.session_data["screenshots"].append(screenshot_entry)
                        self.session_data["screenshot_count"] += 1
                    
                    self._save_session()
                    
        except Exception as e:
            self.log_error(f"Failed to handle auto screenshot response: {e}", f"content_length={len(content)}")

    def _save_screenshot_as_png(self, screenshot_id: str, base64_data: str) -> str:
        """Save base64 image data as PNG file and return the filename"""
        try:
            import base64
            
            # Remove data URL prefix if present
            if base64_data.startswith("data:image"):
                base64_data = base64_data.split(",", 1)[1]
            
            # Decode base64 data
            image_bytes = base64.b64decode(base64_data)
            
            # Generate filename
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # microseconds to milliseconds
            # Clean up screenshot_id for filename (remove special characters)
            clean_id = "".join(c for c in str(screenshot_id) if c.isalnum() or c in "_-")
            filename = f"screenshot_{timestamp_str}_{clean_id}.png"
            filepath = self.screenshots_dir / filename
            
            # Save to file
            with open(filepath, 'wb') as f:
                f.write(image_bytes)
            
            print(f"Screenshot saved: {filepath}", file=sys.stderr)
            
            # Update session data
            for screenshot_entry in self.session_data["screenshots"]:
                if (screenshot_entry.get("screenshot_request_id") == screenshot_id or
                    screenshot_entry.get("actual_response_id") == screenshot_id):
                    screenshot_entry["png_file"] = filename  # Just the filename, not full path
                    screenshot_entry["file_size_bytes"] = len(image_bytes)
                    break
            
            return filename
                    
        except Exception as e:
            self.log_error(f"Failed to save screenshot as PNG: {e}", f"screenshot_id={screenshot_id}")
            return "error_saving_file"

    def _capture_screenshot(self, tool_call_id: str, tool_name: str) -> None:
        """Capture a screenshot by sending a get_screenshot request to the server"""
        if not self.target_process or self.target_process.poll() is not None:
            return
            
        try:
            # Create a screenshot request with a prefixed ID to mark it as auto-screenshot
            screenshot_request = {
                "jsonrpc": "2.0",
                "id": f"screenshot_{tool_call_id}_{int(time.time() * 1000)}",
                "method": "tools/call",
                "params": {
                    "name": "get_screenshot"
                }
            }
            
            request_json = json.dumps(screenshot_request) + "\n"
            
            # Send the request to the server
            self.target_process.stdin.write(request_json.encode('utf-8'))
            self.target_process.stdin.flush()
            
            # Record this as an auto-screenshot request (but don't log it in normal communications)
            screenshot_entry = {
                "triggered_by_tool_call_id": tool_call_id,
                "triggered_by_tool_name": tool_name,
                "screenshot_request_id": screenshot_request["id"],
                "timestamp": datetime.now().isoformat(),
                "request_sent": True,
                "type": "auto_screenshot"
            }
            
            self.session_data["screenshots"].append(screenshot_entry)
            self.session_data["screenshot_count"] += 1
            
            # Save session after screenshot request
            self._save_session()
            
        except Exception as e:
            self.log_error(f"Failed to capture auto screenshot: {e}", 
                          f"tool_call={tool_name}, id={tool_call_id}")

    def set_target_process(self, process) -> None:
        """Set the target process for screenshot requests"""
        self.target_process = process

    def _is_json_content(self, content: str) -> bool:
        """Check if content appears to be JSON"""
        content = content.strip()
        return (content.startswith('{') and content.endswith('}')) or \
               (content.startswith('[') and content.endswith(']'))
    
    def _is_mcp_message(self, content: str) -> bool:
        """Check if content appears to be an MCP protocol message"""
        if not self._is_json_content(content):
            return False
        
        try:
            data = json.loads(content.strip())
            # Check for MCP protocol fields
            if isinstance(data, dict):
                mcp_fields = ['jsonrpc', 'method', 'params', 'id', 'result', 'error']
                return any(field in data for field in mcp_fields)
        except json.JSONDecodeError:
            pass
        
        return False
    
    def log_error(self, error_msg: str, context: str = "") -> None:
        """Log an error event"""
        error_content = f"ERROR: {error_msg}"
        if context:
            error_content += f" (Context: {context})"
        
        self.log_communication("stderr", error_content, "error")
    
    def log_eof(self, stream: str) -> None:
        """Log an EOF event"""
        self.log_communication("stderr", f"EOF reached on {stream}", "eof")
    
    def finalize_session(self, exit_code: int = 0) -> None:
        """Finalize the logging session"""
        with self.lock:
            self.session_data["end_time"] = datetime.now().isoformat()
            self.session_data["exit_code"] = exit_code
            
            # Calculate session duration
            start_time = datetime.fromisoformat(self.session_data["start_time"])
            end_time = datetime.fromisoformat(self.session_data["end_time"])
            duration_seconds = (end_time - start_time).total_seconds()
            self.session_data["duration_seconds"] = duration_seconds
            
            # Add final statistics
            auto_screenshots = sum(1 for s in self.session_data["screenshots"] if s.get("type") == "auto_screenshot")
            agent_screenshots = sum(1 for s in self.session_data["screenshots"] if s.get("type") == "agent_screenshot")
            
            self.session_data["statistics"] = {
                "total_communications": len(self.session_data["communications"]),
                "json_messages": sum(1 for comm in self.session_data["communications"] if comm["is_json"]),
                "mcp_messages": sum(1 for comm in self.session_data["communications"] if comm["is_mcp_message"]),
                "error_messages": sum(1 for comm in self.session_data["communications"] if comm["message_type"] == "error"),
                "average_message_length": sum(comm["content_length"] for comm in self.session_data["communications"]) / len(self.session_data["communications"]) if self.session_data["communications"] else 0,
                "tool_calls_detected": self.session_data["tool_calls_detected"],
                "screenshots_captured": self.session_data["screenshot_count"],
                "auto_screenshots": auto_screenshots,
                "agent_screenshots": agent_screenshots,
                "agent_screenshot_responses_logged": sum(1 for comm in self.session_data["communications"] if comm.get("is_agent_screenshot_response", False))
            }
            
            self._save_session()
            
            print(f"MCP JSON Logger: Session saved to {self.session_dir}", file=sys.stderr)
            print(f"  - Log file: {self.log_file}", file=sys.stderr)
            print(f"  - Screenshots: {self.screenshots_dir}", file=sys.stderr)
            print(f"Total messages logged: {self.session_data['total_messages']}", file=sys.stderr)
            print(f"Tool calls detected: {self.session_data['tool_calls_detected']}", file=sys.stderr)
            print(f"Screenshots captured: {self.session_data['screenshot_count']}", file=sys.stderr)
            print(f"  - Auto screenshots (after tool calls): {self.session_data['statistics']['auto_screenshots']}", file=sys.stderr)
            print(f"  - Agent screenshots (explicitly requested): {self.session_data['statistics']['agent_screenshots']}", file=sys.stderr)
    
    def _save_session(self) -> None:
        """Save the current session data to file"""
        try:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(self.session_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving log file: {e}", file=sys.stderr)

# Global logger instance
logger = MCPCommunicationLogger(LOG_FILE, SCREENSHOTS_DIR, SESSION_DIR)

# --- Argument Parsing ---
parser = argparse.ArgumentParser(
    description="Wrap a command, passing STDIN/STDOUT verbatim while logging them as structured JSON.",
    usage="%(prog)s <command> [args...]"
)
parser.add_argument('command', nargs=argparse.REMAINDER,
                    help='The command and its arguments to execute.')

if len(sys.argv) == 1:
    parser.print_help(sys.stderr)
    sys.exit(1)

args = parser.parse_args()

if not args.command:
    print("Error: No command provided.", file=sys.stderr)
    parser.print_help(sys.stderr)
    sys.exit(1)

target_command = args.command
# --- End Argument Parsing ---

# --- Enhanced I/O Forwarding Functions ---

def forward_and_log_stdin(proxy_stdin, target_stdin, comm_logger):
    """Reads from proxy's stdin, logs it, writes to target's stdin."""
    try:
        while True:
            # Read line by line from the script's actual stdin
            line_bytes = proxy_stdin.readline()
            if not line_bytes:  # EOF reached
                comm_logger.log_eof("stdin")
                break

            # Decode for logging
            try:
                line_str = line_bytes.decode('utf-8')
            except UnicodeDecodeError:
                line_str = f"[Non-UTF8 data, {len(line_bytes)} bytes]"

            # Log the communication
            comm_logger.log_communication(
                direction="agent_to_server",
                content=line_str.rstrip('\n\r'),  # Remove trailing newlines for cleaner logs
                message_type="data",
                raw_bytes=len(line_bytes)
            )

            # Write the original bytes to the target process's stdin
            target_stdin.write(line_bytes)
            target_stdin.flush()

    except Exception as e:
        comm_logger.log_error(f"STDIN Forwarding Error: {e}", "stdin_forwarding")

    finally:
        # Close the target's stdin when proxy's stdin closes
        try:
            target_stdin.close()
            comm_logger.log_communication("stderr", "STDIN stream closed to target", "eof")
        except Exception as e:
            comm_logger.log_error(f"Error closing target STDIN: {e}", "stdin_cleanup")


def forward_and_log_stdout(target_stdout, proxy_stdout, comm_logger):
    """Reads from target's stdout, logs it, writes to proxy's stdout."""
    try:
        while True:
            # Read line by line from the target process's stdout
            line_bytes = target_stdout.readline()
            if not line_bytes:  # EOF reached
                comm_logger.log_eof("stdout")
                break

            # Decode for logging
            try:
                line_str = line_bytes.decode('utf-8')
            except UnicodeDecodeError:
                line_str = f"[Non-UTF8 data, {len(line_bytes)} bytes]"

            # Log the communication
            comm_logger.log_communication(
                direction="server_to_agent",
                content=line_str.rstrip('\n\r'),
                message_type="data",
                raw_bytes=len(line_bytes)
            )

            # Write the original bytes to the script's actual stdout
            proxy_stdout.write(line_bytes)
            proxy_stdout.flush()

    except Exception as e:
        comm_logger.log_error(f"STDOUT Forwarding Error: {e}", "stdout_forwarding")

    finally:
        try:
            proxy_stdout.flush()
        except Exception as e:
            comm_logger.log_error(f"Error flushing proxy stdout: {e}", "stdout_cleanup")


def forward_and_log_stderr(target_stderr, proxy_stderr, comm_logger):
    """Reads from target's stderr, logs it, writes to proxy's stderr."""
    try:
        while True:
            line_bytes = target_stderr.readline()
            if not line_bytes:
                comm_logger.log_eof("stderr")
                break
            
            try:
                line_str = line_bytes.decode('utf-8')
            except UnicodeDecodeError:
                line_str = f"[Non-UTF8 data, {len(line_bytes)} bytes]"
            
            # Log stderr communications
            comm_logger.log_communication(
                direction="stderr",
                content=line_str.rstrip('\n\r'),
                message_type="stderr",
                raw_bytes=len(line_bytes)
            )
            
            proxy_stderr.write(line_bytes)
            proxy_stderr.flush()
            
    except Exception as e:
        comm_logger.log_error(f"STDERR Forwarding Error: {e}", "stderr_forwarding")

    finally:
        try:
            proxy_stderr.flush()
        except Exception as e:
            comm_logger.log_error(f"Error flushing proxy stderr: {e}", "stderr_cleanup")


# --- Main Execution ---
process = None
exit_code = 1  # Default exit code in case of early failure

try:
    print(f"MCP JSON Logger: Starting session, logging to {LOG_FILE}", file=sys.stderr)
    
    # Start the target process
    process = subprocess.Popen(
        target_command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0  # Unbuffered binary I/O
    )
    
    # Set the target process in the logger for screenshot requests
    logger.set_target_process(process)

    # Create and start forwarding threads
    stdin_thread = threading.Thread(
        target=forward_and_log_stdin,
        args=(sys.stdin.buffer, process.stdin, logger),
        daemon=True
    )

    stdout_thread = threading.Thread(
        target=forward_and_log_stdout,
        args=(process.stdout, sys.stdout.buffer, logger),
        daemon=True
    )

    stderr_thread = threading.Thread(
        target=forward_and_log_stderr,
        args=(process.stderr, sys.stderr.buffer, logger),
        daemon=True
    )

    # Start all threads
    stdin_thread.start()
    stdout_thread.start()
    stderr_thread.start()

    # Wait for the target process to complete
    process.wait()
    exit_code = process.returncode

    # Wait for I/O threads to finish
    stdin_thread.join(timeout=2.0)
    stdout_thread.join(timeout=2.0)
    stderr_thread.join(timeout=2.0)

except Exception as e:
    logger.log_error(f"MCP JSON Logger Main Error: {e}", "main_execution")
    print(f"MCP JSON Logger Error: {e}", file=sys.stderr)
    exit_code = 1

finally:
    # Ensure the process is terminated
    if process and process.poll() is None:
        try:
            process.terminate()
            process.wait(timeout=1.0)
        except:
            pass
        if process.poll() is None:
            try:
                process.kill()
            except:
                pass

    # Finalize the logging session
    logger.finalize_session(exit_code)

    # Exit with the target process's exit code
    sys.exit(exit_code)
