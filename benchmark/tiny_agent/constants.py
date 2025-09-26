from __future__ import annotations

import sys
from pathlib import Path
from typing import List

# Import from the specific location where ChatCompletionInputTool is defined
try:
    from huggingface_hub.inference._generated.types import ChatCompletionInputTool
except ImportError:
    from huggingface_hub import ChatCompletionInputTool


FILENAME_CONFIG = "agent.json"
FILENAME_PROMPT = "PROMPT.md"

DEFAULT_AGENT = {
    "model": "Qwen/Qwen2.5-72B-Instruct",
    "provider": "nebius",
    "servers": [
        {
            "type": "stdio",
            "config": {
                "command": "npx",
                "args": [
                    "-y",
                    "@modelcontextprotocol/server-filesystem",
                    str(Path.home() / ("Desktop" if sys.platform == "darwin" else "")),
                ],
            },
        },
        {
            "type": "stdio",
            "config": {
                "command": "npx",
                "args": ["@playwright/mcp@latest"],
            },
        },
    ],
}


DEFAULT_SYSTEM_PROMPT = """
You are an agent - please keep going until the user's query is completely
resolved, before ending your turn and yielding back to the user. Only terminate
your turn when you are sure that the problem is solved, or if you need more
info from the user to solve the problem.
If you are not sure about anything pertaining to the user's request, use your
tools to read files and gather the relevant information: do NOT guess or make
up an answer.
You MUST plan extensively before each function call, and reflect extensively
on the outcomes of the previous function calls. DO NOT do this entire process
by making function calls only, as this can impair your ability to solve the
problem and think insightfully.

IMPORTANT: When you need clarification, additional information, or encounter 
ambiguity in the user's request, use the ask_question tool to ask the user 
for specific details. Do not make assumptions or guess - always ask for 
clarification when needed. And when you ask a question, make sure to use 
the ask_question tool, otherwise the user will not be able to answer you.
But please give responses based on user's answer before you complete the task.
""".strip()

MAX_NUM_TURNS = 20

TASK_COMPLETE_TOOL: ChatCompletionInputTool = ChatCompletionInputTool.parse_obj(  # type: ignore[assignment]
    {
        "type": "function",
        "function": {
            "name": "task_complete",
            "description": "Call this tool when the task given by the user is complete",
            "parameters": {"type": "object", "properties": {}},
        },
    }
)

# Improved ASK_QUESTION_TOOL with proper parameters
ASK_QUESTION_TOOL: ChatCompletionInputTool = ChatCompletionInputTool.parse_obj(  # type: ignore[assignment]
    {
        "type": "function",
        "function": {
            "name": "ask_question",
            "description": "Ask the user for clarification or additional information when needed. Use this tool whenever you encounter ambiguity, need specific details, or require user input to properly complete the task.",
            "parameters": {
                "type": "object", 
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The specific question to ask the user for clarification or additional information"
                    }
                },
                "required": ["question"]
            },
        },
    }
)

# Only TASK_COMPLETE_TOOL exits the loop - ASK_QUESTION_TOOL should continue the conversation
EXIT_LOOP_TOOLS: List[ChatCompletionInputTool] = [TASK_COMPLETE_TOOL]

# Keep ASK_QUESTION_TOOL available but separate for interactive handling
INTERACTIVE_TOOLS: List[ChatCompletionInputTool] = [ASK_QUESTION_TOOL]

DEFAULT_REPO_ID = "tiny-agents/tiny-agents"
