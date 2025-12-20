"""
Examples of Running Existing Agents

This script shows how to run the pre-built agents (ParaView MCP, Napari MCP, ChatVis)
using the evaluation framework.
"""

import asyncio
import os
from pathlib import Path

from benchmark.evaluation_framework import UnifiedTestRunner
from benchmark.evaluation_framework.agents import (
    ParaViewMCPAgent,
    NapariMCPAgent,
    ChatVisAgent
)


async def run_paraview_mcp_example():
    """Example: Run ParaView MCP agent on main benchmark."""

    print("\n" + "="*60)
    print("Running ParaView MCP Agent")
    print("="*60 + "\n")

    # Load configuration
    config_path = "benchmark/configs/paraview_mcp/config_openai.json"

    # Create agent
    agent = ParaViewMCPAgent.from_config_file(config_path)

    # Create runner
    runner = UnifiedTestRunner(
        agent=agent,
        yaml_path="SciVisAgentBench-tasks/main/main_cases.yaml",
        cases_dir="SciVisAgentBench-tasks/main",
        eval_model="gpt-4o",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # Load and run
    runner.load_yaml_test_cases()
    summary = await runner.run_all_test_cases(run_evaluation=True)

    print(f"\n✅ ParaView MCP Results:")
    print(f"   Success rate: {summary['success_rate']:.1%}")
    print(f"   Total tokens: {summary['total_tokens']:,}")


async def run_napari_mcp_example():
    """Example: Run Napari MCP agent on bioimage data."""

    print("\n" + "="*60)
    print("Running Napari MCP Agent")
    print("="*60 + "\n")

    # Load configuration
    config_path = "benchmark/configs/napari_mcp/config_openai.json"

    # Create agent
    agent = NapariMCPAgent.from_config_file(config_path)

    # Create runner for bioimage data
    runner = UnifiedTestRunner(
        agent=agent,
        yaml_path="SciVisAgentBench-tasks/bioimage_data/eval_cases.yaml",
        cases_dir="SciVisAgentBench-tasks/bioimage_data",
        data_dir="SciVisAgentBench-tasks/bioimage_data/data",
        eval_model="gpt-4o",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # Load and run
    runner.load_yaml_test_cases()
    summary = await runner.run_all_test_cases(run_evaluation=True)

    print(f"\n✅ Napari MCP Results:")
    print(f"   Success rate: {summary['success_rate']:.1%}")
    print(f"   Total tokens: {summary['total_tokens']:,}")


async def run_chatvis_example():
    """Example: Run ChatVis agent on main benchmark."""

    print("\n" + "="*60)
    print("Running ChatVis Agent")
    print("="*60 + "\n")

    # Load configuration
    config_path = "benchmark/configs/chatvis/config_openai.json"

    # Create agent
    agent = ChatVisAgent.from_config_file(config_path)

    # Create runner
    runner = UnifiedTestRunner(
        agent=agent,
        yaml_path="SciVisAgentBench-tasks/main/main_cases.yaml",
        cases_dir="SciVisAgentBench-tasks/main",
        eval_model="gpt-4o",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        static_screenshot=True  # ChatVis uses static screenshots
    )

    # Load and run
    runner.load_yaml_test_cases()
    summary = await runner.run_all_test_cases(run_evaluation=True)

    print(f"\n✅ ChatVis Results:")
    print(f"   Success rate: {summary['success_rate']:.1%}")
    print(f"   Total tokens: {summary['total_tokens']:,}")


async def run_single_case_example():
    """Example: Run a single test case."""

    print("\n" + "="*60)
    print("Running Single Test Case")
    print("="*60 + "\n")

    # Create agent
    config_path = "benchmark/configs/paraview_mcp/config_openai.json"
    agent = ParaViewMCPAgent.from_config_file(config_path)

    # Create runner
    runner = UnifiedTestRunner(
        agent=agent,
        yaml_path="SciVisAgentBench-tasks/main/main_cases.yaml",
        cases_dir="SciVisAgentBench-tasks/main",
        eval_model="gpt-4o"
    )

    # Load test cases
    test_cases = runner.load_yaml_test_cases()

    # Find a specific case (e.g., "aneurism")
    target_case = None
    for case in test_cases:
        if case.case_name == "aneurism":
            target_case = case
            break

    if target_case:
        # Setup agent
        await agent.setup()

        try:
            # Run single case
            result = await runner.run_single_test_case(target_case)

            # Run evaluation
            if result.get("status") == "completed":
                eval_result = await runner.run_evaluation(target_case)
                print(f"\n✅ Single Case Results:")
                print(f"   Status: {result['status']}")
                print(f"   Duration: {result['duration_seconds']:.2f}s")
                if eval_result.get("scores"):
                    print(f"   Score: {eval_result['scores']['total_score']}/{eval_result['scores']['max_possible_score']}")

        finally:
            await agent.teardown()


async def compare_agents_example():
    """Example: Compare multiple agents on the same benchmark."""

    print("\n" + "="*60)
    print("Comparing Multiple Agents")
    print("="*60 + "\n")

    agents = [
        ("paraview_mcp", ParaViewMCPAgent, "benchmark/configs/paraview_mcp/config_openai.json"),
        ("chatvis", ChatVisAgent, "benchmark/configs/chatvis/config_openai.json"),
    ]

    results = {}

    for agent_name, agent_class, config_path in agents:
        print(f"\nRunning {agent_name}...")

        # Create agent
        agent = agent_class.from_config_file(config_path)

        # Create runner
        runner = UnifiedTestRunner(
            agent=agent,
            yaml_path="SciVisAgentBench-tasks/main/main_cases.yaml",
            cases_dir="SciVisAgentBench-tasks/main",
            eval_model="gpt-4o",
            static_screenshot=(agent_name == "chatvis")
        )

        # Run evaluation
        runner.load_yaml_test_cases()
        summary = await runner.run_all_test_cases(run_evaluation=True)

        results[agent_name] = summary

    # Print comparison
    print("\n" + "="*60)
    print("Agent Comparison")
    print("="*60)

    for agent_name, summary in results.items():
        print(f"\n{agent_name}:")
        print(f"  Success rate: {summary['success_rate']:.1%}")
        print(f"  Total tokens: {summary['total_tokens']:,}")
        print(f"  Duration: {summary['duration_seconds']:.1f}s")


if __name__ == "__main__":
    # Run examples (uncomment the one you want to try)

    # asyncio.run(run_paraview_mcp_example())
    # asyncio.run(run_napari_mcp_example())
    # asyncio.run(run_chatvis_example())
    # asyncio.run(run_single_case_example())
    asyncio.run(compare_agents_example())
