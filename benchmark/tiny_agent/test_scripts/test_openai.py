#!/usr/bin/env python3
"""
Simple test script to debug OpenAI API issue with TinyAgent
"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from agent import TinyAgent

async def test_openai():
    """Test OpenAI API connection"""
    try:
        print("Creating TinyAgent from config...")
        agent = TinyAgent.from_config_file("config_openai.json")
        
        print("Entering async context...")
        async with agent:
            print("Loading tools...")
            await agent.load_tools()
            print(f"Loaded {len(agent.available_tools)} tools")
            
            print("Sending test message...")
            query = "create a sphere in paraview"
            
            response_parts = []
            async for chunk in agent.run(query):
                if hasattr(chunk, 'choices') and chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        content = delta.content
                        response_parts.append(content)
                        print(content, end="", flush=True)
                elif hasattr(chunk, 'role') and chunk.role == "tool":
                    print(f"\n[Tool: {chunk.name}] {chunk.content}")
            
            print("\n" + "=" * 50)
            print("Test completed successfully!")
            
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_openai())
