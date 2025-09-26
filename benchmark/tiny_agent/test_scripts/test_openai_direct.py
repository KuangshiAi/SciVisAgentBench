#!/usr/bin/env python3
"""
Simple test script to test OpenAI API directly with TinyAgent
"""
import asyncio
import json
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from agent import TinyAgent

async def test_openai_direct():
    """Test OpenAI API connection directly"""
    try:
        print("Creating TinyAgent from config...")
        agent = TinyAgent.from_config_file("config_openai.json")
        
        print("Testing without MCP tools...")
        # Create a simple agent without servers for testing
        agent._servers_cfg = []  # Remove MCP servers for testing
        
        async with agent:
            print("Sending test message...")
            
            # Test the chat completion directly
            response = await agent.client.chat_completion(
                model=agent.payload_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello! Can you tell me what 2+2 equals?"}
                ],
                max_tokens=100
            )
            
            print("Response:")
            print(response.choices[0].message.content)
            print("=" * 50)
            print("OpenAI API test successful!")
            
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_openai_direct())
