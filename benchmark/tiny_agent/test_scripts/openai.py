
import asyncio
import os
from huggingface_hub import AsyncInferenceClient

async def main():
    client = AsyncInferenceClient(
        model="claude-3-5-sonnet-20241022",  # Claude model name
        provider="anthropic",                 # Set provider to anthropic
        api_key=os.getenv("ANTHROPIC_API_KEY")  # Your Anthropic API key
    )

    response = await client.chat_completion(
        messages=[
            {"role": "user", "content": "Tell me a fun fact about space."}
        ]
    )

    print(response.choices[0].message["content"])

if __name__ == "__main__":
    asyncio.run(main())