import asyncio
from BaseAgent import OpenAiChatCompletionClient, Agent

async def main():
    # 1. Initialize the decoupled LLM Client
    client = OpenAiChatCompletionClient(
        model="gpt-4o-mini",
        api_key="your-api-key-here"
    )
    
    # 2. Define the Agent and attach the client
    my_agent = Agent(
        name="Assistant",
        instructions="You are a helpful assistant.",
        model_client=client
    )

    # 3. Execute the task asynchronously
    response = await my_agent.run(task="Hello World")
    
    print(response) # The result! 🚀

if __name__ == "__main__":
    asyncio.run(main())