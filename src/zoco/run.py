
from BaseAgent import OpenAiChatCompletionClient,Agent
import asyncio


async def main():
    # 1. Initialize the client
    client = OpenAiChatCompletionClient(
        model="gpt-4o-mini",
        api_key="your api key"
    )
    # 2. Initialize the agent with instructions and the client
    my_agent = Agent(
        name="Assistant",
        instructions="You are a helpful assistant.",
        model_client=client
    )

    # 3. Invoke run_stream (returns an AsyncGenerator)
    stream = await my_agent.run(task="count from 1 to 20")
    print(stream)

    # 4. Iterate through the generator to pull the response
    

if __name__ == "__main__":
    asyncio.run(main())