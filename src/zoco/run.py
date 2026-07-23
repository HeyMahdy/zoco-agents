import asyncio
from BaseAgent import OpenAiChatCompletionClient, Agent




def get_weather(city: str) -> str:
    """Returns the current weather for a specific city."""
    # This is mock logic. In a real app, you'd call an API here.
    if city.lower() == "dhaka":
        return "It is currently 32 degrees Celsius and raining in Dhaka."
    return f"It is 25 degrees and sunny in {city}."



async def main():
    # 1. Initialize client
    client = OpenAiChatCompletionClient(
        model="gpt-4o-mini",
        api_key="your-api-key" # <-- PUT YOUR API KEY HERE
    )
    
    # 2. Create Agent and pass the tool
    my_agent = Agent(
        name="Assistant",
        instructions="You are a helpful assistant.",
        model_client=client,
        tools=[get_weather] # <-- Pass our test tool here!
    )

    # 3. Ask a question that forces it to use the tool
    print("User: What is the weather like in Dhaka right now?")
    response = await my_agent.run(task="What is the weather like in Dhaka right now?")
    
    # 4. Print final answer
    print(f"\nAssistant: {response}")

if __name__ == "__main__":
    asyncio.run(main())
