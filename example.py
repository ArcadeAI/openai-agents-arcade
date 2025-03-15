from agents import Agent, RunConfig, Runner
from arcadepy import AsyncArcade

from agents_arcade import get_arcade_tools


async def main():
    client = AsyncArcade()
    tools = await get_arcade_tools(client, ["google"])

    google_agent = Agent(
        name="Google agent",
        instructions="You are a helpful assistant that can assist with Google API calls.",
        model="gpt-4o-mini",
        tools=tools,
    )

    result = await Runner.run(
        starting_agent=google_agent,
        input="What are my latest emails?",
        context={"user_id": "user@example.com3"},
        run_config=RunConfig(
            tracing_disabled=True,
        ),
    )
    print("Final output:\n\n", result.final_output)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
