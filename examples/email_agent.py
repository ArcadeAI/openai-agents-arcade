from agents import Agent, Runner
from arcadepy import AsyncArcade

from agents_arcade import get_arcade_tools
from agents_arcade.errors import AuthorizationError


async def main():
    client = AsyncArcade()
    tools = await get_arcade_tools(client, toolkits=["google"])

    google_agent = Agent(
        name="Google agent",
        instructions="You are a helpful assistant that can assist with Google API calls.",
        model="gpt-4o-mini",
        tools=tools,
    )

    try:
        result = await Runner.run(
            starting_agent=google_agent,
            input="What are my latest emails?",
            context={"user_id": "user@example.com"},
        )
        print("Final output:\n\n", result.final_output)
    except AuthorizationError as e:
        print("Please Login to Google:", e)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
