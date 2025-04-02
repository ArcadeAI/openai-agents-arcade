from agents import Agent, Runner
from arcadepy import AsyncArcade

from agents_arcade import get_arcade_tools
from agents_arcade.errors import AuthorizationError


async def main():
    client = AsyncArcade(api_key="arc_o1SX4VZ563zrDz1oJGgdtfUHFH8FAtvD8JSrxoTjxmsaNo4eap3H")
    # Use the "github" toolkit for this example
    # You can use other toolkits like "google", "linkedin", "x", etc.
    tools = await get_arcade_tools(client, ["github"])

    # Create an agent that can use the github toolkit
    github_agent = Agent(
        name="GitHub agent",
        instructions="You are a helpful assistant that can assist with GitHub API calls.",
        model="gpt-4o-mini",
        tools=tools,
    )

    try:
        result = await Runner.run(
            starting_agent=github_agent,
            input="Star the arcadeai/arcade-ai repo",
            # make sure you pass a UNIQUE user_id for auth
            context={"user_id": "user@example.com"},
        )
        print("Final output:\n\n", result.final_output)
    except AuthorizationError as e:
        print("Please Login to Github:", e)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
