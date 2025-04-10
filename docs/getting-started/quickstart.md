# Quickstart Guide

This guide will help you get started with creating an agent that uses Arcade tools. We'll create a simple GitHub agent that can interact with the GitHub API.

## Prerequisites

Before you begin, make sure you have:

1. Installed `agents-arcade` (see [Installation](installation.md))
2. An Arcade API key (sign up at [arcade.dev](https://arcade.dev) if you don't have one)
3. An [OpenAI API Key](https://platform.openai.com/docs/libraries#create-and-export-an-api-key).
4. Set the `ARCADE_API_KEY` environment variable or have your API key ready to use
5. Set the `OPENAI_API_KEY` environment variable or have your API key ready to use

## Create a Basic Agent

Below is a simple example of creating an agent that can use GitHub tools:

```python
from agents import Agent, Runner
from arcadepy import AsyncArcade

from agents_arcade import get_arcade_tools
from agents_arcade.errors import AuthorizationError


async def main():
    # Create an Arcade client - either supply the API key or use ARCADE_API_KEY env var
    client = AsyncArcade()

    # Get GitHub tools from Arcade
    tools = await get_arcade_tools(client, toolkits=["github"])

    # Create an OpenAI agent with the GitHub tools
    github_agent = Agent(
        name="GitHub agent",
        instructions="You are a helpful assistant that can assist with GitHub API calls.",
        model="gpt-4o-mini", # You can use any OpenAI model here
        tools=tools,
    )

    try:
        # Run the agent with a specific task
        result = await Runner.run(
            starting_agent=github_agent,
            input="Show me issues from the openai/openai-python repository",
            # A unique user_id is required for authorization
            context={"user_id": "your-unique-user-id"},
        )
        print("Final output:\n\n", result.final_output)
    except AuthorizationError as e:
        # Handle authorization errors
        print("Please login to GitHub:", e)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
```

## Handle Authentication

If this is your first time using a particular Arcade toolkit with your user ID, you will see an authorization URL:

```
Please login to GitHub: Authorization required: https://github.com/login/oauth/authorize...
```

Open this URL in your browser and complete the authorization. After authorizing, you can run your script again, and it should work without requiring authentication again for that user ID.

## Using Other Toolkits

You can use any of the available Arcade toolkits by specifying them in the `get_arcade_tools` function:

```python
# Use the Google toolkit
tools = await get_arcade_tools(client, toolkits=["google"])

# Use multiple toolkits
tools = await get_arcade_tools(client, toolkits=["github", "google", "linkedin"])
```

## Specifying Specific Tools

If you want to use only specific tools from a toolkit:

```python
# Get only specific GitHub tools
tools = await get_arcade_tools(
    client,
    tools=["github_get_issues", "github_get_repository"]
)
```

## Next Steps

-   Learn more about [Authentication](authentication.md)
-   Explore available [Toolkits and Tools](../guides/toolkits.md)
-   See more complex [Examples](../examples/github.md)
-   Check out the [API Reference](../api/tools.md) for advanced usage
