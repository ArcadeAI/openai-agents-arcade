# GitHub Integration Example

This example shows how to create an agent that can interact with GitHub using Arcade tools.

## Prerequisites

Before running this example, make sure you have:

1. Installed `agents-arcade` (see [Installation](../getting-started/installation.md))
2. Set the `ARCADE_API_KEY` environment variable or have your API key ready to use

## Basic GitHub Agent

Here's a complete example of a GitHub agent that can interact with the GitHub API:

```python
from agents import Agent, Runner
from arcadepy import AsyncArcade

from agents_arcade import get_arcade_tools
from agents_arcade.errors import AuthorizationError


async def main():
    # Create an Arcade client
    client = AsyncArcade()

    # Get GitHub tools from Arcade
    tools = await get_arcade_tools(client, ["github"])

    # Create an OpenAI agent with the GitHub tools
    github_agent = Agent(
        name="GitHub agent",
        instructions="You are a helpful assistant that can assist with GitHub API calls.",
        model="gpt-4o-mini",
        tools=tools,
    )

    try:
        # Run the agent with a specific task
        result = await Runner.run(
            starting_agent=github_agent,
            input="Star the arcadeai/arcade-ai repo",
            # A unique user_id is required for GitHub authorization
            context={"user_id": "user@example.com"},
        )
        print("Final output:\n\n", result.final_output)
    except AuthorizationError as e:
        # Handle authorization errors
        print("Please Login to Github:", e)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
```

Save this as `github_example.py` and run it with `python github_example.py`.

## What This Example Does

1. Creates an Arcade client using your API key
2. Fetches GitHub tools from Arcade
3. Creates an OpenAI agent with access to those tools
4. Runs the agent with a specific task ("Star the arcadeai/arcade-ai repo")
5. Handles any authorization errors that might occur

## Authentication Process

If this is your first time using the GitHub toolkit with your user ID, you will see a message like:

```
Please Login to Github: Authorization required: https://github.com/login/oauth/authorize...
```

Open this URL in your browser and complete the GitHub authorization process. After authorizing, you can run the script again, and it should work without requiring authentication.

## Common GitHub Tasks

Your GitHub agent can perform a wide range of tasks. Here are some example prompts you can try:

-   "List my GitHub repositories"
-   "Create a new repository named 'test-repo'"
-   "Get information about the openai/openai-python repository"
-   "List open issues in the huggingface/transformers repository"
-   "Create a new issue in my test-repo repository"

## Available GitHub Tools

The GitHub toolkit includes tools for:

-   Repository management (create, get, list, delete)
-   Issue management (create, get, list, update)
-   Pull request operations
-   Star/unstar repositories
-   User information
-   And more

## Advanced Usage

### Filtering Tools

If you only need specific GitHub tools, you can include only what you need:

```python
tools = await get_arcade_tools(
    client,
    tools=["github_get_repository", "github_list_user_repositories"]
)
```

### Combining with Other Toolkits

You can combine GitHub tools with other Arcade toolkits:

```python
tools = await get_arcade_tools(client, ["github", "google", "linkedin"])
```

### Custom Instructions

For more targeted GitHub tasks, you can customize the agent's instructions:

```python
github_agent = Agent(
    name="GitHub Repository Manager",
    instructions="""You are a specialized GitHub repository manager.
    You can help users manage their repositories, issues, and pull requests.
    Always ask for clarification if the repository name is not specified.
    When creating issues or PRs, ask for all necessary details.""",
    model="gpt-4o-mini",
    tools=tools,
)
```
