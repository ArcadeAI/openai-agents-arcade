# Agents Arcade

<p align="center">
  <img src="https://docs.arcade.dev/images/logo/arcade-logo.png" alt="Arcade Logo" width="200"/>
</p>

`agents-arcade` provides an integration between [Arcade](https://docs.arcade.dev) and the [OpenAI Agents Library](https://github.com/openai/openai-python). This allows you to enhance your AI agents with Arcade tools like Google Mail, LinkedIn, X, or ones you build yourself with the [Tool SDK](https://github.com/ArcadeAI/arcade-ai).

For a list of all hosted tools and auth providers you can use to build your own, see the [Arcade Integrations](https://docs.arcade.dev/toolkits) documentation.

## What is Agents Arcade?

Agents Arcade connects OpenAI Agents to Arcade's extensive collection of tools and integrations. With Agents Arcade, you can:

-   **Access Powerful Integrations**: Connect your agents to tools like Google Workspace, GitHub, LinkedIn, X, and more
-   **Handle Authentication**: Seamlessly manage user authorization for tools that require it
-   **Build Custom Tools**: Create and integrate your own tools using Arcade's Tool SDK

## Key Features

-   **Easy Integration**: Simple API to connect Arcade tools with OpenAI Agents
-   **Extensive Toolkit Support**: Access to all Arcade toolkits including Gmail, Google Drive, Search, and more
-   **Asynchronous Support**: Built with async/await for compatibility with OpenAI's Agent framework
-   **Authentication Handling**: Manages authorization for tools requiring user permissions

## Installation

```bash
pip install agents-arcade
```

## Requirements 

You need an [Arcade API Key](https://docs.arcade.dev/home/api-keys), and also an
[OpenAI API Key](https://platform.openai.com/docs/libraries#create-and-export-an-api-key).
Make sure they're available


```bash
export ARCADE_API_KEY=...
export OPENAI_API_KEY=...
```

## Quick Example

```python
from agents import Agent, Runner
from arcadepy import AsyncArcade

from agents_arcade import get_arcade_tools
from agents_arcade.errors import AuthorizationError


async def main():
    client = AsyncArcade()
    # Use the "github" toolkit for this example
    tools = await get_arcade_tools(client, toolkits=["github"])

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
```

## Resources

-   [Arcade Documentation](https://docs.arcade.dev)
-   [Arcade Integrations](https://docs.arcade.dev/toolkits)
-   [Arcade Python Client](https://github.com/ArcadeAI/arcade-py)
-   [OpenAI Agents](https://platform.openai.com/docs/guides/agents)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
