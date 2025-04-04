<h3 align="center">
  <a name="readme-top"></a>
  <img
    src="https://docs.arcade.dev/images/logo/arcade-logo.png"
  >
</h3>
<div align="center">
  <h3>Arcade Library for OpenAI Agents</h3>
    <a href="https://github.com/your-organization/agents-arcade/blob/main/LICENSE">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</a>
  <a href="https://pypi.org/project/agents-arcade/">
    <img src="https://img.shields.io/pypi/v/agents-arcade.svg" alt="PyPI">
  </a>
</div>

<p align="center">
    <a href="https://docs.arcade.dev" target="_blank">Arcade Documentation</a> •
    <a href="https://docs.arcade.dev/toolkits" target="_blank">Integrations</a> •
    <a href="https://github.com/ArcadeAI/arcade-py" target="_blank">Arcade Python Client</a> •
    <a href="https://platform.openai.com/docs/guides/agents" target="_blank">OpenAI Agents</a>
</p>

# agents-arcade

`agents-arcade` provides an integration between [Arcade](https://docs.arcade.dev) and the [OpenAI Agents Library](https://github.com/openai/openai-python). This allows you to enhance your AI agents with Arcade tools like Google Mail, Linkedin, X, or ones you build yourself with the [Tool SDK](https://github.com/ArcadeAI/arcade-ai).

For a list of all hosted tools and auth providers you can use to build your own, see the [Arcade Integrations](https://docs.arcade.dev/toolkits) documentation.

## Installation

```bash
pip install agents-arcade
```

## Basic Usage

Below shows a basic example of how to use the `agents-arcade` library to create an agent that can use the
GitHub toolkit hosted in Arcade Cloud. You can use other hosted tools like Google, or you can build your own
and host them to the agents library with the [Tool SDK](https://github.com/ArcadeAI/arcade-ai).

```python
from agents import Agent, Runner
from arcadepy import AsyncArcade

from agents_arcade import get_arcade_tools
from agents_arcade.errors import AuthorizationError


async def main():
    client = AsyncArcade()
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
            context={"user_id": "user@example.comiii"},
        )
        print("Final output:\n\n", result.final_output)
    except AuthorizationError as e:
        print("Please Login to Github:", e)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
```

If you haven't auth'd Arcade with GitHub yet, you'll see a message similar to this:

```bash
> python examples/github.py
Please Login to Github: Authorization required: https://github.com/login/oauth/authorize...
```

You can then visit the URL in your browser to auth'd Arcade with GitHub and run the script again.

```bash
> python examples/github.py
The repository **arcadeai/arcade-ai** has been successfully starred! If you need any more assistance, feel free to ask.
```

You can also wait for authorization to complete before running the script using the helper
methods in arcadepy.

Once you have auth'd Arcade with a tool, you can use the tool in your agent by passing the `user_id`
and never having to worry about auth'ing for that specific tool again.

## Key Features

-   **Easy Integration**: Simple API (one function) to connect Arcade tools with OpenAI Agents
-   **Extensive Toolkit Support**: Access to all Arcade toolkits including Gmail, Google Drive, Search, and more
-   **Asynchronous Support**: Built with async/await for compatibility with OpenAI's Agent framework
-   **Authentication Handling**: Manages authorization for tools requiring user permissions like Google, LinkedIn, etc

## Authentication

Many Arcade tools require user authentication. The authentication flow is managed by Arcade and can be triggered by providing a `user_id` in the context when running your agent. Refer to the Arcade documentation for more details on managing tool authentication.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Documentation

The project documentation is available at [docs.arcadeai.dev/agents-arcade](https://docs.arcadeai.dev/agents-arcade/) and includes:

-   Installation instructions
-   Quickstart guides
-   API reference
-   Advanced usage patterns
-   Toolkit guides
-   Examples

To build and serve the documentation locally:

```bash
# Install development dependencies
pip install -e ".[dev]"

# Serve the documentation
make serve-docs
# or
mkdocs serve
```

Then visit `http://localhost:8000` in your browser.
