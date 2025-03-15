<h3 align="center">
  <a name="readme-top"></a>
  <img
    src="https://docs.arcade.dev/images/logo/arcade-logo.png"
  >
</h3>
<div align="center">
  <h3>Arcade Integration for OpenAI Agents</h3>
    <a href="https://github.com/your-organization/agents-arcade/blob/main/LICENSE">
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">
</a>
  <a href="https://pypi.org/project/agents-arcade/">
    <img src="https://img.shields.io/pypi/v/agents-arcade.svg" alt="PyPI">
  </a>
</div>

<p align="center">
    <a href="https://docs.arcade.dev" target="_blank">Arcade Documentation</a> •
    <a href="https://docs.arcade.dev/toolkits" target="_blank">Toolkits</a> •
    <a href="https://github.com/ArcadeAI/arcade-py" target="_blank">Arcade Python Client</a> •
    <a href="https://platform.openai.com/docs/guides/agents" target="_blank">OpenAI Agents</a>
</p>

# agents-arcade

`agents-arcade` provides an integration between [Arcade](https://docs.arcade.dev) and the [OpenAI Agents Library](https://github.com/openai/openai-python). This allows you to enhance your AI agents with Arcade's extensive toolkit ecosystem, including tools that reqwuire authorization like Google Mail, Linkedin, X, and more.

## Installation

```bash
pip install agents-arcade
```

## Basic Usage

```python
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
        context={"user_id": "user@example.com"},
    )
    print("Final output:\n\n", result.final_output)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
```

## Key Features

-   **Easy Integration**: Simple API (one function) to connect Arcade tools with OpenAI Agents
-   **Extensive Toolkit Support**: Access to all Arcade toolkits including Gmail, Google Drive, Search, and more
-   **Asynchronous Support**: Built with async/await for compatibility with OpenAI's Agent framework
-   **Authentication Handling**: Manages authorization for tools requiring user permissions like Google, LinkedIn, etc

## Available Toolkits

Arcade provides many toolkits including:

-   **Google**: Gmail, Google Drive, Google Calendar
-   **Search**: Google search, Bing search
-   **Web**: Crawling, scraping, etc
-   **Github**: Repository operations
-   **Slack**: Sending messages to Slack
-   **LinkedIn**: Posting to LinkedIn
-   **X**: Posting and reading tweets on X
-   And many more

For a complete list, see the [Arcade Toolkits documentation](https://docs.arcade.dev/toolkits).

## Authentication

Many Arcade tools require user authentication. The authentication flow is managed by Arcade and can be triggered by providing a `user_id` in the context when running your agent. Refer to the Arcade documentation for more details on managing tool authentication.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
