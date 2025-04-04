# Combining Multiple Toolkits

This example demonstrates how to create an agent that combines multiple Arcade toolkits to provide a powerful multi-service assistant.

## Overview

One of the most powerful features of Agents Arcade is the ability to combine multiple toolkits in a single agent. This example shows how to create an agent that can:

1. Access GitHub repositories and issues
2. Read and send emails via Gmail
3. Search the web for information

## Prerequisites

Before running this example, make sure you have:

1. Installed `agents-arcade` (see [Installation](installation.md))
2. An Arcade API key (sign up at [arcade.dev](https://arcade.dev) if you don't have one)
3. An [OpenAI API Key](https://platform.openai.com/docs/libraries#create-and-export-an-api-key).
4. Set the `ARCADE_API_KEY` environment variable or have your API key ready to use
5. Set the `OPENAI_API_KEY` environment variable or have your API key ready to use
3. Have access to the GitHub, Google, and Web toolkits in your Arcade subscription

## Multi-Toolkit Agent

Here's a complete example of a multi-toolkit agent:

```python
from agents import Agent, Runner
from arcadepy import AsyncArcade

from agents_arcade import get_arcade_tools
from agents_arcade.errors import AuthorizationError


async def main():
    # Create an Arcade client
    client = AsyncArcade()

    # Get tools from multiple toolkits
    tools = await get_arcade_tools(
        client,
        toolkits=["github", "google", "web"]
    )

    # Create an agent with custom instructions that reference all capabilities
    multi_agent = Agent(
        name="Multi-service Assistant",
        instructions="""You are a helpful assistant that can:
        1. Access GitHub repositories, issues, and pull requests
        2. Read and send emails via Gmail
        3. Search the web for information

        For GitHub tasks: Use the github_* tools to interact with GitHub
        For email tasks: Use the google_gmail_* tools to read or send emails
        For web searches: Use the web_search tool to find information online

        Always use the most appropriate tools for the user's request.
        If authentication is required for a service, inform the user.
        """,
        model="gpt-4o-mini",
        tools=tools,
    )

    try:
        # Run the agent with a specific multi-service task
        result = await Runner.run(
            starting_agent=multi_agent,
            input="""
            1. Check if there are any open issues in the 'arcadeai/arcade-ai' GitHub repo
            2. Then search the web for 'agents and tools architectures'
            3. Finally, draft an email summarizing the findings
            """,
            # A unique user_id is required for authentication
            context={"user_id": "user@example.com"},
        )
        print("Final output:\n\n", result.final_output)
    except AuthorizationError as e:
        # Handle authorization errors - might need multiple authorizations
        print(f"Authorization required: {e}")
        print(f"Please visit: {e.result.url}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
```

Save this as `multi_toolkit_example.py` and run it with `python multi_toolkit_example.py`.

## Authentication Flow

When running this example for the first time, you'll need to authenticate with multiple services. The authentication will happen one service at a time:

1. You might first get a GitHub authentication prompt:

    ```
    Authorization required: https://github.com/login/oauth/authorize...
    ```

2. After authenticating with GitHub and running again, you might get a Google authentication prompt:

    ```
    Authorization required: https://accounts.google.com/o/oauth2/auth...
    ```

3. After completing all required authorizations, the agent will be able to execute the tasks.

## Handling Multiple Authorizations

For a better user experience, you can implement a more sophisticated authentication flow:

```python
async def run_with_multi_auth():
    client = AsyncArcade()
    user_id = "user@example.com"

    # Track which services have been authorized
    auth_status = {
        "github": False,
        "google": False,
        "web": False  # Some web tools might require auth too
    }

    # Keep trying until all services are authorized
    while not all(auth_status.values()):
        try:
            # Get tools for services not yet authorized
            toolkits = [tk for tk, status in auth_status.items() if not status]
            tools = await get_arcade_tools(client, toolkits)

            # Create agent with the tools
            agent = Agent(
                name="Multi-service Assistant",
                instructions="You are a helpful assistant with multiple service capabilities.",
                model="gpt-4o-mini",
                tools=tools,
            )

            # Try to run the agent
            result = await Runner.run(
                starting_agent=agent,
                input="Test connection to all services",
                context={"user_id": user_id},
            )

            # If we get here without an AuthorizationError, all services are authorized
            print("All services successfully authorized!")
            auth_status = {tk: True for tk in auth_status}

        except AuthorizationError as e:
            # Extract the toolkit from the tool name (e.g., github_get_issues -> github)
            toolkit = e.result.tool_name.split('_')[0]

            print(f"Authorization required for {toolkit}. Please visit: {e.result.url}")
            print("Waiting for authorization completion...")

            # Wait for the user to complete authorization
            auth_result = await client.wait_for_completion(
                e.result.authorization_id
            )

            if auth_result.status == "completed":
                print(f"{toolkit} authorization completed successfully!")
                auth_status[toolkit] = True
            else:
                print(f"{toolkit} authorization failed or timed out.")

    # Now create the fully authorized agent
    tools = await get_arcade_tools(client, list(auth_status.keys()))
    return Agent(
        name="Multi-service Assistant",
        instructions="You are a helpful assistant with multiple service capabilities.",
        model="gpt-4o-mini",
        tools=tools,
    )
```

## Advanced Use Cases

With a multi-toolkit agent, you can build powerful assistants that can perform complex tasks across different services. Here are some example use cases:

1. **Research Assistant**: Search the web, save findings to GitHub, and email summaries
2. **Project Manager**: Track GitHub issues, send email updates, and search for solutions
3. **Content Creator**: Research topics online, draft content in Google Docs, and share via email
4. **Customer Support**: Search knowledge bases, check GitHub issues, and respond to customer emails

## Conclusion

By combining multiple toolkits, you can create sophisticated agents that seamlessly work across different services. This approach allows you to build truly helpful assistants that can handle complex workflows that span multiple platforms and services.
