# Authentication

Many Arcade tools require user authentication to access third-party services like GitHub, Google, LinkedIn, etc. This guide explains how authentication works in `agents-arcade` and how to implement it properly.

## Overview

The authentication flow in `agents-arcade` is handled by Arcade and follows this process:

1. You provide a unique `user_id` in the agent's context
2. When a tool requires authentication, Arcade checks if the user is already authorized
3. If not authorized, an `AuthorizationError` is raised with a URL for the user to authenticate
4. After authentication, subsequent calls with the same `user_id` will be authorized

## Implementing Authentication

### Basic Authentication Flow

Here's a simple example of handling authentication in your agent:

```python
from agents import Agent, Runner
from arcadepy import AsyncArcade

from agents_arcade import get_arcade_tools
from agents_arcade.errors import AuthorizationError

async def main():
    client = AsyncArcade()
    tools = await get_arcade_tools(client, toolkits=["github"])

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
            # Provide a unique user_id for authentication
            context={"user_id": "user@example.com"},
        )
        print("Final output:", result.final_output)
    except AuthorizationError as e:
        # Show the authentication URL to the user
        print("Please login to GitHub:", e)

if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
```

### Waiting for Authentication

In some scenarios, you might want to wait for the user to complete the authentication before proceeding. You can use the `arcadepy` library's authorization helpers for this:

```python
from arcadepy import AsyncArcade
from agents_arcade import get_arcade_tools
from agents_arcade.errors import AuthorizationError

async def main():
    client = AsyncArcade()
    user_id = "user@example.com"

    try:
        # Try to use a tool that requires auth
        tools = await get_arcade_tools(client, toolkits=["github"])

        # Attempt to run your agent...
    except AuthorizationError as e:
        # Get the auth URL from the error
        auth_url = e.result.url

        # Show URL to user
        print(f"Please authorize access at: {auth_url}")

        # Wait for the user to complete authorization
        authorization = await client.wait_for_completion(
            e.result.authorization_id)

        if authorization.status == "completed":
            print("Authorization completed successfully!")
            # Try again with the authorized user
            # ...

if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
```

## User ID Best Practices

The `user_id` is crucial for authentication as it links tool authorizations to specific users:

-   **Use a consistent identifier**: Use the same `user_id` for the same user across sessions
-   **Make it unique**: Ensure each user has a unique identifier to avoid authorization conflicts
-   **Privacy considerations**: Consider using hashed or anonymized IDs if privacy is a concern
-   **Persistent storage**: Store authorized user IDs securely to maintain the user's authorization state

## Managing Multiple Users

If your application serves multiple users, you should maintain a mapping between your internal user identifiers and the user IDs you provide to Arcade.

## Revoking Access

Users can revoke access to their accounts from the third-party service's settings (e.g., GitHub settings). When access is revoked, the next tool call will raise an `AuthorizationError` again, prompting re-authentication.

## Troubleshooting

### Common Issues

-   **Authentication URL not working**: Ensure your Arcade API key has the correct permissions
-   **Authorization expired**: Third-party tokens can expire; handle `AuthorizationError` to re-authenticate
-   **Authorization stuck**: If authorization seems stuck, check if the timeout value is appropriate

### Debugging Authentication

You can enable debug logging in `arcadepy` to see more detailed information about the authentication process:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```
