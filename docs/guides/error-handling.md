# Error Handling

This guide covers how to properly handle errors when using `agents-arcade` with OpenAI Agents. Effective error handling is crucial for creating a smooth user experience, especially when working with tools that require authentication.

## Common Errors

When working with Arcade tools, you might encounter several types of errors:

1. **Authorization Errors**: Occur when a tool requires authentication but the user hasn't completed the authorization flow
2. **Tool Execution Errors**: Occur when a tool fails to execute properly
3. **Client Errors**: Issues with the Arcade client itself, such as invalid API keys or network problems

## Handling Authorization Errors

The most common error when using Arcade tools is the `AuthorizationError`, which occurs when a tool requires user authorization.

### Basic Authorization Error Handling

```python
from agents import Agent, Runner
from arcadepy import AsyncArcade
from agents_arcade import get_arcade_tools
from agents_arcade.errors import AuthorizationError

async def run_agent():
    client = AsyncArcade()
    tools = await get_arcade_tools(client, ["github"])

    agent = Agent(
        name="GitHub Agent",
        instructions="You are a helpful assistant for GitHub.",
        model="gpt-4o-mini",
        tools=tools,
    )

    try:
        result = await Runner.run(
            starting_agent=agent,
            input="List my repositories",
            context={"user_id": "user@example.com"},
        )
        return result.final_output
    except AuthorizationError as e:
        # Extract the authorization URL and present it to the user
        auth_url = e.result.url
        return f"Please authorize access at: {auth_url}"
```

### Interactive Authorization Flow

For a better user experience, you might want to guide the user through the authorization process and retry the operation once authorized:

```python
from arcadepy.auth import wait_for_authorization_completion

async def run_with_authorization():
    client = AsyncArcade()
    user_id = "user@example.com"

    try:
        tools = await get_arcade_tools(client, ["github"])
        agent = Agent(
            name="GitHub Agent",
            instructions="You are a helpful GitHub assistant.",
            model="gpt-4o-mini",
            tools=tools,
        )

        return await Runner.run(
            starting_agent=agent,
            input="List my repositories",
            context={"user_id": user_id},
        ).final_output
    except AuthorizationError as e:
        print(f"Authorization required. Please visit: {e.result.url}")
        print("Once you've completed authorization, the operation will continue...")

        # Wait for the user to complete authorization
        auth_result = await wait_for_authorization_completion(
            client,
            e.result.authorization_id,
            timeout=300  # 5 minutes
        )

        if auth_result.status == "completed":
            print("Authorization successful! Retrying operation...")
            # Retry the operation now that authorization is complete
            return await run_with_authorization()
        else:
            return "Authorization failed or timed out."
```

## Handling Tool Execution Errors

When a tool fails to execute properly, a `ToolError` is raised with details about what went wrong.

```python
from agents_arcade.errors import ToolError, AuthorizationError

async def run_agent_safely():
    try:
        tools = await get_arcade_tools(client, ["github"])
        agent = Agent(
            name="GitHub Agent",
            instructions="You are a helpful GitHub assistant.",
            model="gpt-4o-mini",
            tools=tools,
        )

        return await Runner.run(
            starting_agent=agent,
            input="Create a repository named my-repo",
            context={"user_id": "user@example.com"},
        ).final_output
    except AuthorizationError as e:
        return f"Authorization required: {e.result.url}"
    except ToolError as e:
        # Handle specific tool errors
        error_message = e.message

        if "already exists" in error_message.lower():
            return "A repository with that name already exists. Please try a different name."
        elif "permission" in error_message.lower():
            return "You don't have permission to perform this action."
        else:
            return f"An error occurred: {error_message}"
```

## Comprehensive Error Handling

For a production application, you'll want to handle all potential errors:

```python
from agents_arcade.errors import ToolError, AuthorizationError
from arcadepy.exceptions import ArcadeError

async def run_with_comprehensive_error_handling():
    try:
        # Create client
        client = AsyncArcade()

        # Get tools
        try:
            tools = await get_arcade_tools(client, ["github", "google"])
        except ValueError as e:
            return f"Configuration error: {e}"
        except ArcadeError as e:
            return f"Arcade client error: {e}"

        # Create agent
        agent = Agent(
            name="Multi-tool Agent",
            instructions="You can help with GitHub and Google services.",
            model="gpt-4o-mini",
            tools=tools,
        )

        # Run agent
        try:
            result = await Runner.run(
                starting_agent=agent,
                input="List my GitHub repositories and recent emails",
                context={"user_id": "user@example.com"},
            )
            return result.final_output
        except AuthorizationError as e:
            return f"Authorization required: {e.result.url}"
        except ToolError as e:
            return f"Tool execution failed: {e.message}"
        except Exception as e:
            return f"Unexpected error running agent: {e}"

    except Exception as e:
        return f"Unexpected error: {e}"
```

## User-Friendly Error Messages

When displaying errors to end users, it's important to provide helpful guidance:

```python
def handle_arcade_error(error):
    if isinstance(error, AuthorizationError):
        return {
            "type": "auth_required",
            "message": "Authentication required",
            "details": "Please click the link below to authorize access",
            "url": error.result.url,
            "auth_id": error.result.authorization_id
        }
    elif isinstance(error, ToolError):
        # Customize based on the tool and error message
        tool_name = error.result.tool_name
        error_msg = error.message

        if tool_name.startswith("github"):
            return handle_github_error(error_msg)
        elif tool_name.startswith("google"):
            return handle_google_error(error_msg)
        else:
            return {
                "type": "tool_error",
                "message": f"Error using {tool_name}",
                "details": error_msg
            }
    else:
        return {
            "type": "unknown_error",
            "message": "An unexpected error occurred",
            "details": str(error)
        }

def handle_github_error(error_msg):
    if "not found" in error_msg.lower():
        return {
            "type": "not_found",
            "message": "GitHub repository not found",
            "details": "Please check the repository name and try again"
        }
    # Add more specific GitHub error handling
    return {
        "type": "github_error",
        "message": "GitHub error",
        "details": error_msg
    }
```

## Logging Errors for Debugging

For debugging and monitoring purposes, it's important to log errors:

```python
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("agents-arcade")

async def run_with_logging():
    try:
        # Your agent code here
        pass
    except AuthorizationError as e:
        logger.info(f"Authorization required for user: {user_id}, tool: {e.result.tool_name}")
        # Handle error for user
    except ToolError as e:
        logger.error(f"Tool error: {e.result.tool_name} failed with: {e.message}")
        # Handle error for user
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        # Handle error for user
```

## Conclusion

Proper error handling is essential for creating a good user experience with Arcade tools. By anticipating and gracefully handling different types of errors, you can guide users through authentication processes and provide helpful feedback when things go wrong.
