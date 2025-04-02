# Errors API Reference

This page documents the API for the `agents_arcade.errors` module, which defines custom exceptions for error handling in the Agents Arcade integration.

## Exceptions

### `ToolError`

```python
class ToolError(AgentsException)
```

Exception raised when an Arcade tool execution fails. This exception is raised when a tool returns an unsuccessful response.

#### Attributes

-   `result` (`ExecuteToolResponse`): The ExecuteToolResponse object containing error details.

#### Properties

-   `message` (`str`): A human-readable error message extracted from the response.

#### Methods

-   `__str__()`: Returns a formatted error message including the tool name and error details.

#### Example

```python
from agents_arcade.errors import ToolError
from agents import Runner

try:
    result = await Runner.run(
        starting_agent=agent,
        input="Perform some action with a tool",
        context={"user_id": "user@example.com"},
    )
except ToolError as e:
    print(f"Tool execution failed: {e}")
    print(f"Error message: {e.message}")
```

### `AuthorizationError`

```python
class AuthorizationError(AgentsException)
```

Exception raised when authorization for an Arcade tool fails or is required. This exception includes a URL that the user should visit to complete the authorization process.

#### Attributes

-   `result` (`AuthorizationResponse`): The AuthorizationResponse object containing authorization details.

#### Properties

-   `message` (`str`): A human-readable message indicating that authorization is required, including the authorization URL.

#### Methods

-   `__str__()`: Returns the authorization error message.

#### Example

```python
from agents_arcade.errors import AuthorizationError
from agents import Runner

try:
    result = await Runner.run(
        starting_agent=agent,
        input="Perform some action with a tool",
        context={"user_id": "user@example.com"},
    )
except AuthorizationError as e:
    print(f"Authorization required: {e}")
    # The URL to authorize is in e.result.url
    print(f"Please visit this URL to authorize: {e.result.url}")
```

## Error Handling

When using Agents Arcade tools, it's important to handle these exceptions appropriately:

```python
from agents import Runner, Agent
from agents_arcade import get_arcade_tools
from agents_arcade.errors import ToolError, AuthorizationError

async def run_agent_with_tools():
    tools = await get_arcade_tools(toolkits=["github"])
    agent = Agent(
        name="GitHub agent",
        instructions="You are a helpful assistant that can assist with GitHub API calls.",
        model="gpt-4o-mini",
        tools=tools,
    )

    try:
        result = await Runner.run(
            starting_agent=agent,
            input="List my GitHub repositories",
            context={"user_id": "user@example.com"},
        )
        return result.final_output
    except AuthorizationError as e:
        # Handle authorization requirement
        print(f"Please authorize: {e}")
        return f"Authorization required. Please visit: {e.result.url}"
    except ToolError as e:
        # Handle tool execution failure
        print(f"Tool error: {e}")
        return f"Error executing tool: {e.message}"
```
