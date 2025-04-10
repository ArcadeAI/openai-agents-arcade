# Tools API Reference

This page documents the API for the `agents_arcade.tools` module, which provides integration between OpenAI Agents and Arcade tools.

## Functions

### `get_arcade_tools`

```python
async def get_arcade_tools(
    client: AsyncArcade | None = None,
    toolkits: list[str] | None = None,
    tools: list[str] | None = None,
    **kwargs: dict[str, Any],
) -> list[FunctionTool]
```

Asynchronously fetches Arcade tools and converts them into OpenAI Agent-compatible FunctionTool objects.

!!! note "Not all tools require authentication"
While many Arcade tools require user authentication, some don't. The `get_arcade_tools` function automatically handles the distinction.

#### Parameters

-   `client` (`AsyncArcade | None`): Optional AsyncArcade client. If not provided, one will be created using environment variables or the provided kwargs.
-   `toolkits` (`list[str] | None`): Optional list of toolkit names to fetch tools from (e.g., `["github", "google"]`).
-   `tools` (`list[str] | None`): Optional list of specific tool names to include. If None, all tools from the specified toolkits are included.
-   `**kwargs` (`dict[str, Any]`): Additional arguments passed to `get_arcade_client` if a client is not provided.

#### Returns

-   `list[FunctionTool]`: A list of `FunctionTool` objects that can be used with OpenAI Agents.

#### Exceptions

-   `ValueError`: If no API key is provided and the environment variable is not set.
-   Authentication errors are not raised here but when the tools are used.

#### Example

```python
from agents_arcade import get_arcade_tools
from arcadepy import AsyncArcade

# Get all tools from a specific toolkit
client = AsyncArcade()
tools = await get_arcade_tools(client, toolkits=["github"])

# Get specific tools
tools = await get_arcade_tools(
    client,
    tools=["github_get_issues", "github_get_repository"]
)

# Create a client with a specific API key and base URL
tools = await get_arcade_tools(
    None,  # No client provided, will create one with the kwargs
    toolkits=["google"],
    api_key="your_api_key",
    base_url="https://api.arcade.dev"
)
```

## Internal Functions

These functions are used internally by the library and are not typically called directly.

### `_async_invoke_arcade_tool`

```python
async def _async_invoke_arcade_tool(
    context: RunContextWrapper,
    tool_args: str,
    tool_name: str,
    requires_auth: bool,
    client: AsyncArcade,
)
```

Internal function used to execute Arcade tools. Handles authorization when required and processes tool execution results.

### `_authorize_tool`

```python
async def _authorize_tool(client: AsyncArcade, context: RunContextWrapper, tool_name: str)
```

Internal function to authorize a tool for a specific user. Raises appropriate exceptions if authorization fails.
