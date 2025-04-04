# Utils API Reference

This page documents the API for the `agents_arcade._utils` module, which provides utility functions for working with the Arcade API.

## Functions

### `get_arcade_client`

```python
def get_arcade_client(
    base_url: str = "https://api.arcade.dev",
    api_key: str = os.getenv("ARCADE_API_KEY", None),
    **kwargs: dict[str, Any],
) -> AsyncArcade
```

Creates and returns an AsyncArcade client.

#### Parameters

-   `base_url` (`str`): The base URL for the Arcade API. Default is "https://api.arcade.dev".
-   `api_key` (`str`): The Arcade API key. If not provided, it will be read from the `ARCADE_API_KEY` environment variable.
-   `**kwargs` (`dict[str, Any]`): Additional arguments to pass to the AsyncArcade constructor.

#### Returns

-   `AsyncArcade`: An initialized AsyncArcade client.

#### Raises

-   `ValueError`: If no API key is provided and the environment variable is not set.

#### Example

```python
from agents_arcade._utils import get_arcade_client

# Using environment variable for API key
client = get_arcade_client()

# Explicitly providing API key
client = get_arcade_client(api_key="your_api_key_here")

# Custom base URL (e.g., for staging environment)
client = get_arcade_client(base_url="https://api.staging.arcade.dev")
```

### `_get_arcade_tool_definitions`

```python
async def _get_arcade_tool_definitions(
    client: AsyncArcade,
    toolkits: list[str],
    tools: list[str] | None = None
) -> dict[str, bool]
```

Asynchronously fetches tool definitions from specified toolkits and determines which tools require authorization.

#### Parameters

-   `client` (`AsyncArcade`): AsyncArcade client to use for API requests.
-   `toolkits` (`list[str]`): List of toolkit names to fetch tools from.
-   `tools` (`list[str] | None`): Optional list of specific tool names to include. If None, all tools are included.

#### Returns

-   `dict[str, bool]`: A dictionary mapping tool names to booleans indicating whether each tool requires authorization.

### `convert_output_to_json`

```python
def convert_output_to_json(output: Any) -> str
```

Converts tool output to a JSON string.

#### Parameters

-   `output` (`Any`): Any value to convert to JSON.

#### Returns

-   `str`: A JSON string representation of the output. If the input is already a dict or list, it's converted to JSON. Otherwise, it's converted to a string.

#### Example

```python
from agents_arcade._utils import convert_output_to_json

# Convert a dictionary to JSON
result = convert_output_to_json({"name": "John", "age": 30})
# '{"name": "John", "age": 30}'

# Convert a list to JSON
result = convert_output_to_json([1, 2, 3])
# '[1, 2, 3]'

# Convert a non-JSON value to string
result = convert_output_to_json("Hello")
# 'Hello'
```
