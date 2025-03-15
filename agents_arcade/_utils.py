import asyncio
import json
from typing import Any

from arcadepy import AsyncArcade


def convert_output_to_json(output: Any) -> str:
    if isinstance(output, dict) or isinstance(output, list):
        return json.dumps(output)
    else:
        return str(output)


async def get_arcade_client() -> AsyncArcade:
    return AsyncArcade()


async def _get_arcade_tool_definitions(
    client: AsyncArcade, toolkits: list[str], tools: list[str] | None = None
) -> dict[str, bool]:
    """
    Asynchronously fetches tool definitions for each toolkit using client.tools.list,
    and returns a dictionary mapping each tool's name to a boolean indicating whether
    the tool requires authorization.

    Args:
        client: AsyncArcade client
        toolkits: List of toolkit names to get tools from
        tools: Optional list of specific tool names to include. If None, all tools are included.
    """
    # Create a task for each toolkit to fetch its tool definitions concurrently.
    tasks = [client.tools.list(toolkit=toolkit) for toolkit in toolkits]
    responses = await asyncio.gather(*tasks)

    # Combine the tool definitions from each response.
    all_tool_definitions = []
    for response in responses:
        # Here we assume the returned response has an "items" attribute
        # containing a list of ToolDefinition objects.
        all_tool_definitions.extend(response.items)

    # Create dictionary mapping tool name to a boolean for whether authorization is required.
    tool_auth_requirements = {}
    for tool_def in all_tool_definitions:
        # If tools is None, include all tools
        # If tools is not None, only include tools in the list
        if tools is None or tool_def.name in tools:
            # A tool requires authorization if its requirements exist and its
            # authorization is not None.
            requires_auth = bool(tool_def.requirements and tool_def.requirements.authorization)
            tool_name = "_".join((tool_def.toolkit.name, tool_def.name))
            tool_auth_requirements[tool_name] = requires_auth

    return tool_auth_requirements
