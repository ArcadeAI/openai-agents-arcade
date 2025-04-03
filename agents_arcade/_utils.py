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


async def _get_arcade_tool_formats(
    client: AsyncArcade,
    tools: list[str] | None = None,
    toolkits: list[str] | None = None,
    raise_on_empty: bool = True,
) -> list:
    """
    Asynchronously fetches tool definitions for each toolkit using client.tools.list,
    and returns a list of formatted tools respecting OpenAI's formatting.

    Args:
        client: AsyncArcade client
        tools: Optional list of specific tool names to include.
        toolkits: Optional list of toolkit names to include all tools from.
        raise_on_empty: Whether to raise an error if no tools or toolkits are provided.

    Returns:
        A list of formatted tools respecting OpenAI's formatting.
    """
    if not tools and not toolkits:
        if raise_on_empty:
            raise ValueError(
                "No tools or toolkits provided to retrieve tool definitions")
        return {}

    all_tool_formats = []
    # Retrieve individual tools if specified
    if tools:
        tasks = [client.tools.formatted.get(name=tool_id, format="openai")
                 for tool_id in tools]
        responses = await asyncio.gather(*tasks)
        for response in responses:
            all_tool_formats.append(response)

    # Retrieve tools from specified toolkits
    if toolkits:
        # Create a task for each toolkit to fetch its tool definitions concurrently.
        tasks = [client.tools.formatted.list(toolkit=tk, format="openai")
                 for tk in toolkits]
        responses = await asyncio.gather(*tasks)

        # Combine the tool definitions from each response.
        for response in responses:
            # Here we assume the returned response has an "items" attribute
            # containing a list of ToolDefinition objects.
            all_tool_formats.extend(response.items)

    return all_tool_formats


async def _get_arcade_tool_definitions(
    client: AsyncArcade,
    tools: list[str] | None = None,
    toolkits: list[str] | None = None,
    raise_on_empty: bool = True,
) -> dict[str, bool]:
    """
    Asynchronously fetches tool definitions for each toolkit using client.tools.list,
    and returns a dictionary mapping each tool's name to a boolean indicating whether
    the tool requires authorization.

    Args:
        client: AsyncArcade client
        tools: Optional list of specific tool names to include.
        toolkits: Optional list of toolkit names to include all tools from.
        raise_on_empty: Whether to raise an error if no tools or toolkits are provided.

    Returns:
        A dictionary mapping each tool's name to a boolean indicating whether the
        tool requires authorization.
    """
    if not tools and not toolkits:
        if raise_on_empty:
            raise ValueError(
                "No tools or toolkits provided to retrieve tool definitions")
        return {}

    all_tool_definitions = []
    # Retrieve individual tools if specified
    if tools:
        tasks = [client.tools.get(name=tool_id) for tool_id in tools]
        responses = await asyncio.gather(*tasks)
        for response in responses:
            all_tool_definitions.append(response)

    # Retrieve tools from specified toolkits
    if toolkits:
        # Create a task for each toolkit to fetch its tool definitions concurrently.
        tasks = [client.tools.list(toolkit=toolkit) for toolkit in toolkits]
        responses = await asyncio.gather(*tasks)

        # Combine the tool definitions from each response.
        for response in responses:
            # Here we assume the returned response has an "items" attribute
            # containing a list of ToolDefinition objects.
            all_tool_definitions.extend(response.items)

    # Create dictionary mapping tool name to a boolean for whether authorization is required.
    tool_auth_requirements = {}
    for tool_def in all_tool_definitions:
        # A tool requires authorization if its requirements exist and its
        # authorization is not None.
        requires_auth = bool(
            tool_def.requirements and tool_def.requirements.authorization)
        tool_name = "_".join((tool_def.toolkit.name, tool_def.name))
        tool_auth_requirements[tool_name] = requires_auth

    return tool_auth_requirements
