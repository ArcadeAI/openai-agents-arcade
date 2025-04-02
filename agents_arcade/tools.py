import json
from functools import partial
from typing import Any

from agents.run_context import RunContextWrapper
from agents.tool import FunctionTool
from arcadepy import AsyncArcade

from agents_arcade._utils import (
    _get_arcade_tool_definitions,
    convert_output_to_json,
    get_arcade_client,
)
from agents_arcade.errors import AuthorizationError, ToolError


async def _authorize_tool(client: AsyncArcade, context: RunContextWrapper, tool_name: str):
    if not context.context.get("user_id"):
        raise ToolError("No user ID and authorization required for tool")

    result = await client.tools.authorize(
        tool_name=tool_name,
        user_id=context.context.get("user_id"),
    )
    if result.status != "completed":
        raise AuthorizationError(result)


async def _async_invoke_arcade_tool(
    context: RunContextWrapper,
    tool_args: str,
    tool_name: str,
    requires_auth: bool,
    client: AsyncArcade,
):
    args = json.loads(tool_args)
    if requires_auth:
        await _authorize_tool(client, context, tool_name)

    result = await client.tools.execute(
        tool_name=tool_name,
        input=args,
        user_id=context.context.get("user_id"),
    )

    if not result.success:
        raise ToolError(result)

    return convert_output_to_json(result.output.value)


async def get_arcade_tools(
    client: AsyncArcade | None = None,
    toolkits: list[str] | None = None,
    tools: list[str] | None = None,
    **kwargs: dict[str, Any],
) -> list[FunctionTool]:
    if not client:
        client = get_arcade_client(**kwargs)

    tool_formats = await client.tools.formatted.list(toolkit=toolkits, format="openai")
    auth_spec = await _get_arcade_tool_definitions(client, toolkits, tools)

    tool_functions = []
    for tool in tool_formats.items:
        tool_name = tool["function"]["name"]
        tool_description = tool["function"]["description"]
        tool_params = tool["function"]["parameters"]
        requires_auth = auth_spec.get(tool_name, False)
        tool_function = FunctionTool(
            name=tool_name,
            description=tool_description,
            params_json_schema=tool_params,
            on_invoke_tool=partial(
                _async_invoke_arcade_tool,
                tool_name=tool_name,
                requires_auth=requires_auth,
                client=client,
            ),
            strict_json_schema=False,
        )
        tool_functions.append(tool_function)

    return tool_functions
