from agents.exceptions import AgentsException
from arcadepy.types.execute_tool_response import ExecuteToolResponse
from arcadepy.types.shared.authorization_response import AuthorizationResponse


class ToolError(AgentsException):
    def __init__(self, result: ExecuteToolResponse):
        self.result = result

    @property
    def message(self):
        return self.result.output.error.message

    def __str__(self):
        return f"Tool {self.result.tool_name} failed with error: {self.message}"


class AuthorizationError(AgentsException):
    def __init__(self, result: AuthorizationResponse):
        self.result = result

    @property
    def message(self):
        return "Authorization required: " + self.result.url

    def __str__(self):
        return self.message
