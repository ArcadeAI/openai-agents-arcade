# Advanced & Custom Usage

This guide covers advanced usage patterns and customizations for the `agents-arcade` library, helping you build more sophisticated agent applications with Arcade tools.

## Custom Client Configuration

You can configure the Arcade client with custom settings:

```python
from arcadepy import AsyncArcade
from agents_arcade import get_arcade_tools

# Custom base URL (e.g., for staging environment)
client = AsyncArcade(
    base_url="https://api.staging.arcade.dev",
    api_key="your_api_key",
    timeout=30  # Custom timeout in seconds
)

# Get tools using the custom client
tools = await get_arcade_tools(client, ["github"])
```

Alternatively, you can pass configuration directly to `get_arcade_tools`:

```python
tools = await get_arcade_tools(
    client=None,  # No client provided, will create one with the kwargs
    toolkits=["github"],
    base_url="https://api.staging.arcade.dev",
    api_key="your_api_key",
    timeout=30
)
```

## Building Multi-Agent Systems

You can create multiple specialized agents with different tools:

```python
from agents import Agent, Runner, Message

# Create specialized agents
github_tools = await get_arcade_tools(client, ["github"])
github_agent = Agent(
    name="GitHub Specialist",
    instructions="You are a GitHub expert. Help users with GitHub tasks.",
    model="gpt-4o-mini",
    tools=github_tools,
)

google_tools = await get_arcade_tools(client, ["google"])
google_agent = Agent(
    name="Google Workspace Specialist",
    instructions="You are a Google Workspace expert. Help users with Gmail, Drive, etc.",
    model="gpt-4o-mini",
    tools=google_tools,
)

# Create a coordinator agent
coordinator = Agent(
    name="Coordinator",
    instructions="""You are a coordination agent. Based on the user's request,
    determine whether to route to the GitHub Specialist or Google Workspace Specialist.
    For GitHub tasks: {{ to: "GitHub Specialist", task: "specific github task" }}
    For Google tasks: {{ to: "Google Workspace Specialist", task: "specific google task" }}
    """,
    model="gpt-4o-mini",
)

# Set up the agent network
coordinator.add_tool_agent(github_agent)
coordinator.add_tool_agent(google_agent)

# Run with a single context for authentication
result = await Runner.run(
    starting_agent=coordinator,
    input="Create a GitHub repository and then send an email about it",
    context={"user_id": "user@example.com"},
)
```

## Customizing Tool Behavior

You can create custom tool wrappers to add functionality:

```python
from agents.tool import FunctionTool
from functools import partial

# Get the original tools
original_tools = await get_arcade_tools(client, ["github"])

# Create enhanced versions with logging and error handling
enhanced_tools = []
for tool in original_tools:
    # Create a wrapped version of the tool's invoke function
    async def wrapped_invoke(context, tool_args, original_invoke):
        print(f"[LOG] Executing tool: {tool.name} with args: {tool_args}")
        try:
            result = await original_invoke(context, tool_args)
            print(f"[LOG] Tool {tool.name} completed successfully")
            return result
        except Exception as e:
            print(f"[ERROR] Tool {tool.name} failed: {e}")
            raise

    # Create a new tool with the wrapped invoke function
    enhanced_tool = FunctionTool(
        name=tool.name,
        description=tool.description,
        params_json_schema=tool.params_json_schema,
        on_invoke_tool=partial(wrapped_invoke, original_invoke=tool.on_invoke_tool),
        strict_json_schema=tool.strict_json_schema,
    )
    enhanced_tools.append(enhanced_tool)

# Use the enhanced tools
agent = Agent(
    name="GitHub agent",
    instructions="You are a helpful assistant that can assist with GitHub API calls.",
    model="gpt-4o-mini",
    tools=enhanced_tools,
)
```

## Building Web Applications

Here's an example of integrating `agents-arcade` with a FastAPI web application:

```python
from fastapi import FastAPI, BackgroundTasks, HTTPException, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import Dict, List, Optional
import asyncio
import uuid

from agents import Agent, Runner
from arcadepy import AsyncArcade
from agents_arcade import get_arcade_tools
from agents_arcade.errors import AuthorizationError, ToolError

app = FastAPI()
api_key_header = APIKeyHeader(name="X-API-Key")

# Store active jobs
active_jobs = {}

# Models
class AgentRequest(BaseModel):
    user_id: str
    input: str
    toolkits: List[str]
    model: str = "gpt-4o-mini"

class AgentResponse(BaseModel):
    job_id: str
    status: str
    output: Optional[str] = None
    authorization_url: Optional[str] = None

# Authentication dependency
def get_api_key(api_key: str = Depends(api_key_header)):
    if api_key != "your-api-key":  # Replace with actual API key validation
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

# Create an Arcade client
arcade_client = AsyncArcade()

@app.post("/agent/run", response_model=AgentResponse)
async def run_agent(request: AgentRequest, api_key: str = Depends(get_api_key), background_tasks: BackgroundTasks = None):
    # Generate a job ID
    job_id = str(uuid.uuid4())

    # Store initial job status
    active_jobs[job_id] = {
        "status": "pending",
        "output": None,
        "authorization_url": None
    }

    # Start the agent task in the background
    background_tasks.add_task(process_agent_request, job_id, request)

    return AgentResponse(
        job_id=job_id,
        status="pending"
    )

async def process_agent_request(job_id: str, request: AgentRequest):
    try:
        # Get the requested tools
        tools = await get_arcade_tools(arcade_client, request.toolkits)

        # Create the agent
        agent = Agent(
            name="API Agent",
            instructions="You are a helpful assistant with access to various tools.",
            model=request.model,
            tools=tools,
        )

        # Run the agent
        result = await Runner.run(
            starting_agent=agent,
            input=request.input,
            context={"user_id": request.user_id},
        )

        # Update job status
        active_jobs[job_id] = {
            "status": "completed",
            "output": result.final_output,
            "authorization_url": None
        }

    except AuthorizationError as e:
        # Update job status with authorization URL
        active_jobs[job_id] = {
            "status": "authorization_required",
            "output": None,
            "authorization_url": e.result.url
        }

    except ToolError as e:
        # Update job status with error
        active_jobs[job_id] = {
            "status": "error",
            "output": f"Tool error: {e.message}",
            "authorization_url": None
        }

    except Exception as e:
        # Update job status with error
        active_jobs[job_id] = {
            "status": "error",
            "output": f"Error: {str(e)}",
            "authorization_url": None
        }

@app.get("/agent/status/{job_id}", response_model=AgentResponse)
async def get_job_status(job_id: str, api_key: str = Depends(get_api_key)):
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = active_jobs[job_id]
    return AgentResponse(
        job_id=job_id,
        status=job["status"],
        output=job["output"],
        authorization_url=job["authorization_url"]
    )
```

## Caching Tool Results

For performance optimization, you might want to cache tool results:

```python
import functools
import asyncio
import json

# Simple in-memory cache
cache = {}
cache_ttl = 300  # 5 minutes

# Create a caching decorator for tool calls
def cache_tool_result(ttl=cache_ttl):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(context, tool_args, *args, **kwargs):
            # Create a cache key from the tool name and arguments
            tool_name = kwargs.get("tool_name", "unknown_tool")
            cache_key = f"{tool_name}:{tool_args}"

            # Check if result is in cache and not expired
            if cache_key in cache:
                timestamp, result = cache[cache_key]
                if (asyncio.get_event_loop().time() - timestamp) < ttl:
                    print(f"Cache hit for {tool_name}")
                    return result

            # Execute the tool and cache the result
            result = await func(context, tool_args, *args, **kwargs)
            cache[cache_key] = (asyncio.get_event_loop().time(), result)
            return result
        return wrapper
    return decorator

# Apply the cache decorator to _async_invoke_arcade_tool
from agents_arcade.tools import _async_invoke_arcade_tool

# Cached version
@cache_tool_result(ttl=300)
async def cached_invoke_arcade_tool(context, tool_args, tool_name, requires_auth, client):
    return await _async_invoke_arcade_tool(
        context=context,
        tool_args=tool_args,
        tool_name=tool_name,
        requires_auth=requires_auth,
        client=client
    )

# Replace the original function with the cached version in your tools
# Note: This requires modifying the tool construction process
```

## Persistent User Storage

For applications with many users, you might want to store user authentication state:

```python
import sqlite3
import json
from contextlib import contextmanager

# Simple SQLite-based user storage
class UserStore:
    def __init__(self, db_path="users.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                auth_state TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            conn.commit()

    @contextmanager
    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()

    def get_user(self, user_id):
        with self._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT user_id, auth_state FROM users WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()
            if row:
                return {
                    "user_id": row[0],
                    "auth_state": json.loads(row[1]) if row[1] else {}
                }
            return None

    def create_or_update_user(self, user_id, auth_state=None):
        with self._get_conn() as conn:
            cursor = conn.cursor()
            existing = self.get_user(user_id)

            if existing:
                cursor.execute(
                    "UPDATE users SET auth_state = ?, updated_at = CURRENT_TIMESTAMP WHERE user_id = ?",
                    (json.dumps(auth_state), user_id)
                )
            else:
                cursor.execute(
                    "INSERT INTO users (user_id, auth_state) VALUES (?, ?)",
                    (user_id, json.dumps(auth_state) if auth_state else None)
                )
            conn.commit()

    def record_authorization(self, user_id, toolkit, status):
        user = self.get_user(user_id)
        auth_state = user["auth_state"] if user else {}
        auth_state[toolkit] = {
            "status": status,
            "timestamp": datetime.now().isoformat()
        }
        self.create_or_update_user(user_id, auth_state)

# Example usage
user_store = UserStore()

# Before running agent
user = user_store.get_user("user@example.com")
if not user:
    user_store.create_or_update_user("user@example.com")

# After successful authorization
user_store.record_authorization("user@example.com", "github", "authorized")
```

## Conclusion

These advanced techniques should help you build more sophisticated applications with `agents-arcade`. You can combine these patterns to create powerful, robust, and user-friendly agent systems that leverage Arcade's extensive toolkit ecosystem.
