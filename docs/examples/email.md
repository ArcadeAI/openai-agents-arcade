# Email Agent Example

This example demonstrates how to create an agent that can access and manage Gmail using Arcade tools.

## Prerequisites

Before running this example, make sure you have:

1. Installed `agents-arcade` (see [Installation](installation.md))
2. An Arcade API key (sign up at [arcade.dev](https://arcade.dev) if you don't have one)
3. An [OpenAI API Key](https://platform.openai.com/docs/libraries#create-and-export-an-api-key).
4. Set the `ARCADE_API_KEY` environment variable or have your API key ready to use
5. Set the `OPENAI_API_KEY` environment variable or have your API key ready to use

## Basic Email Agent

Here's a complete example of an email agent that can interact with Gmail:

```python
from agents import Agent, Runner
from arcadepy import AsyncArcade

from agents_arcade import get_arcade_tools
from agents_arcade.errors import AuthorizationError


async def main():
    # Create an Arcade client
    client = AsyncArcade()

    # Get Google tools from Arcade (includes Gmail)
    tools = await get_arcade_tools(client, toolkits=["google"])

    # Create an OpenAI agent with Google tools
    google_agent = Agent(
        name="Google agent",
        instructions="You are a helpful assistant that can assist with Gmail and other Google services.",
        model="gpt-4o-mini",
        tools=tools,
    )

    try:
        # Run the agent with a specific task
        result = await Runner.run(
            starting_agent=google_agent,
            input="What are my latest emails?",
            # A unique user_id is required for Google authorization
            context={"user_id": "user@example.com"},
        )
        print("Final output:\n\n", result.final_output)
    except AuthorizationError as e:
        # Handle authorization errors
        print("Please Login to Google:", e)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
```

Save this as `email_agent.py` and run it with `python email_agent.py`.

## What This Example Does

1. Creates an Arcade client using your API key
2. Fetches Google tools from Arcade (which include Gmail tools)
3. Creates an OpenAI agent with access to those tools
4. Runs the agent with a specific task ("What are my latest emails?")
5. Handles any authorization errors that might occur

## Authentication Process

If this is your first time using the Google toolkit with your user ID, you will see a message like:

```
Please Login to Google: Authorization required: https://accounts.google.com/o/oauth2/auth...
```

Open this URL in your browser and complete the Google authorization process. After authorizing, you can run the script again, and it should work without requiring authentication.

## Common Email Tasks

Your email agent can perform a wide range of Gmail-related tasks. Here are some example prompts you can try:

-   "What are my latest emails?"
-   "Send an email to example@example.com with the subject 'Hello' and body 'How are you?'"
-   "Show me unread emails from the last week"
-   "Find emails with attachments"
-   "Summarize emails from a specific sender"
-   "Draft a reply to my most recent email"

## Available Google Tools

The Google toolkit includes tools for:

-   Gmail (read, send, search, draft)
-   Google Drive (list, create, read files)
-   Google Calendar (list, create, update events)
-   Google Docs (create, read, update)
-   Google Sheets (read, update)
-   And more

## Advanced Usage

### Email-Only Agent

If you only want to include Gmail-related tools and not other Google services:

```python
tools = await get_arcade_tools(
    client,
    tools=[
        "google_gmail_get_messages",
        "google_gmail_send_message",
        "google_gmail_search_messages"
    ]
)
```

### Email Assistant with Specific Instructions

For a more specialized email assistant:

```python
email_agent = Agent(
    name="Email Assistant",
    instructions="""You are a specialized email assistant.
    You can help read, summarize, and compose emails.
    When asked to send an email, always confirm the recipient, subject, and content.
    When summarizing emails, focus on the most important information.
    Be concise and professional in your responses.""",
    model="gpt-4o-mini",
    tools=tools,
)
```

### Combining with Other Toolkits

You can combine Google tools with other Arcade toolkits for a more versatile agent:

```python
tools = await get_arcade_tools(client, toolkits=["google", "github", "linkedin"])
```

This creates an agent that can handle emails, GitHub tasks, and LinkedIn interactions in a single conversation.
