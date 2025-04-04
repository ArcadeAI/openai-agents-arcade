# Arcade Toolkits Guide

This guide provides an overview of the available Arcade toolkits that you can use with OpenAI Agents through the `agents-arcade` library.

## Available Toolkits

Arcade offers a variety of toolkits that provide access to different services and APIs:

| Toolkit       | Description                                        | Authentication Required |
| ------------- | -------------------------------------------------- | ----------------------- |
| `github`      | Interact with GitHub repositories, issues, PRs     | Yes                     |
| `google`      | Access Gmail, Drive, Calendar, Docs, etc.          | Yes                     |
| `linkedin`    | Interact with LinkedIn (posts, profile, messaging) | Yes                     |
| `x`           | Interact with X/Twitter (tweets, timeline, etc.)   | Yes                     |
| `web`         | Web search, browser automation                     | Partial                 |
| `news`        | Access news articles and headlines                 | No                      |
| `maps`        | Geolocation, directions, place info                | Partial                 |
| `weather`     | Current weather and forecasts                      | No                      |
| `datasources` | Connect to databases, APIs, and other data sources | Varies                  |

For the most up-to-date list of available toolkits, check the [Arcade Integrations documentation](https://docs.arcade.dev/toolkits).

## Using Toolkits

To use a toolkit with your agent, you need to include it when fetching tools:

```python
from agents_arcade import get_arcade_tools

# Use a single toolkit
tools = await get_arcade_tools(client, ["github"])

# Use multiple toolkits
tools = await get_arcade_tools(client, ["github", "google", "news"])
```

### Filtering Tools Within a Toolkit

If you only need specific tools from a toolkit, you can include only what you need:

```python
# Only get specific GitHub tools
tools = await get_arcade_tools(
    client,
    tools=["github_get_repository", "github_list_user_repositories"]
)
```

## GitHub Toolkit

The GitHub toolkit allows agents to interact with GitHub repositories, issues, pull requests, and more.

### Common GitHub Tools

-   `github_get_repository`: Get repository information
-   `github_list_user_repositories`: List repositories for a user
-   `github_create_repository`: Create a new repository
-   `github_get_issues`: Get issues for a repository
-   `github_create_issue`: Create a new issue
-   `github_star_repository`: Star a repository

### Example GitHub Usage

```python
tools = await get_arcade_tools(client, ["github"])
agent = Agent(
    name="GitHub agent",
    instructions="You are a helpful assistant that can assist with GitHub API calls.",
    model="gpt-4o-mini",
    tools=tools,
)
```

## Google Toolkit

The Google toolkit provides access to Google services like Gmail, Drive, Calendar, Docs, and more.

### Common Google Tools

-   `google_gmail_get_messages`: Get Gmail messages
-   `google_gmail_send_message`: Send an email
-   `google_drive_list_files`: List files in Google Drive
-   `google_calendar_list_events`: List calendar events
-   `google_docs_create_document`: Create a new Google Doc

### Example Google Usage

```python
tools = await get_arcade_tools(client, ["google"])
agent = Agent(
    name="Google Assistant",
    instructions="You are a helpful assistant that can work with Google services.",
    model="gpt-4o-mini",
    tools=tools,
)
```

## LinkedIn Toolkit

The LinkedIn toolkit allows agents to interact with LinkedIn profiles, posts, and messaging.

### Common LinkedIn Tools

-   `linkedin_get_profile`: Get LinkedIn profile information
-   `linkedin_create_post`: Create a LinkedIn post
-   `linkedin_send_message`: Send a LinkedIn message
-   `linkedin_get_feed`: Get the LinkedIn feed

### Example LinkedIn Usage

```python
tools = await get_arcade_tools(client, ["linkedin"])
agent = Agent(
    name="LinkedIn Assistant",
    instructions="You are a helpful assistant for LinkedIn interactions.",
    model="gpt-4o-mini",
    tools=tools,
)
```

## Web Toolkit

The Web toolkit provides tools for searching the web and interacting with web content.

### Common Web Tools

-   `web_search`: Search the web for information
-   `web_browse`: Visit and extract content from a specific URL
-   `web_scrape`: Extract structured data from web pages

### Example Web Usage

```python
tools = await get_arcade_tools(client, ["web"])
agent = Agent(
    name="Web Assistant",
    instructions="You are a helpful assistant for web searches and browsing.",
    model="gpt-4o-mini",
    tools=tools,
)
```

## Combining Toolkits

One of the most powerful features of `agents-arcade` is the ability to combine multiple toolkits in a single agent:

```python
tools = await get_arcade_tools(client, ["github", "google", "web"])
agent = Agent(
    name="Super Assistant",
    instructions="""You are a versatile assistant that can:
    - Help with GitHub repositories and issues
    - Manage emails and Google documents
    - Search the web for information
    Always use the most appropriate tools for the user's request.""",
    model="gpt-4o-mini",
    tools=tools,
)
```

This creates a powerful agent that can handle a wide range of tasks across different services, providing a seamless experience for the user.

## Authentication Considerations

Remember that many toolkits require authentication. When using multiple authenticated toolkits, the user will need to complete the authentication flow for each service the first time they use it. See the [Authentication](../getting-started/authentication.md) guide for more details.
