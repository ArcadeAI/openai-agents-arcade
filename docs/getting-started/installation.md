# Installation

This guide covers how to install the `agents-arcade` package and its dependencies.

## Requirements

-   Python 3.9 or higher
-   `openai-agents` 0.0.4 or higher
-   `arcadepy` 1.3.0 or higher

## Installation Options

### pip

The recommended way to install `agents-arcade` is using pip:

```bash
pip install agents-arcade
```

### From Source

You can also install the latest version directly from the GitHub repository:

```bash
git clone https://github.com/ArcadeAI/agents-arcade.git
cd agents-arcade
pip install -e .
```

## Development Installation

If you're contributing to `agents-arcade` or need the development dependencies:

```bash
git clone https://github.com/ArcadeAI/agents-arcade.git
cd agents-arcade
pip install -e ".[dev]"
```

## Verify Installation

You can verify that the installation was successful by importing the package in Python:

```python
from agents_arcade import get_arcade_tools
print("Installation successful!")
```

## Next Steps

Now that you have `agents-arcade` installed, you can:

-   Check out the [Quickstart Guide](quickstart.md) to create your first agent
-   Learn about [Authentication](authentication.md) for Arcade tools
-   Explore the available [toolkits](../guides/toolkits.md)
