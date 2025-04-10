# Agents Arcade Documentation

This directory contains the documentation for the Agents Arcade project, built with [MkDocs](https://www.mkdocs.org/) and the [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme.

## Documentation Structure

-   **api/**: API reference documentation for each module
-   **examples/**: Detailed example walkthroughs
-   **getting-started/**: Installation and quickstart guides
-   **guides/**: More in-depth guides on specific topics

## Building the Documentation

To build the documentation locally:

```bash
# Install development dependencies if you haven't already
poetry install --with dev

# Build the docs
mkdocs build

# Or serve locally with hot-reloading
mkdocs serve
```

Then visit `http://localhost:8000` in your browser.

Alternatively, you can use the provided build script:

```bash
./build_docs.sh       # Just build
./build_docs.sh --serve   # Build and serve locally
```

## Contributing to the Documentation

When contributing to the documentation, please follow these guidelines:

1. **Use Markdown**: All documentation is written in Markdown
2. **Follow the structure**: Place new files in the appropriate directories
3. **Include code examples**: Provide examples for all features when possible
4. **Check links**: Ensure all links to other pages or sections work
5. **Run local builds**: Always check your changes locally before submitting

## Automatic Deployment

The documentation is automatically built and deployed when changes are pushed to the main branch. This is handled by the GitHub Actions workflow in `.github/workflows/docs.yml`.

## Documentation Style Guide

-   Use title case for headings (e.g., "Getting Started with Agents Arcade")
-   Use backticks for code, module names, functions, and parameters (e.g., `get_arcade_tools()`)
-   Include type annotations in function signatures
-   Write in clear, concise language
-   Use admonitions for notes, warnings, and tips:

```markdown
!!! note
This is a note admonition.

!!! warning
This is a warning admonition.

!!! tip
This is a tip admonition.
```
