# Contributing to Agents Arcade

Thank you for your interest in contributing to Agents Arcade! This document provides guidelines and instructions for contributing to the project.

## Getting Started

### Prerequisites

-   Python 3.9 or higher
-   Git
-   A GitHub account

### Development Setup

1. Fork the repository on GitHub
2. Clone your fork locally:
    ```bash
    git clone https://github.com/your-username/agents-arcade.git
    cd agents-arcade
    ```
3. Create a virtual environment and install dependencies:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .venv\Scripts\activate
    pip install -e ".[dev]"
    ```

## Development Workflow

### Branching Strategy

We follow a simple branching model:

-   `main` - the main development branch
-   feature branches - for new features or significant changes
-   bug fix branches - for bug fixes

When working on a new feature or bug fix, create a new branch from `main`:

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bug-fix
```

### Coding Standards

We use the following tools to maintain code quality:

-   **Ruff** - for code formatting and linting
-   **MyPy** - for type checking

Make sure your code passes all checks before submitting a pull request:

```bash
# Run linter
ruff check .

# Run type checker
mypy agents_arcade

# Run tests
pytest
```

### Commit Messages

Please write clear and descriptive commit messages that explain the changes you've made. We follow these conventions:

-   Use present tense ("Add feature" not "Added feature")
-   First line is a summary (50 chars or less)
-   Reference issues and pull requests where appropriate ("Fix #123")

### Testing

Please add tests for any new features or bug fixes. We use pytest for testing:

```bash
# Run all tests
pytest

# Run specific tests
pytest tests/test_tools.py
```

## Pull Request Process

1. Update the documentation with details of any changes to interfaces, new features, etc.
2. Update the README.md or documentation with details of changes if applicable
3. Make sure all tests pass and the code follows our style guidelines
4. Submit a pull request to the `main` branch
5. Wait for review and address any feedback

## Documentation

We use MkDocs with Material theme for our documentation. To run the documentation locally:

```bash
mkdocs serve
```

Then navigate to `http://localhost:8000` in your browser.

Make sure to update the documentation when adding new features or changing existing ones. The documentation source files are in the `docs` directory.

## Releasing

The release process is handled by the maintainers. If you have questions about releasing, please open an issue.

## Code of Conduct

Please be respectful and considerate of others when contributing to this project. We aim to foster an inclusive and welcoming environment.

## License

By contributing to Agents Arcade, you agree that your contributions will be licensed under the project's MIT license.

## Questions?

If you have any questions or need help with the contribution process, please open an issue or reach out to the maintainers.
