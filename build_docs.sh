#!/bin/bash
set -e  # Exit on first error

# Check if virtual environment exists, if not create one
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python -m venv .venv
fi

# Activate virtual environment
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source .venv/Scripts/activate
else
    source .venv/bin/activate
fi

# Install dependencies
echo "Installing dependencies..."
pip install -e ".[dev]"

# Build the documentation
echo "Building documentation..."
mkdocs build

# Optional: Serve the documentation locally
if [[ "$1" == "--serve" ]]; then
    echo "Serving documentation locally at http://localhost:8000"
    mkdocs serve
fi

# Optional: Deploy to GitHub Pages
if [[ "$1" == "--deploy" ]]; then
    echo "Deploying documentation to GitHub Pages..."
    mkdocs gh-deploy --force
fi

echo "Documentation build complete! Find the built site in the 'site' directory."