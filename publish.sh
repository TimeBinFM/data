#!/bin/bash
set -e

# --- Settings ---
PACKAGE_NAME="ts-preprocessing"
USE_TEST_PYPI=true       # true to upload to TestPyPI
USE_PYPI=true            # true to upload to PyPI

# --- Step 1: Clean old builds ---
echo "=== Cleaning old builds ==="
rm -rf dist build *.egg-info

# --- Step 2: Install build and twine ---
echo "=== Installing required tools ==="
pip install --upgrade build twine

# --- Step 3: Build the package ---
echo "=== Building the package ==="
python -m build

# --- Step 4: Upload to TestPyPI ---
if [ "$USE_TEST_PYPI" = true ]; then
    echo "=== Uploading to TestPyPI ==="
    twine upload --repository testpypi dist/*
fi

# --- Step 5: Upload to PyPI ---
if [ "$USE_PYPI" = true ]; then
    echo "=== Uploading to PyPI ==="
    twine upload dist/*
fi

echo "=== ✅ Publishing complete ==="

