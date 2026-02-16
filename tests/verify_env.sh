#!/usr/bin/env bash

# Exit immediately if a pipeline returns a non-zero status
set -e

echo "Running environment verifications..."
echo "------------------------------------"

# 1. Verify Conda Environment
if [[ "$CONDA_DEFAULT_ENV" != "tecpg-dev" ]]; then
    echo "❌ ERROR: Active conda environment is '${CONDA_DEFAULT_ENV:-None}', not 'tecpg-dev'."
    echo "   Action: Run 'conda activate tecpg-dev' and try again."
    exit 1
fi
echo "✅ Conda environment 'tecpg-dev' is active."

# 2. Verify Git Branch
# Ensure the script is being run from inside a git work tree
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "❌ ERROR: Not currently inside a git repository work tree."
    exit 1
fi

CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [[ "$CURRENT_BRANCH" != "dev" ]]; then
    echo "❌ ERROR: Current git branch is '$CURRENT_BRANCH', not 'dev'."
    echo "   Action: Run 'git checkout dev' to switch branches."
    exit 1
fi
echo "✅ Git branch 'dev' is checked out."

# 3. Verify Library Version and Installation Status
# Querying the installed package metadata directly via Python
TECPG_VERSION=$(python3 -c "
try:
    import importlib.metadata
    print(importlib.metadata.version('tecpg'))
except importlib.metadata.PackageNotFoundError:
    print('NOT_INSTALLED')
")

if [[ "$TECPG_VERSION" == "NOT_INSTALLED" ]]; then
    echo "❌ ERROR: The 'tecpg' package is not installed in the active environment."
    echo "   Action: Run 'pip install --editable .' while in the tecpg-dev environment."
    exit 1
fi

echo "✅ 'tecpg' library is installed (Version: $TECPG_VERSION)."
echo "------------------------------------"
echo "All environment checks passed successfully."
