# Copyright (c) Meta Platforms, Inc. and affiliates.
# BSD-3-Clause License
"""
AgentSafetyEnv server/app.py — OpenEnv validator entry point.

This file satisfies the OpenEnv multi-mode deployment check:
  - server/app.py must exist at repo root
  - main() must be callable with no required args

The actual application is in agent_safety_env/server/app.py.
This module re-exports everything from there.
"""

import sys
import os

# Add parent directory to path so agent_safety_env package is importable
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Re-export the full app from the package
from agent_safety_env.server.app import app, main  # noqa: F401

if __name__ == "__main__":
    main()
