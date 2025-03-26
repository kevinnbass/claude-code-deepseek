#!/usr/bin/env python3
"""
Main test runner script for Claude-on-Deepseek/Gemini Proxy.

This script provides a simple entry point to run all tests.
"""

import os
import asyncio
import sys

# Ensure directory paths and imports work correctly
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from tests.run_tests import main

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))