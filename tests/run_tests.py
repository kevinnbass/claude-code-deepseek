#!/usr/bin/env python3
"""
Main test runner for Claude-on-Deepseek/Gemini Proxy.

This script provides a CLI for running various tests against the proxy server
and comparing performance with the Anthropic API.
"""

import os
import sys
import asyncio
import argparse
from dotenv import load_dotenv
from .basic_tests import run_basic_tests
from .utils import is_server_running

async def main():
    """Main entry point."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run tests for Claude-on-Deepseek/Gemini Proxy")
    parser.add_argument("--basic", action="store_true", help="Run basic functionality tests")
    parser.add_argument("--performance", action="store_true", help="Run performance comparison tests")
    parser.add_argument("--key-test", action="store_true", help="Test Anthropic API key")
    parser.add_argument("--proxy-only", action="store_true", help="Only test the proxy (skip Anthropic API)")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--anthropic-key", help="Override Anthropic API key")
    args = parser.parse_args()
    
    # If no specific tests are requested, run all
    if not (args.basic or args.performance or args.key_test):
        args.all = True
    
    # Override API key if provided
    if args.anthropic_key:
        os.environ["ANTHROPIC_API_KEY"] = args.anthropic_key
    
    # Check if server is running (needed for most tests)
    if not is_server_running() and not args.key_test:
        print("Error: Proxy server is not running. Please start it first.")
        print("Run: python server.py --always-cot")
        return 1
    
    # Run selected tests
    results = {}
    
    # Basic functionality tests
    if args.all or args.basic:
        print("\n=== RUNNING BASIC FUNCTIONALITY TESTS ===")
        results["basic"] = run_basic_tests(args.proxy_only)
    
    # Performance comparison tests
    if args.all or args.performance:
        print("\n=== RUNNING PERFORMANCE COMPARISON TESTS ===")
        # Import inside the function to avoid circular imports
        from .performance_tests import run_comparison_tests, print_comparison_table, save_results_to_file
        
        comparison_results = await run_comparison_tests(iterations=1, proxy_only=args.proxy_only)
        print_comparison_table(comparison_results)
        save_results_to_file(comparison_results, filename="comparison_results.json")
        
        # Record if all tests passed
        results["performance"] = all(r.status for r in comparison_results.results)
    
    # API key test
    if args.all or args.key_test:
        print("\n=== TESTING ANTHROPIC API KEY ===")
        if os.environ.get("ANTHROPIC_API_KEY"):
            from .api_key_test import test_key
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            results["key_test"] = await test_key(api_key)
        else:
            print("No Anthropic API key found in environment")
            results["key_test"] = False
    
    # Print overall summary
    print("\n=== OVERALL TEST SUMMARY ===")
    for test_name, result in results.items():
        print(f"{test_name.upper()}: {'✅ PASS' if result else '❌ FAIL'}")
    
    # Return success if all tests passed
    return 0 if all(results.values()) else 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))