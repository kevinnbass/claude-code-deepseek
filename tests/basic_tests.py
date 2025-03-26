#!/usr/bin/env python3
"""
Basic functionality tests for Claude-on-Deepseek/Gemini Proxy.

This module provides simple tests to verify that the proxy server 
is functioning properly with basic requests.
"""

import os
import json
import sys
import asyncio
import argparse
from dotenv import load_dotenv
from . import COMMON_TEST_REQUESTS, ANTHROPIC_API_URL, PROXY_API_URL
from .utils import (
    get_anthropic_headers, 
    get_proxy_headers, 
    is_server_running, 
    get_response,
    extract_text_from_response,
    extract_tool_use_from_response
)

# Load environment variables
load_dotenv()

def check_api_key():
    """Check if the Anthropic API key is set."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Warning: ANTHROPIC_API_KEY not set in environment")
        return False
    return True

def test_simple_query(proxy_only=False):
    """Test a simple query to verify basic functionality."""
    print("\n=== Testing Simple Query ===")
    
    # Get request data
    request_data = COMMON_TEST_REQUESTS["simple_text"]
    
    # Test proxy
    print("\nTesting Proxy API...")
    proxy_result = get_response(
        PROXY_API_URL, 
        get_proxy_headers(), 
        request_data
    )
    
    if proxy_result["success"]:
        print(f"✅ Success - {proxy_result['elapsed']:.2f}s")
        response_json = proxy_result["response"].json()
        text = extract_text_from_response(response_json)
        print(f"Response: {text[:100]}...")
    else:
        print(f"❌ Failed - {proxy_result['elapsed']:.2f}s")
        print(f"Error: {proxy_result['response'].text}")
        return False
    
    # Test Anthropic (if not proxy_only)
    if not proxy_only and check_api_key():
        print("\nTesting Anthropic API...")
        anthropic_result = get_response(
            ANTHROPIC_API_URL, 
            get_anthropic_headers(), 
            request_data
        )
        
        if anthropic_result["success"]:
            print(f"✅ Success - {anthropic_result['elapsed']:.2f}s")
            response_json = anthropic_result["response"].json()
            text = extract_text_from_response(response_json)
            print(f"Response: {text[:100]}...")
            
            # Compare times
            ratio = proxy_result["elapsed"] / anthropic_result["elapsed"]
            diff = proxy_result["elapsed"] - anthropic_result["elapsed"]
            print(f"\nComparison: Proxy is {ratio:.2f}x {'slower' if ratio > 1 else 'faster'} ({diff:+.2f}s)")
        else:
            print(f"❌ Failed - {anthropic_result['elapsed']:.2f}s")
            print(f"Error: {anthropic_result['response'].text}")
    
    return True

def test_calculator_tool(proxy_only=False):
    """Test calculator tool usage."""
    print("\n=== Testing Calculator Tool Usage ===")
    
    # Get request data
    request_data = COMMON_TEST_REQUESTS["calculator"]
    
    # Test proxy
    print("\nTesting Proxy API...")
    proxy_result = get_response(
        PROXY_API_URL, 
        get_proxy_headers(), 
        request_data
    )
    
    if proxy_result["success"]:
        print(f"✅ Success - {proxy_result['elapsed']:.2f}s")
        response_json = proxy_result["response"].json()
        
        # Check for tool use
        tool_use = extract_tool_use_from_response(response_json)
        if tool_use:
            print(f"Tool Use: {json.dumps(tool_use, indent=2)}")
        else:
            text = extract_text_from_response(response_json)
            print(f"Response (no tool use): {text[:100]}...")
    else:
        print(f"❌ Failed - {proxy_result['elapsed']:.2f}s")
        print(f"Error: {proxy_result['response'].text}")
        return False
    
    # Test Anthropic (if not proxy_only)
    if not proxy_only and check_api_key():
        print("\nTesting Anthropic API...")
        anthropic_result = get_response(
            ANTHROPIC_API_URL, 
            get_anthropic_headers(), 
            request_data
        )
        
        if anthropic_result["success"]:
            print(f"✅ Success - {anthropic_result['elapsed']:.2f}s")
            response_json = anthropic_result["response"].json()
            
            # Check for tool use
            tool_use = extract_tool_use_from_response(response_json)
            if tool_use:
                print(f"Tool Use: {json.dumps(tool_use, indent=2)}")
            else:
                text = extract_text_from_response(response_json)
                print(f"Response (no tool use): {text[:100]}...")
            
            # Compare times
            ratio = proxy_result["elapsed"] / anthropic_result["elapsed"]
            diff = proxy_result["elapsed"] - anthropic_result["elapsed"]
            print(f"\nComparison: Proxy is {ratio:.2f}x {'slower' if ratio > 1 else 'faster'} ({diff:+.2f}s)")
        else:
            print(f"❌ Failed - {anthropic_result['elapsed']:.2f}s")
            print(f"Error: {anthropic_result['response'].text}")
    
    return True

def test_complex_reasoning(proxy_only=False):
    """Test complex reasoning."""
    print("\n=== Testing Complex Reasoning ===")
    
    # Get request data
    request_data = COMMON_TEST_REQUESTS["complex_reasoning"]
    
    # Test proxy
    print("\nTesting Proxy API (this may take 15-20 seconds)...")
    proxy_result = get_response(
        PROXY_API_URL, 
        get_proxy_headers(), 
        request_data
    )
    
    if proxy_result["success"]:
        print(f"✅ Success - {proxy_result['elapsed']:.2f}s")
        response_json = proxy_result["response"].json()
        text = extract_text_from_response(response_json)
        print(f"Response: {text[:200]}...")
    else:
        print(f"❌ Failed - {proxy_result['elapsed']:.2f}s")
        print(f"Error: {proxy_result['response'].text}")
        return False
    
    # Test Anthropic (if not proxy_only)
    if not proxy_only and check_api_key():
        print("\nTesting Anthropic API...")
        anthropic_result = get_response(
            ANTHROPIC_API_URL, 
            get_anthropic_headers(), 
            request_data
        )
        
        if anthropic_result["success"]:
            print(f"✅ Success - {anthropic_result['elapsed']:.2f}s")
            response_json = anthropic_result["response"].json()
            text = extract_text_from_response(response_json)
            print(f"Response: {text[:200]}...")
            
            # Compare times
            ratio = proxy_result["elapsed"] / anthropic_result["elapsed"]
            diff = proxy_result["elapsed"] - anthropic_result["elapsed"]
            print(f"\nComparison: Proxy is {ratio:.2f}x {'slower' if ratio > 1 else 'faster'} ({diff:+.2f}s)")
        else:
            print(f"❌ Failed - {anthropic_result['elapsed']:.2f}s")
            print(f"Error: {anthropic_result['response'].text}")
    
    return True

def run_basic_tests(proxy_only=False):
    """Run all basic tests."""
    if not is_server_running():
        print("Error: Proxy server is not running. Please start it first.")
        return False
    
    results = []
    
    # Run tests
    results.append(("Simple Query", test_simple_query(proxy_only)))
    results.append(("Calculator Tool", test_calculator_tool(proxy_only)))
    results.append(("Complex Reasoning", test_complex_reasoning(proxy_only)))
    
    # Print summary
    print("\n=== TEST SUMMARY ===")
    for name, result in results:
        print(f"{name}: {'✅ PASS' if result else '❌ FAIL'}")
    
    # Return True if all tests passed
    return all(result for _, result in results)

def main():
    parser = argparse.ArgumentParser(description="Run basic tests for Claude-on-Deepseek Proxy")
    parser.add_argument("--proxy-only", action="store_true", help="Only test the proxy (skip Anthropic API)")
    args = parser.parse_args()
    
    success = run_basic_tests(args.proxy_only)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())