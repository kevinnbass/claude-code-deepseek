#!/usr/bin/env python3
"""
Test Anthropic API key validity.

This script checks if the provided Anthropic API key works
by making a simple request to the Anthropic API.
"""

import os
import json
import sys
import asyncio
import argparse
import httpx
from dotenv import load_dotenv
from . import ANTHROPIC_API_URL

# Load environment variables
load_dotenv()

ANTHROPIC_VERSION = "2023-06-01"

async def test_key(api_key):
    """Test if an Anthropic API key works."""
    headers = {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }
    
    request_data = {
        "model": "claude-3-haiku-20240307",
        "max_tokens": 50,
        "messages": [
            {"role": "user", "content": "Say hello world"}
        ]
    }
    
    print(f"\nTesting Anthropic API key: {api_key[:10]}...{api_key[-6:]}")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                ANTHROPIC_API_URL,
                json=request_data,
                headers=headers,
                timeout=10
            )
            
        if response.status_code == 200:
            response_json = response.json()
            print(f"✅ SUCCESS - API key works!")
            print(f"Response: {json.dumps(response_json, indent=2)[:200]}...")
            return True
        else:
            print(f"❌ FAILED - Status code: {response.status_code}")
            print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False

async def main():
    """Test multiple API keys."""
    parser = argparse.ArgumentParser(description="Test Anthropic API keys")
    parser.add_argument("keys", nargs="*", help="API key(s) to test")
    parser.add_argument("--env", action="store_true", help="Test API key from ANTHROPIC_API_KEY environment variable")
    args = parser.parse_args()
    
    keys = args.keys[:]
    
    # Add environment key if requested
    if args.env:
        env_key = os.environ.get("ANTHROPIC_API_KEY")
        if env_key:
            keys.append(env_key)
            print(f"Added API key from environment variable")
        else:
            print("No API key found in ANTHROPIC_API_KEY environment variable")
    
    if not keys:
        print("Error: No API keys provided")
        print("Usage: python -m tests.api_key_test API_KEY1 [API_KEY2 ...] [--env]")
        return 1
    
    results = []
    
    for key in keys:
        result = await test_key(key)
        results.append(result)
    
    if any(results):
        print("\n✅ At least one API key works!")
        for i, result in enumerate(results):
            if result:
                print(f"Working key: {keys[i][:10]}...{keys[i][-6:]}")
    else:
        print("\n❌ None of the API keys work.")
    
    return 0 if any(results) else 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))