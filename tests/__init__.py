# Test package initialization
"""
Tests package for Claude-on-Deepseek/Gemini Proxy.

This package contains all testing utilities for verifying the proxy server
functionality with both Deepseek and Gemini models.
"""

from typing import Dict, Any

# Common test requests that can be reused across test files
COMMON_TEST_REQUESTS = {
    # Simple text response
    "simple_text": {
        "description": "Simple text response (Haiku/Gemini)",
        "model": "claude-3-haiku-20240307",
        "max_tokens": 300,
        "messages": [
            {"role": "user", "content": "Hello, what is Paris known for? Please answer in 2-3 sentences."}
        ]
    },
    
    # Tool usage
    "calculator": {
        "description": "Calculator tool usage (Haiku/Gemini)",
        "model": "claude-3-haiku-20240307",
        "max_tokens": 300,
        "messages": [
            {"role": "user", "content": "Calculate 135 + 7.5 divided by 2.5?"}
        ],
        "tools": [{
            "name": "calculator",
            "description": "Evaluate mathematical expressions",
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate"
                    }
                },
                "required": ["expression"]
            }
        }],
        "tool_choice": {"type": "auto"}
    },
    
    # Complex reasoning
    "complex_reasoning": {
        "description": "Complex reasoning (Sonnet/Deepseek)",
        "model": "claude-3-sonnet-20240229",
        "max_tokens": 1000,
        "messages": [
            {"role": "user", "content": "If a train travels at 120 km/h and another train travels at 180 km/h in the opposite direction, and they start 900 km apart, how long will it take for them to meet?"}
        ]
    },
    
    # Code generation
    "code_generation": {
        "description": "Python code generation (Sonnet/Deepseek)",
        "model": "claude-3-sonnet-20240229",
        "max_tokens": 1000,
        "messages": [
            {"role": "user", "content": "Write a Python function to check if a string is a palindrome."}
        ]
    }
}

# Default API endpoints
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
PROXY_API_URL = "http://localhost:8082/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"