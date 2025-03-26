#!/usr/bin/env python3
"""
Test utilities for Claude-on-Deepseek/Gemini Proxy.

This module provides common utilities for testing the proxy server
including shared functions for making API requests and handling responses.
"""

import os
import json
import time
import httpx
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional, Set, Tuple
from dotenv import load_dotenv
from . import ANTHROPIC_API_URL, PROXY_API_URL, ANTHROPIC_VERSION

# Load environment variables
load_dotenv()

# Configuration
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# Headers
def get_anthropic_headers():
    """Get headers for Anthropic API requests."""
    return {
        "x-api-key": ANTHROPIC_API_KEY if ANTHROPIC_API_KEY else "invalid-key",
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }

def get_proxy_headers():
    """Get headers for proxy API requests."""
    return {
        "content-type": "application/json",
        "anthropic-version": ANTHROPIC_VERSION,
    }

def is_server_running():
    """Check if the proxy server is running."""
    try:
        response = httpx.get(PROXY_API_URL.replace("/messages", ""), timeout=2)
        return True
    except:
        return False

def get_response(url, headers, data):
    """Send a request and get the response."""
    start_time = time.time()
    # Use a longer timeout for Sonnet/reasoning tests
    timeout_val = 60 if "sonnet" in data.get("model", "").lower() else 30
    response = httpx.post(url, headers=headers, json=data, timeout=timeout_val)
    elapsed = time.time() - start_time
    
    return {
        "response": response,
        "elapsed": elapsed,
        "success": response.status_code == 200
    }

class StreamStats:
    """Track statistics about a streaming response."""
    
    def __init__(self):
        self.event_types = set()
        self.event_counts = {}
        self.first_event_time = None
        self.last_event_time = None
        self.total_chunks = 0
        self.events = []
        self.text_content = ""
        self.content_blocks = {}
        self.has_tool_use = False
        self.has_error = False
        self.error_message = ""
        self.text_content_by_block = {}
        
    def add_event(self, event_data):
        """Track information about each received event."""
        now = datetime.now()
        if self.first_event_time is None:
            self.first_event_time = now
        self.last_event_time = now
        
        self.total_chunks += 1
        
        # Record event type and increment count
        if "type" in event_data:
            event_type = event_data["type"]
            self.event_types.add(event_type)
            self.event_counts[event_type] = self.event_counts.get(event_type, 0) + 1
            
            # Track specific event data
            if event_type == "content_block_start":
                block_idx = event_data.get("index")
                content_block = event_data.get("content_block", {})
                if content_block.get("type") == "tool_use":
                    self.has_tool_use = True
                self.content_blocks[block_idx] = content_block
                self.text_content_by_block[block_idx] = ""
                
            elif event_type == "content_block_delta":
                block_idx = event_data.get("index")
                delta = event_data.get("delta", {})
                if delta.get("type") == "text_delta":
                    text = delta.get("text", "")
                    self.text_content += text
                    # Also track text by block ID
                    if block_idx in self.text_content_by_block:
                        self.text_content_by_block[block_idx] += text
                        
        # Keep track of all events for debugging
        self.events.append(event_data)
                
    def get_duration(self):
        """Calculate the total duration of the stream in seconds."""
        if self.first_event_time is None or self.last_event_time is None:
            return 0
        return (self.last_event_time - self.first_event_time).total_seconds()
        
    def summarize(self):
        """Print a summary of the stream statistics."""
        print(f"Total chunks: {self.total_chunks}")
        print(f"Unique event types: {sorted(list(self.event_types))}")
        print(f"Event counts: {json.dumps(self.event_counts, indent=2)}")
        print(f"Duration: {self.get_duration():.2f} seconds")
        print(f"Has tool use: {self.has_tool_use}")
        
        # Print the first few lines of content
        if self.text_content:
            max_preview_lines = 5
            text_preview = "\n".join(self.text_content.strip().split("\n")[:max_preview_lines])
            print(f"Text preview:\n{text_preview}")
        else:
            print("No text content extracted")
            
        if self.has_error:
            print(f"Error: {self.error_message}")

async def stream_response(url, headers, data, stream_name):
    """Send a streaming request and process the response."""
    print(f"\nStarting {stream_name} stream...")
    stats = StreamStats()
    error = None
    
    try:
        async with httpx.AsyncClient() as client:
            # Add stream flag to ensure it's streamed
            request_data = data.copy()
            request_data["stream"] = True
            
            start_time = time.time()
            # Use a longer timeout for Sonnet/reasoning tests
            timeout_val = 60 if "sonnet" in request_data.get("model", "").lower() else 30
            async with client.stream("POST", url, json=request_data, headers=headers, timeout=timeout_val) as response:
                if response.status_code != 200:
                    error_text = await response.aread()
                    stats.has_error = True
                    stats.error_message = f"HTTP {response.status_code}: {error_text.decode('utf-8')}"
                    error = stats.error_message
                    print(f"Error: {stats.error_message}")
                    return stats, error
                
                print(f"{stream_name} connected, receiving events...")
                
                # Process each chunk
                buffer = ""
                async for chunk in response.aiter_text():
                    if not chunk.strip():
                        continue
                    
                    # Handle multiple events in one chunk
                    buffer += chunk
                    events = buffer.split("\n\n")
                    
                    # Process all complete events
                    for event_text in events[:-1]:  # All but the last (possibly incomplete) event
                        if not event_text.strip():
                            continue
                        
                        # Parse server-sent event format
                        if "data: " in event_text:
                            # Extract the data part
                            data_parts = []
                            for line in event_text.split("\n"):
                                if line.startswith("data: "):
                                    data_part = line[len("data: "):]
                                    # Skip the "[DONE]" marker
                                    if data_part == "[DONE]":
                                        break
                                    data_parts.append(data_part)
                            
                            if data_parts:
                                try:
                                    event_data = json.loads("".join(data_parts))
                                    stats.add_event(event_data)
                                except json.JSONDecodeError as e:
                                    print(f"Error parsing event: {e}\nRaw data: {''.join(data_parts)}")
                    
                    # Keep the last (potentially incomplete) event for the next iteration
                    buffer = events[-1] if events else ""
                    
                # Process any remaining complete events in the buffer
                if buffer.strip():
                    lines = buffer.strip().split("\n")
                    data_lines = [line[len("data: "):] for line in lines if line.startswith("data: ")]
                    if data_lines and data_lines[0] != "[DONE]":
                        try:
                            event_data = json.loads("".join(data_lines))
                            stats.add_event(event_data)
                        except:
                            pass
                
            elapsed = time.time() - start_time
            print(f"{stream_name} stream completed in {elapsed:.2f} seconds")
    except Exception as e:
        stats.has_error = True
        stats.error_message = str(e)
        error = str(e)
        print(f"Error in {stream_name} stream: {e}")
    
    return stats, error

def extract_text_from_response(response_json):
    """Extract text content from a response JSON."""
    if not response_json or "content" not in response_json:
        return ""
        
    content = response_json["content"]
    if not isinstance(content, list):
        return str(content)
        
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            return item.get("text", "")
    
    return ""
    
def extract_tool_use_from_response(response_json):
    """Extract tool use from a response JSON."""
    if not response_json or "content" not in response_json:
        return None
        
    content = response_json["content"]
    if not isinstance(content, list):
        return None
        
    for item in content:
        if isinstance(item, dict) and item.get("type") == "tool_use":
            return item
    
    return None