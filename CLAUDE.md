# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a proxy server that enables Claude Code to work with GLM models while maintaining the same interface. The proxy intercepts Anthropic API requests, transforms them for GLM provider, and converts responses back to Anthropic format.

## Common Development Commands

### Running the Server
```bash
# Start the proxy server (recommended)
python server.py --always-cot
# OR with UV
uv run server.py --always-cot

# Basic server without chain-of-thought
python server.py
```

### Running Tests
```bash
# Run all tests
python run_tests.py --all

# Run specific test suites
python run_tests.py --basic              # Basic functionality tests
python run_tests.py --performance        # Performance comparison tests
python run_tests.py --key-test          # API key validation test

# Test proxy only (skip Anthropic API comparisons)
python run_tests.py --proxy-only
```

### Dependencies Management
```bash
# Install dependencies
pip install -e .
# OR with UV (recommended)
uv pip install -e .
```

## Code Architecture

### Core Components

**server.py** - Main FastAPI application with three key layers:
- **API Compatibility Layer**: Implements Anthropic's `/v1/messages` and `/v1/messages/count_tokens` endpoints
- **Model Routing Logic**: Routes requests to Deepseek (Sonnet) or Gemini (Haiku) based on model name
- **Request/Response Transformation**: Converts between Anthropic and OpenAI/LiteLLM formats

**Key Architecture Patterns**:
- **Model Mapping**: All Claude models → GLM-4.5
- **Chain-of-Thought Enhancement**: Optional CoT system prompt injection for Sonnet models (`--always-cot`)
- **Content Block Transformation**: Converts Anthropic's content blocks to OpenAI-style messages
- **Tool Call Handling**: Translates function calling between API formats

### Request Flow
1. Claude Code sends Anthropic-format request
2. `MessagesRequest.validate_model()` performs model mapping and CoT injection
3. `convert_anthropic_to_litellm()` transforms request format
4. LiteLLM routes to GLM provider with custom API base URL
5. `convert_litellm_to_anthropic()` transforms response back
6. Streaming handled by `handle_streaming()` with proper event formatting

### Special Features

**Custom Commands**:
- `/brainstorm` - Uses GLM-4.5 with specialized brainstorming system prompt
- Future planned: `/debug`, `/refactor`, `/perf`

**Environment Configuration**:
- `BIG_MODEL` - Model for Sonnet requests (default: glm-4.5)
- `SMALL_MODEL` - Model for Haiku requests (default: glm-4.5)
- `DEBUG` - Enable debug logging
- `ALWAYS_COT` - Command line flag for chain-of-thought mode

## Testing Structure

Tests are organized in the `tests/` directory:
- **basic_tests.py** - Core functionality verification
- **performance_tests.py** - Speed and quality comparisons
- **api_key_test.py** - API key validation
- **utils.py** - Shared testing utilities

## Key Implementation Notes

- **Token Limits**: GLM-4.5 supports up to 98,304 tokens output (much higher than previous models)
- **Content Block Processing**: GLM supports tool_use blocks similar to Claude models
- **Streaming Compatibility**: Full Server-Sent Events support matching Anthropic's format
- **Error Handling**: Comprehensive error logging with LiteLLM exception details
- **Model Capabilities**: GLM models support native tool_use blocks and function calling

## Environment Setup

Required API keys in `.env`:
- `GLM_API_KEY` - For all model requests including `/brainstorm` (get from https://open.bigmodel.cn)
- `ANTHROPIC_API_KEY` - Optional, not used in current configuration

## Development Tips

- Use `--always-cot` flag for better reasoning on complex tasks
- Monitor logs for model mapping confirmations (📌 MODEL MAPPING logs)
- Test both streaming and non-streaming modes
- Verify tool calling works correctly with different providers
- Check token count accuracy with `/v1/messages/count_tokens` endpoint
- GLM API uses OpenAI-compatible format with custom base URL: https://open.bigmodel.cn/api/paas/v4