# Claude to Alternative Models Capability Mapping

This document provides a reference for how Claude models and capabilities map to their Deepseek and Gemini equivalents when using the proxy.

## Model Mapping

| Claude Capability | Alternative Equivalent | Provider | Notes |
|-------------------|------------------------|----------|-------|
| **Core Models** | | | |
| Claude Haiku | gemini-2.0-flash | Google | Default for simpler tasks and quick responses |
| Claude Sonnet | deepseek-chat | Deepseek | Default for complex tasks and code generation |
| Claude Opus | Not directly mapped | - | No direct equivalent in current lineup |
| **Specialized Tasks** | | | |
| Code generation | deepseek-chat | Deepseek | Recommended for all coding tasks |
| Reasoning | deepseek-chat + CoT prompt | Deepseek | Use with Chain-of-Thought system prompt for complex reasoning |
| Quick answers | gemini-2.0-flash | Google | Efficient for simple factual responses |
| **Technical Capabilities** | | | |
| Function calling/Tool use | Supported by both | Both | Based on OpenAI-compatible function calling format |
| Streaming responses | Supported by both | Both | Server-sent events (SSE) format for streaming |
| System prompts | Supported by both | Both | As first message in conversation |
| **Context Windows** | | | |
| Deepseek | 8192 tokens | Deepseek | Smaller than Claude Sonnet (200K) and Opus (1M) |
| Gemini | 1,048,576 input / 8,192 output | Google | Large input context but limited output tokens |
| **Integration** | | | |
| API format | OpenAI-compatible | Both | Similar format to OpenAI, works with LiteLLM |
| Base URLs | https://api.deepseek.com | Deepseek | Default API endpoints |
| | https://generativelanguage.googleapis.com | Google | |
| API parameters | OpenAI-compatible | Both | Temperature, top_p, top_k, similar to OpenAI/Claude |
| **Content Types** | | | |
| Text generation | Supported by both | Both | Core capability |
| Code generation | Supported by both | Both | Strong in Deepseek models |
| Image understanding | Supported | Google | Gemini has strong multimodal capabilities |
| **Developer Tools** | | | |
| Integration | LiteLLM | Both | Common integration framework |
| Utilities | Function calling frameworks | Both | Support for structured tool use |

## Key Considerations

When using alternative models with Claude Code through this proxy:

1. **Model Selection**: 
   - **Haiku models** map to `gemini-2.0-flash` (general tasks, quick responses)
   - **Sonnet models** map to `deepseek-chat` + optional Chain-of-Thought (CoT) system prompt (optimal for reasoning and coding tasks)

2. **Token Limits**: 
   - Both models have output token limits of 8192 (compared to Claude's much larger limits)
   - Gemini models have a large input context window (over 1M tokens)

3. **Function Calling**: 
   - Both models support function calling in an OpenAI-compatible format
   - Implementation details vary slightly between providers

4. **Performance**: 
   - Response times are generally slower than native Claude models
   - Consider experimenting with different models for your specific use case

5. **Customization**:
   - You can override default mapping through environment variables
   - Setting `--always-cot` flag enhances reasoning for Sonnet-mapped models