# Claude to Deepseek Capability Mapping

This document provides a reference for how Claude models and capabilities map to their Deepseek equivalents when using the proxy.

## Model Mapping

| Claude Capability | Deepseek Equivalent | Notes |
|-------------------|---------------------|-------|
| **Core Models** | | |
| Claude Haiku | deepseek-chat | General purpose chat model, default for smaller tasks |
| Claude Sonnet | deepseek-chat | General purpose chat model, default for complex tasks |
| Claude Opus | Not directly mapped | No direct equivalent in current Deepseek lineup |
| **Specialized Tasks** | | |
| Code generation | deepseek-chat | Use standard deepseek-chat for all coding tasks |
| Reasoning | deepseek-chat + CoT prompt | Use with Chain-of-Thought system prompt for complex reasoning |
| **Technical Capabilities** | | |
| Function calling/Tool use | Supported | Based on OpenAI-compatible function calling format |
| Streaming responses | Supported | Server-sent events (SSE) format for streaming |
| System prompts | Supported | As first message in conversation |
| Context window | 8192 tokens | Smaller than Claude Sonnet (200K) and Opus (1M) |
| JSON mode | Unknown | No specific information found |
| Multi-turn conversations | Supported | Standard chat format with role-based messages |
| **Integration** | | |
| API format | OpenAI-compatible | Uses similar format to OpenAI, works with LiteLLM |
| Base URL | https://api.deepseek.com | Default API endpoint |
| API parameters | OpenAI-compatible | Temperature, top_p, top_k, similar to OpenAI/Claude |
| **Content Types** | | |
| Text generation | Supported | Core capability |
| Code generation | Supported | Use standard deepseek-chat for all coding tasks |
| Image understanding | Unknown | No clear documentation on multimodal capabilities |
| **Developer Tools** | | |
| ChatUI | Multiple options | Many integrations like Cursor, 16x Prompt |
| SDK/Tools | LiteLLM, YoMo | Support for function calling frameworks |

## Key Considerations

When using Deepseek models with Claude Code through this proxy:

1. **Token Limit**: Deepseek models have a significantly smaller context window (8192 tokens) compared to Claude models (200K for Sonnet, 1M for Opus)

2. **Model Selection**: 
   - **Haiku models** map to `deepseek-chat` (general tasks)
   - **Sonnet models** map to `deepseek-chat` + automatic Chain-of-Thought (CoT) system prompt (optimal for reasoning tasks)
   - All coding tasks use `deepseek-chat` regardless of the selected model

3. **Function Calling**: Deepseek supports function calling in an OpenAI-compatible format

4. **Performance**: Consider experimenting with different models for your specific use case to find the optimal balance between performance and capability