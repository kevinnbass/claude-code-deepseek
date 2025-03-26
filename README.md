# Claude Code with Deepseek Models 🧠

A proxy server that lets you use Claude Code with Deepseek models, providing a cost-effective alternative while maintaining high-quality code assistance capabilities.

## What This Does 🔄

This proxy acts as a bridge between the Claude Code client and Deepseek's AI models:

1. It intercepts requests from Claude Code intended for Anthropic's API
2. Transforms these requests into a format compatible with Deepseek models
3. Forwards them to Deepseek's API service
4. Converts Deepseek responses back to match Anthropic's expected format
5. Returns them to the Claude Code client

The result: You can use Claude Code's excellent interface while leveraging the more affordable Deepseek models.

## Quick Start ⚡

### Prerequisites

- Deepseek API key 🔑
- Node.js (for Claude Code CLI)

### Setup 🛠️

1. **Clone this repository**:
   ```bash
   git clone https://github.com/kevinnbass/claude-code-deepseek.git
   cd claude-code-deepseek
   ```

2. **Install UV** (for Python dependency management):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. **Configure your API keys**:
   Create a `.env` file with:
   ```
   DEEPSEEK_API_KEY=your-deepseek-key
   ```

4. **Start the proxy server**:
   ```bash
   uv run uvicorn server:app --host 0.0.0.0 --port 8082
   ```

   To always add Chain-of-Thought prompting for Sonnet models (regardless of thinking parameter):
   ```bash
   uv run server.py --always-cot
   ```

### Using with Claude Code 🖥️

1. **Install Claude Code** (if you haven't already):
   ```bash
   npm install -g @anthropic-ai/claude-code
   ```

2. **Connect to your proxy**:
   ```bash
   ANTHROPIC_BASE_URL=http://127.0.0.1:8082 claude
   ```
   
   Note: Using the IP address directly (127.0.0.1) instead of localhost can resolve connection issues.

3. **Start coding!** 👨‍💻👩‍💻 Your Claude Code client will now use Deepseek models through the proxy.

## Features 🌟

### Model Mapping 🗺️

| Claude Model | Deepseek Model | Use Case |
|--------------|----------------|----------|
| claude-3-haiku | deepseek-chat | Quick responses, simpler tasks |
| claude-3-sonnet | deepseek-chat | Complex reasoning, longer code generation |

### Customization Options ⚙️

Customize which models are used via environment variables in your `.env` file:

```
DEEPSEEK_API_KEY=your-deepseek-key
BIG_MODEL=deepseek-chat    # Model to use for Sonnet (complex tasks)
SMALL_MODEL=deepseek-chat  # Model to use for Haiku (simpler tasks)
```

### Chain-of-Thought Prompting 🧠

The proxy supports automatic Chain-of-Thought (CoT) prompting to enhance reasoning capabilities:

- **Default behavior**: CoT prompting is applied to Sonnet models only when thinking mode is enabled
- **Always-CoT mode**: Force CoT prompting for all Sonnet requests with the `--always-cot` flag

```bash
uv run server.py --always-cot
```

### Core Capabilities ✨

- ✅ **Text & Code generation**
- ✅ **Function calling / Tool usage** 
- ✅ **Streaming responses**
- ✅ **Multi-turn conversations**
- ✅ **System prompts**

## Performance Comparison 📊

| Feature | Response Time Comparison |
|---------|--------------------------|
| Simple text generation | Deepseek ~5-7s vs Claude ~1-2s |
| Complex reasoning | Deepseek ~12-15s vs Claude ~2-3s |
| Code generation | Deepseek ~15-18s vs Claude ~3-4s |
| Tool usage | Deepseek ~5-6s vs Claude ~1-2s |

## Limitations ⚠️

- **Token limit**: Deepseek models have a maximum token limit of 8192 (automatically enforced)
- **Response time**: Deepseek models typically have longer response times than Claude models

## Technical Details 🔧

- Uses [LiteLLM](https://github.com/BerriAI/litellm) for model API standardization
- Handles both streaming and non-streaming responses
- Implements proper error handling and graceful fallbacks
- Manages content blocks and tool usage conversions between different API formats

## Detailed Capabilities

For a comprehensive comparison between Claude and Deepseek model capabilities, see [CAPABILITIES.md](CAPABILITIES.md).

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue.

## License

[Insert your license information here]