# Claude Code with Deepseek and Gemini Models 🧠

A proxy server that lets you use Claude Code with Deepseek and Google Gemini models, providing a cost-effective alternative while maintaining high-quality code assistance capabilities.

## What This Does 🔄

This proxy acts as a bridge between the Claude Code client and alternative AI models:

1. It intercepts requests from Claude Code intended for Anthropic's API
2. Transforms these requests into a format compatible with Deepseek or Gemini models
3. Forwards them to the appropriate API service
4. Converts responses back to match Anthropic's expected format
5. Returns them to the Claude Code client

The result: You can use Claude Code's excellent interface while leveraging more affordable Deepseek and Gemini models.

## Quick Start ⚡

### Prerequisites

- Deepseek API key 🔑 (for Sonnet models)
- Gemini API key 🔑 (for Haiku models)
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
   GEMINI_API_KEY=your-gemini-key
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

3. **Start coding!** 👨‍💻👩‍💻 Your Claude Code client will now use alternative models through the proxy.

## Features 🌟

### Model Mapping 🗺️

| Claude Model | Mapped Model | Provider | Use Case |
|--------------|--------------|----------|----------|
| claude-3-haiku | gemini-2.0-flash | Google | Quick responses, simpler tasks |
| claude-3-sonnet | deepseek-chat | Deepseek | Complex reasoning, longer code generation |

### Customization Options ⚙️

Customize which models are used via environment variables in your `.env` file:

```
DEEPSEEK_API_KEY=your-deepseek-key
GEMINI_API_KEY=your-gemini-key
BIG_MODEL=deepseek-chat         # Model to use for Sonnet (complex tasks)
SMALL_MODEL=gemini-2.0-flash    # Model to use for Haiku (simpler tasks)
```

You can change the models used for each Claude model type. For example:
- To use a different Gemini model for Haiku: `SMALL_MODEL=gemini-2.0-flash-lite-preview-02-05`
- To use Deepseek for both model types: `SMALL_MODEL=deepseek-chat`

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
| Simple text generation | Alternative models ~5-7s vs Claude ~1-2s |
| Complex reasoning | Alternative models ~12-15s vs Claude ~2-3s |
| Code generation | Alternative models ~15-18s vs Claude ~3-4s |
| Tool usage | Alternative models ~5-6s vs Claude ~1-2s |

## Model Provider Information 🏢

### Deepseek

- Used for Sonnet model mapping by default
- Provides strong code generation and reasoning capabilities
- API documentation: [Deepseek AI](https://platform.deepseek.com/)

### Google Gemini

- Used for Haiku model mapping by default
- Offers fast responses for simpler tasks
- API documentation: [Google AI](https://ai.google.dev/gemini-api/docs)

## Limitations ⚠️

- **Token limit**: Both Deepseek and Gemini models have a maximum output token limit of 8192 (automatically enforced)
- **Response time**: Alternative models typically have longer response times than Claude models
- **Multimodal content**: Image processing capabilities may vary by model
- **Specialized formats**: Some Claude-specific format options may not be fully supported

## Technical Details 🔧

- Uses [LiteLLM](https://github.com/BerriAI/litellm) for model API standardization
- Handles both streaming and non-streaming responses
- Implements proper error handling and graceful fallbacks
- Manages content blocks and tool usage conversions between different API formats

### How It Works

1. The proxy receives requests from Claude Code in Anthropic's format
2. It identifies whether the request is for a Haiku (→ Gemini) or Sonnet (→ Deepseek) model
3. It transforms the request to the appropriate format for the target model
4. For Sonnet models, it optionally adds Chain-of-Thought prompting
5. It processes the response from the target API and converts it back to Claude format
6. The Claude Code client receives responses in the expected format

## Detailed Capabilities

For a comprehensive comparison between Claude and alternative model capabilities, see [CAPABILITIES.md](CAPABILITIES.md).

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue.

## License

[MIT License](LICENSE)