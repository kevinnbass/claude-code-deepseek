# Claude Code with Deepseek and Gemini Models 🧠

A proxy server that lets you use Claude Code with Deepseek and Google Gemini models, providing a cost-effective alternative while maintaining high-quality code assistance capabilities. This solution delivers up to **96% cost savings** compared to using Claude directly, with comparable or better performance for many tasks.

**✨ SPECIAL FEATURE: `/brainstorm` command connects directly to Claude 3.7 Sonnet for exceptional brainstorming quality! ✨**

![DEEPSEEK CODE banner](https://raw.githubusercontent.com/kevinnbass/claude-code-deepseek/main/assets/banner.png)

## What This Does 🔄

This proxy acts as a bridge between the Claude Code client and alternative AI models:

1. It intercepts requests from Claude Code intended for Anthropic's API
2. Transforms these requests into a format compatible with Deepseek or Gemini models
3. Forwards them to the appropriate API service
4. Converts responses back to match Anthropic's expected format
5. Returns them to the Claude Code client

The result: You can use Claude Code's excellent interface while leveraging more affordable Deepseek and Gemini models.

### Key Benefits 💰

- **Massive Cost Savings**: Use Claude Code with up to 96% lower costs
- **Similar Performance**: Comparable response quality for code tasks
- **Best of Both Worlds**: Use cost-effective models for most tasks, but access Claude 3.7 for brainstorming
- **Transparency**: Use the same Claude Code interface without changing your workflow
- **Flexibility**: Choose which models to use for different types of tasks
- **Customizability**: Added features not available in Claude Code by default

### New Feature: Custom Commands 🌟

This proxy extends Claude Code with custom slash commands for specialized tasks:

- **`/brainstorm`** - A special command that connects directly to Claude 3.7 Sonnet for high-quality code brainstorming (requires Anthropic API key)

## Quick Start ⚡

### Prerequisites

- Deepseek API key 🔑 (for Sonnet models)
- Gemini API key 🔑 (for Haiku models)
- Anthropic API key 🔑 (optional, required only for `/brainstorm` command)
- Node.js (for Claude Code CLI)

### Setup 🛠️

1. **Clone this repository**:
   ```bash
   git clone https://github.com/kevinnbass/claude-code-deepseek.git
   cd claude-code-deepseek
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -e .
   # OR with UV (recommended)
   curl -LsSf https://astral.sh/uv/install.sh | sh
   uv pip install -e .
   ```

3. **Configure your API keys**:
   ```bash
   cp .env.example .env
   ```
   
   Then edit the `.env` file with your API keys:
   ```
   DEEPSEEK_API_KEY=your-deepseek-key  # Get from https://platform.deepseek.com
   GEMINI_API_KEY=your-gemini-key      # Get from https://ai.google.dev/
   ANTHROPIC_API_KEY=your-anthropic-key  # Optional, needed for /brainstorm
   ```

4. **Start the proxy server**:
   ```bash
   python server.py --always-cot
   # OR with UV
   uv run server.py --always-cot
   ```

   **Server Options:**
   - `--always-cot`: Recommended flag that improves reasoning capability with Chain-of-Thought prompting
   - Debug mode enabled by setting `DEBUG=true` in `.env` file

### Using with Claude Code 🖥️

1. **Install Claude Code CLI**:
   ```bash
   npm install -g @anthropic-ai/claude-code
   ```

2. **Connect to your proxy**:
   ```bash
   ANTHROPIC_BASE_URL=http://127.0.0.1:8082 claude
   ```

3. **Verify proxy connection**: You should see the standard Claude Code welcome message with no errors.

4. **Try custom commands**: Test the `/brainstorm` command
   ```
   /brainstorm How could I refactor a complex legacy codebase?
   ```

5. **Start using it!** Your Claude Code client is now connected to cost-effective alternative models.

## Features 🌟

### Custom Commands 🤖

#### `/brainstorm` Command

Generate creative ideas for any code challenge or problem:

```
/brainstorm How can I optimize CI/CD pipelines for our microservices architecture?
```

![Brainstorm Command Example](https://raw.githubusercontent.com/kevinnbass/claude-code-deepseek/main/assets/brainstorm-example.png)

The `/brainstorm` command:
- Uses a specialized system prompt tailored for code-related brainstorming
- **Connects directly to Claude 3.7 Sonnet** (requires ANTHROPIC_API_KEY)
- Generates diverse, actionable ideas with implementation details and code snippets
- Includes tradeoffs and considerations for each solution

**When to use**: Architecture decisions, code optimization, refactoring approaches, testing strategies, and when quality of ideas is more important than response speed.

**Example use cases:** API design, database optimization, refactoring approaches, testing strategies, performance optimization, and error handling patterns.

Note: The `/brainstorm` command is the only feature that uses the Anthropic API; all other interactions use the cost-saving models.

Future planned commands include `/debug`, `/refactor`, and `/perf`. Contributions welcome!

### Model Mapping 🗺️

| Claude Model | Mapped Model | Provider | Use Case |
|--------------|--------------|----------|----------|
| claude-3-haiku | gemini-2.0-flash | Google | Quick responses, simpler tasks |
| claude-3-sonnet | deepseek-chat | Deepseek | Complex reasoning, longer code generation |

Customize via environment variables:
```
BIG_MODEL=deepseek-chat         # Model to use for Sonnet tasks
SMALL_MODEL=gemini-2.0-flash    # Model to use for Haiku tasks
```

### Core Capabilities ✨

- ✅ **Text & Code generation** - High-quality responses and code
- ✅ **Function calling / Tool usage** - Full tool support
- ✅ **Streaming responses** - Real-time streaming 
- ✅ **Multi-turn conversations** - Context preservation
- ✅ **System prompts** - Full system instruction support
- ✅ **Chain-of-Thought** - Enhanced reasoning with `--always-cot` flag

## Performance & Cost Comparison 📊

### Response Times

| Task Type | Gemini Flash | Deepseek Chat | Claude | Performance |
|-----------|--------------|---------------|--------|-------------|
| Simple text | 0.7-0.8s | - | 1.0-1.1s | 0.7x (faster) |
| Tool usage | 0.6-0.7s | - | ~1.0s | 0.65x (faster) |
| Complex reasoning | - | 15-16s | ~7-8s | 2.0x (slower) |
| Code generation | - | 15-20s | ~16s | 1.1x (similar) |

**Key observations:**
* Gemini Flash is faster than Claude Haiku for simple tasks
* Deepseek Chat takes longer for complex reasoning but produces high-quality outputs
* Code generation performance is comparable to Claude
* Performance formula: Haiku tasks = 0.7x Claude time, Sonnet tasks = ~2.0x Claude time

### Cost Savings

| Model | Input Cost | Output Cost | Relative Cost |
|-------|------------|-------------|---------------|
| **Claude 3.7 Sonnet** | $3.00 per 1M | $15.00 per 1M | Baseline |
| **DeepSeek Chat** | $0.135 per 1M | $0.55 per 1M | ~3.7% of Claude |
| **Claude 3.5 Haiku** | $0.80 per 1M | $4.00 per 1M | Baseline |
| **Gemini 2.0 Flash** | $0.10 per 1M | $0.40 per 1M | ~10% of Claude |

For a typical coding session (20K input, 5K output tokens):
- **Claude Sonnet**: $135
- **DeepSeek Chat**: $5.47 (~96% savings)
- **Claude Haiku**: $36
- **Gemini Flash**: $3.02 (~92% savings)

## Limitations ⚠️

- **Token limit**: Maximum output token limit of 8192
- **Response time**: Complex reasoning tasks with Deepseek can take 30-45 seconds vs. Claude's 2-3s
- **Context window**: Smaller context windows (8K tokens vs. Claude's 200K+)
- **Multimodal content**: Image processing capabilities may vary by model

## Technical Details 🔧

### Architecture

- **API Compatibility Layer**: Implements Anthropic's API endpoints
- **Model Routing Logic**: Routes requests to appropriate providers
- **Request/Response Transformation**: Converts between API formats
- **Custom Command System**: Extends functionality with specialized commands
- **Streaming Support**: Full event streaming compatibility

### Technologies

- **FastAPI**: High-performance web framework
- **LiteLLM**: Standardized interface to multiple LLM providers
- **HTTPX**: Modern HTTP client for async API communication
- **Python 3.9+**: Core language with robust async support

### How It Works

1. The proxy receives requests from Claude Code in Anthropic's format
2. It identifies request type and routes to appropriate provider
3. It transforms the request format for the target API
4. For Sonnet models, it optionally adds Chain-of-Thought prompting
5. It processes the response and converts it back to Anthropic's format
6. The Claude Code client receives responses in the expected format

For a comprehensive comparison between Claude and alternative model capabilities, see [CAPABILITIES.md](CAPABILITIES.md).
For detailed performance metrics and analysis, see [PERFORMANCE_SUMMARY.md](PERFORMANCE_SUMMARY.md).

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue.

## License

[MIT License](LICENSE)