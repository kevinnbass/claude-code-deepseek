# Claude Code with GLM 4.5 🧠

A proxy server that lets you use Claude Code with GLM 4.5, providing a cost-effective alternative while maintaining high-quality code assistance capabilities.

![DEEPSEEK CODE banner](https://raw.githubusercontent.com/kevinnbass/claude-code-deepseek/main/assets/banner.png)

## What This Does 🔄

This proxy acts as a bridge between the Claude Code client and GLM 4.5:

1. It intercepts requests from Claude Code intended for Anthropic's API
2. Transforms these requests into a format compatible with GLM 4.5
3. Forwards them to the GLM API service
4. Converts responses back to match Anthropic's expected format
5. Returns them to the Claude Code client

The result: You can use Claude Code's excellent interface while leveraging the powerful GLM 4.5 model.

### Key Benefits 💰

- **Cost Savings**: Use Claude Code with more affordable GLM 4.5
- **High Performance**: GLM 4.5 provides excellent code assistance capabilities
- **Unified Model**: All requests use the same high-quality GLM 4.5 model
- **Transparency**: Use the same Claude Code interface without changing your workflow
- **Enhanced Features**: Added `/brainstorm` command and other customizations
- **High Token Limits**: GLM 4.5 supports up to 98K output tokens

## Quick Start ⚡

### Prerequisites

- GLM API key 🔑 (for all models - get from https://open.bigmodel.cn)
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
   
   Then edit the `.env` file with your API key:
   ```
   GLM_API_KEY=your-glm-key  # Get from https://open.bigmodel.cn
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

4. **Start using it!** Your Claude Code client is now connected to cost-effective alternative models.

## Features 🌟

### Model Mapping 🗺️

| Claude Model | Mapped Model | Provider | Use Case |
|--------------|--------------|----------|----------|
| claude-3-haiku | glm-4.5 | GLM | All tasks - high-quality responses |
| claude-3-sonnet | glm-4.5 | GLM | All tasks - high-quality responses |

Customize via environment variables:
```
BIG_MODEL=glm-4.5         # Model to use for Sonnet tasks
SMALL_MODEL=glm-4.5       # Model to use for Haiku tasks  
```

### Core Capabilities ✨

- ✅ **Text & Code generation** - High-quality responses and code
- ✅ **Function calling / Tool usage** - Full tool support
- ✅ **Streaming responses** - Real-time streaming 
- ✅ **Multi-turn conversations** - Context preservation
- ✅ **System prompts** - Full system instruction support
- ✅ **Chain-of-Thought** - Enhanced reasoning with `--always-cot` flag

## Cost Comparison & Savings 💰

### GLM 4.5 Pricing vs Anthropic Models

*Pricing confirmed from https://open.bigmodel.cn/pricing and https://www.anthropic.com/pricing#api*

**GLM 4.5 Tiered Pricing Structure (converted from yuan to USD at ~0.14 exchange rate):**

| Input Length (K tokens) | Output Length (K tokens) | Input Cost (50% off) | Output Cost (50% off) | Input Cost (full price) | Output Cost (full price) |
|-------------------------|---------------------------|---------------------|----------------------|----------------------|---------------------|
| 0-32K | 0-200 | $0.14/1M | $0.28/1M | $0.28/1M | $0.56/1M |
| 0-32K | 200+ | $0.21/1M | $0.42/1M | $0.42/1M | $0.84/1M |
| 32-128K | Any | $0.28/1M | $0.56/1M | $0.56/1M | $1.12/1M |

**Anthropic Pricing:**

| Model | Input Cost (per 1M tokens) | Output Cost (per 1M tokens) |
|-------|----------------------------|------------------------------|
| **Claude Opus 4.1** | $15.00 | $75.00 |
| **Claude Sonnet 4** | $3.00 (≤200K) / $6.00 (>200K) | $15.00 (≤200K) / $22.50 (>200K) |

**Cost Comparison for Typical Coding Session (20K input, 5K output tokens):**

| Model | Session Cost | Annual Cost (500K tokens/month) |
|-------|-------------|--------------------------------|
| **Claude Opus 4.1** | $675.00 | $40,500 |
| **Claude Sonnet 4** | $135.00 | $8,100 |
| **GLM 4.5 (50% off)** | $4.20 | $252 |
| **GLM 4.5 (full price)** | $8.40 | $504 |

### Massive Cost Savings

**With Current 50% Off Promotion (until August 31, 2025):**
- **99.4% savings** vs Claude Opus ($4.20 vs $675.00)
- **96.9% savings** vs Claude Sonnet ($4.20 vs $135.00)

**Even at Full Price (after August 31, 2025):**
- **98.8% savings** vs Claude Opus ($8.40 vs $675.00)
- **93.8% savings** vs Claude Sonnet ($8.40 vs $135.00)

### Promotional Benefits

🎉 **Current Limited-Time Offers:**
- **50% discount** on all GLM models until **August 31, 2025**
- **20M free tokens** for new registrations (worth $11.20 at full price!)
- Tiered pricing that scales with usage - pay less for smaller context windows
- Additional promotional credits available through Zhipu AI platform

**Bottom line**: GLM 4.5 delivers comparable performance at a fraction of the cost, making advanced AI assistance accessible for individual developers and small teams.

## Limitations ⚠️

- **Context window**: Smaller context windows compared to Claude's 200K+
- **Multimodal content**: Limited compared to Claude's multimodal capabilities

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
2. It routes all requests to GLM 4.5 via the GLM API
3. It transforms the request format for the GLM API
4. It optionally adds Chain-of-Thought prompting (when `--always-cot` flag is used)
5. It processes the response and converts it back to Anthropic's format
6. The Claude Code client receives responses in the expected format

### Special Feature: `/brainstorm` Command ✨

This proxy extends Claude Code with custom slash commands for specialized tasks. The most powerful is the `/brainstorm` command that uses GLM 4.5 with a specialized brainstorming system prompt.

Generate creative ideas for any code challenge or problem:

```
/brainstorm How can I optimize CI/CD pipelines for our microservices architecture?
```

The `/brainstorm` command:
- Uses a specialized system prompt tailored for code-related brainstorming
- **Powered by GLM 4.5** with enhanced reasoning capabilities
- Generates diverse, actionable ideas with implementation details and code snippets
- Includes tradeoffs and considerations for each solution

**When to use**: Architecture decisions, code optimization, refactoring approaches, testing strategies, and creative problem solving.

**Example use cases:** API design, database optimization, refactoring approaches, testing strategies, performance optimization, and error handling patterns.

Future planned commands include `/debug`, `/refactor`, and `/perf`. Contributions welcome!

For a comprehensive comparison between Claude and alternative model capabilities, see [CAPABILITIES.md](CAPABILITIES.md).
For detailed performance metrics and analysis, see [PERFORMANCE_SUMMARY.md](PERFORMANCE_SUMMARY.md).

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue.

## License

[MIT License](LICENSE)