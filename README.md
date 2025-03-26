# Claude Code with Deepseek and Gemini Models 🧠

A proxy server that lets you use Claude Code with Deepseek and Google Gemini models, providing a cost-effective alternative while maintaining high-quality code assistance capabilities. This solution delivers up to **96% cost savings** compared to using Claude directly, with comparable or better performance for many tasks.

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
   Using pip:
   ```bash
   pip install -e .
   ```
   
   Or using UV (recommended for faster installation):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   uv pip install -e .
   ```

3. **Configure your API keys**:
   Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
   
   Then edit the `.env` file with your API keys:
   ```
   # Required: At least one of these is needed
   DEEPSEEK_API_KEY=your-deepseek-key  # Get from https://platform.deepseek.com
   GEMINI_API_KEY=your-gemini-key      # Get from https://ai.google.dev/
   
   # Optional: Only needed for /brainstorm command
   ANTHROPIC_API_KEY=your-anthropic-key  # Get from https://console.anthropic.com
   
   # Uncomment to enable debug logging
   # DEBUG=true
   ```

4. **Start the proxy server**:
   ```bash
   python server.py --always-cot
   ```
   
   Or with UV:
   ```bash
   uv run server.py --always-cot
   ```

   The server will display a colorful ASCII art logo and information about its configuration:
   
   ```
   ████████╗███████╗███████╗██████╗ ███████╗███████╗███████╗██╗  ██╗
   ██╔═══██╗██╔════╝██╔════╝██╔══██╗██╔════╝██╔════╝██╔════╝██║ ██╔╝
   ██║   ██║█████╗  █████╗  ██████╔╝███████╗█████╗  █████╗  █████╔╝ 
   ██║   ██║██╔══╝  ██╔══╝  ██╔═══╝ ╚════██║██╔══╝  ██╔══╝  ██╔═██╗ 
   ████████║███████╗███████╗██║     ███████║███████╗███████╗██║  ██╗
   ╚═══════╝╚══════╝╚══════╝╚═╝     ╚══════╝╚══════╝╚══════╝╚═╝  ╚═╝

    ██████╗ ██████╗ ██████╗ ███████╗
   ██╔════╝██╔═══██╗██╔══██╗██╔════╝
   ██║     ██║   ██║██║  ██║█████╗  
   ██║     ██║   ██║██║  ██║██╔══╝  
   ╚██████╗╚██████╔╝██████╔╝███████╗
    ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝
   ```

   **Server Options:**
   - `--always-cot`: Recommended flag that significantly improves reasoning capability by adding Chain-of-Thought prompting for all Sonnet model requests
   - Debug mode can be enabled by setting `DEBUG=true` in your `.env` file

### Using with Claude Code 🖥️

1. **Install Claude Code CLI** (if you haven't already):
   ```bash
   npm install -g @anthropic-ai/claude-code
   ```

2. **Connect to your proxy**:
   In a new terminal window, run:
   ```bash
   ANTHROPIC_BASE_URL=http://127.0.0.1:8082 claude
   ```
   
   **Tip**: Using the IP address directly (127.0.0.1) instead of localhost can resolve connection issues.

3. **Verify proxy connection**:
   When Claude Code starts, you should see:
   - Standard Claude Code welcome message
   - No errors about connection issues
   - Commands should work as expected

4. **Try custom commands**:
   Test that the `/brainstorm` command works by typing:
   ```
   /brainstorm How could I refactor a complex legacy codebase?
   ```
   
   You should see a comprehensive, structured response with multiple approaches.

5. **Start using it!** 👨‍💻👩‍💻
   Your Claude Code client is now connected to cost-effective alternative models through the proxy, while maintaining the same interface and user experience.

### Custom Commands 🤖

This proxy extends Claude Code with powerful custom commands:

#### `/brainstorm` Command

Generate creative ideas for any code challenge or problem:

```
/brainstorm How can I optimize CI/CD pipelines for our microservices architecture?
```

![Brainstorm Command Example](https://raw.githubusercontent.com/kevinnbass/claude-code-deepseek/main/assets/brainstorm-example.png)

The `/brainstorm` command:
- Uses a specialized system prompt tailored for code-related brainstorming
- **Connects directly to Claude 3.7 Sonnet** (requires ANTHROPIC_API_KEY in your .env file)
- Generates at least 5 diverse, actionable ideas with implementation details and code snippets
- Includes tradeoffs and considerations for each solution
- Perfect for architecture decisions, code optimization, and solving complex technical challenges

**When to use `/brainstorm`:**
- When you need deep, creative thinking about complex code problems
- For architectural decisions that require multiple perspectives
- When generating alternative approaches to implementation
- For team planning sessions where multiple viable options are needed
- When quality of ideas is more important than response speed

**Example use cases:**
- API design strategies
- Database schema optimization
- Refactoring approaches
- Testing strategies
- Performance optimization techniques
- Error handling patterns

Note: The `/brainstorm` command is the only feature that uses the actual Anthropic API; all other interactions still use the cost-saving Deepseek/Gemini models.

#### Future Custom Commands

The extensible architecture allows for easy addition of new custom commands:

- `/debug` - Step-by-step debugging assistance (planned)
- `/refactor` - Code refactoring suggestions (planned)
- `/perf` - Performance optimization strategies (planned)

Contributions of new commands are welcome!

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
- **Always-CoT mode**: Force CoT prompting for all Sonnet requests with the `--always-cot` flag (recommended)

```bash
python server.py --always-cot
```

### Core Capabilities ✨

Our recent testing confirms full compatibility with:

- ✅ **Text & Code generation** - Reliable generation of text responses and high-quality code
- ✅ **Function calling / Tool usage** - Full support for tool definitions and function calling
- ✅ **Streaming responses** - Proper event handling for streaming text and tool use 
- ✅ **Multi-turn conversations** - Context preservation across multiple turns
- ✅ **System prompts** - Full support for system instructions
- ✅ **Custom commands** - Special slash commands with enhanced functionality

## Test Results 📊

All core capabilities have been verified through comprehensive testing:

| Test Case | Status | Notes |
|-----------|--------|-------|
| Simple text generation | ✅ PASS | Fast responses (~0.9s) |
| Calculator tool usage | ✅ PASS | Properly formats function calls |
| Multiple tool usage | ✅ PASS | Successfully handles weather and search tools |
| Multi-turn conversation | ✅ PASS | Maintains context across messages |
| Complex content blocks | ✅ PASS | Correctly processes different content types |
| Chain-of-Thought reasoning | ✅ PASS | Successfully solves math problems with step-by-step reasoning |
| Code generation | ✅ PASS | Generates correct, well-formatted code |
| Streaming text | ✅ PASS | All required event types present |
| Streaming with tools | ✅ PASS | Proper handling of streaming tool calls |

### Running Tests

A comprehensive test suite is included to verify functionality and performance:

```bash
# Run all tests
python run_tests.py --all

# Run only basic functionality tests
python run_tests.py --basic

# Run performance comparison tests
python run_tests.py --performance

# Test only proxy functionality (skip Anthropic API)
python run_tests.py --all --proxy-only

# Test with specific Anthropic API key
python run_tests.py --all --anthropic-key=your-api-key
```

Test results and performance comparisons are saved to the `tests/results/` directory.

### Project Structure

```
claude-code-deepseek/
├── server.py           # Main proxy server implementation
├── run_tests.py        # Primary test runner script
├── logs/               # Server logs
├── .env                # API keys and configuration (create from .env.example)
├── .env.example        # Template for environment variables
├── tests/              # Test package
│   ├── __init__.py     # Common test constants and data
│   ├── utils.py        # Shared test utilities
│   ├── basic_tests.py  # Basic functionality tests
│   ├── performance_tests.py  # Performance comparison tests
│   ├── api_key_test.py       # Anthropic API key validation
│   ├── run_tests.py          # Package test runner
│   ├── simple_comparison.py  # Simple performance comparison
│   ├── backups/              # Original test file backups 
│   └── results/              # Test output directory
```

The codebase is organized for maintainability with separated concerns:
- The main server implements the proxy functionality and custom commands
- A comprehensive test suite verifies functionality and performance 
- Logs are stored in a dedicated directory for easier debugging

### Performance Metrics

Latest benchmark testing shows the following response times:

| Task Type | Gemini Flash | Deepseek Chat | Claude (measured) | Performance Ratio |
|-----------|--------------|---------------|-------------------|-----------------|
| Simple text generation | 0.7-0.8s | - | 1.0-1.1s | 0.7x (faster) |
| Tool usage (calculator) | 0.6-0.7s | - | ~1.0s | 0.65x (faster) |
| Multi-turn conversation | 0.5-0.6s | - | ~1.0s | 0.6x (faster) |
| Complex reasoning | - | 15-16s | ~7-8s | 2.0x (slower) |
| Code generation | - | 15-20s | ~16s | 1.1x (similar) |
| Streaming responses | Real-time | Real-time | Real-time | 1.0x (similar) |

**Key observations:**
* Gemini Flash is significantly faster than Claude Haiku for simple tasks
* Deepseek Chat takes longer for complex reasoning but produces high-quality outputs
* Code generation performance is comparable to Claude
* Performance formula: Haiku tasks = 0.7x Claude time, Sonnet tasks = ~2.0x Claude time
* All tests pass with 100% success rate in comprehensive testing
* Best performance is achieved with `--always-cot` flag for reasoning tasks

For detailed performance metrics and analysis, see [PERFORMANCE_SUMMARY.md](PERFORMANCE_SUMMARY.md).

### Cost Comparison 💰

One of the main benefits of using this proxy is the significant cost savings compared to using Claude directly, while maintaining high-quality responses:

| Model | Input Tokens | Output Tokens | Context Window | Relative Cost |
|-------|--------------|---------------|----------------|---------------|
| **Claude 3.7 Sonnet** | $3.00 per 1M | $15.00 per 1M | 200K tokens | Baseline |
| **DeepSeek Chat** (discount pricing) | $0.135 per 1M | $0.55 per 1M | 8K tokens | ~3.7% of Claude |
| **Claude 3.5 Haiku** | $0.80 per 1M | $4.00 per 1M | 200K tokens | Baseline |
| **Gemini 2.0 Flash** | $0.10 per 1M | $0.40 per 1M | 1M input / 8K output | ~10% of Claude |

**Savings Examples:**

For a typical coding session with 20K input tokens and 5K output tokens:
- **Claude Sonnet**: $135 (20K input + 5K output)
- **DeepSeek Chat**: $5.47 (~96% savings)
- **Claude Haiku**: $36 (20K input + 5K output)
- **Gemini Flash**: $3.02 (~92% savings)

**Notes on Cost Structure:**

- **DeepSeek** offers discounted rates (50% off) during UTC 16:30-00:30
- **Gemini** has recently lowered their prices significantly with the 2.0 Flash model
- Both alternative providers offer additional savings with caching
- Batch processing with Claude can offer discounts, but alternative models remain more cost-effective

This proxy enables you to use Claude Code's excellent interface while leveraging the more cost-effective Deepseek and Gemini models for significant savings.

## Model Provider Information 🏢

### Deepseek

- Used for Sonnet model mapping by default
- Provides strong code generation and reasoning capabilities
- API documentation: [Deepseek AI](https://platform.deepseek.com/)

### Google Gemini

- Used for Haiku model mapping by default (using Gemini 2.0 Flash)
- Offers fast responses for simpler tasks and tool usage
- API documentation: [Google AI](https://ai.google.dev/gemini-api/docs)

## Limitations ⚠️

- **Token limit**: Both Deepseek and Gemini models have a maximum output token limit of 8192 (automatically enforced)
- **Response time**: Complex reasoning tasks with Deepseek can take 30-45 seconds, compared to Claude's 2-3s
- **Multimodal content**: Image processing capabilities may vary by model
- **Specialized formats**: Some Claude-specific format options may not be fully supported

## Technical Details 🔧

### Architecture

This proxy provides a transparent bridge between Claude Code and alternative AI models:

- **API Compatibility Layer**: Implements Anthropic's API endpoints and response formats
- **Model Routing Logic**: Routes requests to the appropriate provider based on model type
- **Request/Response Transformation**: Converts between API formats with full compatibility
- **Custom Command System**: Extends functionality with specialized commands
- **Streaming Support**: Full compatibility with event streaming for real-time responses

### Technologies

- **[FastAPI](https://fastapi.tiangolo.com/)**: High-performance web framework for the proxy server
- **[LiteLLM](https://github.com/BerriAI/litellm)**: Standardized interface to multiple LLM providers
- **[HTTPX](https://www.python-httpx.org/)**: Modern HTTP client for async API communication
- **[Python 3.9+](https://www.python.org/)**: Core language with robust async support

### How It Works

1. The proxy receives requests from Claude Code in Anthropic's format
2. It identifies request type:
   - Normal model request: Routes to Deepseek/Gemini
   - Custom command: Uses specialized handler (e.g., `/brainstorm` routes to Anthropic)
3. It transforms the request format appropriately for the target API
4. For Sonnet models, it optionally adds Chain-of-Thought prompting (with `--always-cot`)
5. It processes the response and converts it back to Anthropic's format
6. The Claude Code client receives responses in the expected format

### Environment Configuration

The proxy can be configured through environment variables:
- `DEEPSEEK_API_KEY`: Your Deepseek API key
- `GEMINI_API_KEY`: Your Google Gemini API key
- `ANTHROPIC_API_KEY`: Optional, required only for `/brainstorm` command
- `BIG_MODEL`: Model to use for Sonnet requests (default: `deepseek-chat`)
- `SMALL_MODEL`: Model to use for Haiku requests (default: `gemini-2.0-flash`)
- `DEBUG`: Set to `true` to enable debug logging

### Server Command-Line Options

The server accepts these command-line arguments:
- `--always-cot`: Always add Chain-of-Thought system prompt for Sonnet models (recommended)
- `--help`: Display help information

## Detailed Capabilities

For a comprehensive comparison between Claude and alternative model capabilities, see [CAPABILITIES.md](CAPABILITIES.md).

## Contributing 🤝

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue.

## License

[MIT License](LICENSE)