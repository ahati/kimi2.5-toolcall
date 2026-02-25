# AI-Proxy

A high-performance AI API proxy server that provides unified access to multiple AI providers with protocol conversion and response transformation capabilities.

## Features

- **Multi-Provider Support**: Route requests to different upstream providers based on model names
- **Protocol Conversion**: Convert between Anthropic and OpenAI API formats seamlessly
- **Response Transformation**: Fix malformed responses (e.g., Kimi K2.5 proprietary tool call tokens → OpenAI format)
- **Unified Interface**: Single OpenAI-compatible endpoint for multiple backends
- **Metrics Tracking**: Comprehensive latency and token usage monitoring
- **Environment-Based Configuration**: Load API keys from environment variables
- **Streaming Support**: Handle streaming responses with transformation

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd ai-proxy

# Build the binary
go build -o ai-proxy

# Or install directly
go install
```

## Quick Start

1. Create a configuration file `ai-proxy-config.json`:

```json
{
  "port": 8080,
  "log_level": "info",
  "timeout": "600s",
  "upstreams": [
    {
      "name": "chutes",
      "provider": "chutes.ai",
      "endpoint": "https://api.chutes.ai/v1",
      "api_key": "ENV:CHUTES_API_KEY",
      "models": [
        {
          "pattern": "moonshotai/Kimi-K2.5*",
          "transformers": ["kimi-tool-calls"]
        },
        {
          "pattern": "*"
        }
      ]
    },
    {
      "name": "minimax",
      "provider": "minimax.io",
      "endpoint": "https://api.minimax.io/v1",
      "api_key": "ENV:MINIMAX_API_KEY",
      "models": [
        {
          "pattern": "MiniMax-M2.5"
        }
      ]
    }
  ]
}
```

2. Set your API key environment variables:

```bash
export CHUTES_API_KEY="your-api-key-here"
export MINIMAX_API_KEY="your-minimax-key-here"
```

3. Start the proxy:

```bash
./ai-proxy serve
```

4. Make requests:

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "chutes.ai/moonshotai/Kimi-K2.5-TEE",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

## Configuration

### Configuration Sources (Priority Order)

1. **CLI flags** (highest priority)
2. **Custom config file** (`--config-file` argument)
3. **XDG data directory** (`$XDG_DATA_DIR/ai-proxy-config.json`)
4. **Working directory** (`./ai-proxy-config.json`)
5. **Default values** (lowest priority)

### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--port, -p` | 8080 | Listen port |
| `--config-file` | "" | Path to JSON config file |
| `--log-level` | info | Logging level (debug, info, warn, error) |

### API Keys from Environment

API keys can be loaded from environment variables by using the `ENV:` prefix:

```json
{
  "api_key": "ENV:CHUTES_API_KEY"
}
```

Set the environment variable:
```bash
export CHUTES_API_KEY="sk-chutes-xxx"
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat/completions` | OpenAI chat completions |
| POST | `/v1/anthropic/messages` | Anthropic messages (converted to OpenAI) |
| GET | `/v1/models` | List available models |
| GET | `/health` | Health check |
| GET | `/metrics` | Metrics and statistics |

## Metrics

The proxy tracks comprehensive metrics including:

### Latency (in milliseconds)
- **Per request**: Current request latency
- **Rolling average**: Average across all requests
- **Last 10 average**: Average of the most recent 10 requests
- **Available at**: Global, per-provider, and per-model levels

### Token Usage (TPS counters)
- **Input tokens**: Total tokens sent to models
- **Output tokens**: Total tokens received from models
- **Request count**: Total number of requests
- **Available at**: Per-provider and per-model levels

Access metrics at: `GET /metrics`

Example response:
```json
{
  "global_latency": {
    "current_ms": 1234,
    "rolling_avg_ms": 1150.5,
    "last_10_avg_ms": 1200.3,
    "total_count": 42
  },
  "models": [...],
  "providers": [...]
}
```

## Model Routing

Models are routed based on the provider prefix in the model name:

```
<provider>/<path>/<model>
```

Examples:
- `chutes.ai/moonshotai/Kimi-K2.5-TEE`
- `minimax.io/abab6.5s-chat`

The proxy extracts the provider (first segment) and routes to the matching upstream configuration.

## Transformers

### Available Transformers

#### 1. Kimi Tool Calls Fixer (`kimi-tool-calls`)

**Problem**: Kimi K2.5 returns tool calls using proprietary special tokens instead of standard OpenAI format:

```
<|tool_call_begin|>get_weather<|tool_call_argument_begin|>{"location": "Beijing"}<|tool_call_end|>
```

This causes errors like: `Cannot read "image.png" (this model does not support image input)`

**Solution**: Converts raw tool call tokens emitted by Kimi K2.5 into proper OpenAI `tool_calls` format.

**Tokens handled**:
- `<|tool_calls_section_begin|>` - Start of tool calls section
- `<|tool_call_begin|>` - Start of individual tool call
- `<|tool_call_argument_begin|>` - Start of tool arguments
- `<|tool_call_end|>` - End of individual tool call
- `<|tool_calls_section_end|>` - End of tool calls section

**Supported modes**:
- Non-streaming: Parses complete response and injects `tool_calls` array
- Streaming: Accumulates chunks and emits proper SSE chunks with `tool_calls`

#### 2. Anthropic to OpenAI Converter (`anthropic-to-openai`)
Converts Anthropic-style messages to OpenAI format for request transformation.

### Configuring Transformers

Transformers are configured per-model in the config file:

```json
{
  "models": [
    {
      "pattern": "moonshotai/Kimi-K2.5*",
      "transformers": ["kimi-tool-calls"],
      "request_transformers": ["anthropic-to-openai"]
    }
  ]
}
```

## Commands

### Serve

Start the AI proxy server:

```bash
./ai-proxy serve
./ai-proxy serve --port 3000
./ai-proxy serve --config-file /path/to/config.json
```

### Config

Generate opencode configuration from available models:

```bash
./ai-proxy config --proxy-url http://localhost:8080
./ai-proxy config --output ~/.config/opencode/opencode.json
```

## Architecture

See [docs/Design.md](docs/Design.md) for detailed architecture documentation.

## License

MIT License
