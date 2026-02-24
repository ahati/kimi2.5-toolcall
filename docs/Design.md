# AI-Proxy Design Document

## Overview

AI-Proxy is a generic protocol converter and proxy server designed to sit between AI clients and upstream AI service providers. It supports multiple upstream providers simultaneously, each with their own API keys, and can apply various transformations to requests and responses.

The primary use cases include:
1. **Multi-provider routing**: Route requests to different upstream providers based on model names
2. **Protocol conversion**: Convert Anthropic-style messages to OpenAI format
3. **Response transformation**: Fix malformed responses (e.g., Kimi K2.5 tool call tokens)
4. **Unified interface**: Provide a single OpenAI-compatible endpoint for multiple backends

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌──────────────────┐
│   Client    │────▶│   AI-Proxy  │────▶│  Upstream 1      │
│             │     │             │     │  (chutes.ai)     │
│             │     │  - Router   │     └──────────────────┘
│             │     │  - Transform│     ┌──────────────────┐
│             │     │  - Proxy    │────▶│  Upstream 2      │
│             │     │             │     │  (minimax.io)    │
└─────────────┘     └─────────────┘     └──────────────────┘
```

## Core Components

### 1. Configuration System

#### Configuration Sources (Priority Order)

1. **CLI flags** (highest priority)
2. **Custom config file** (`--config-file` argument)
3. **XDG data directory** (`$XDG_DATA_DIR/ai-proxy-config.json`)
4. **Working directory** (`./ai-proxy-config.json`)
5. **Default values** (lowest priority)

#### Configuration File Format

```json
{
  "port": 8080,
  "log_level": "info",
  "timeout": "600s",
  
  "transformers": {
    "kimi-tool-calls": {
      "enabled": true,
      "tokens": {
        "section_begin": "<|tool_calls_section_begin|>",
        "section_end": "<|tool_calls_section_end|>",
        "call_begin": "<|tool_call_begin|>",
        "call_end": "<|tool_call_end|>",
        "arg_begin": "<|tool_call_argument_begin|>"
      }
    },
    "anthropic-to-openai": {
      "enabled": true
    }
  },
  
  "upstreams": [
    {
      "name": "chutes",
      "provider": "chutes.ai",
      "endpoint": "https://api.chutes.ai/v1",
      "api_key": "sk-chutes-xxx",
      "models": [
        {
          "pattern": "moonshotai/Kimi-K2.5*",
          "transformers": ["kimi-tool-calls"],
          "request_transformers": ["anthropic-to-openai"]
        },
        {
          "pattern": "moonshotai/*",
          "transformers": [],
          "request_transformers": []
        }
      ]
    }
  ]
}
```

#### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--port` | 8080 | Listen port |
| `--config-file` | "" | Path to JSON config file |
| `--log-level` | info | Logging level (debug, info, warn, error) |

### 2. Router

The router determines which upstream provider should handle a request based on the model name.

#### Model Name Format

```
<provider>/<path>/<model>
```

Examples:
- `chutes.ai/moonshotai/Kimi-K2.5-TEE`
- `minimax.io/abab6.5s-chat`

#### Routing Algorithm

1. **Extract provider**: Take the first segment of the model name (everything before the first `/`)
2. **Match upstream**: Find the upstream configuration where `provider` field equals the extracted provider
3. **Match model pattern**: Within the matched upstream, find the model pattern that matches the remaining model path
   - Patterns support wildcards (`*` matches any string)
   - Patterns are matched in order of specificity (longer patterns first)
4. **Strip prefix**: Remove the provider prefix before forwarding to upstream

#### Example Routing

Given model `chutes.ai/moonshotai/Kimi-K2.5-TEE`:
1. Extract provider: `chutes.ai`
2. Match upstream: Find upstream with `provider: "chutes.ai"`
3. Match model: Pattern `moonshotai/Kimi-K2.5*` matches `moonshotai/Kimi-K2.5-TEE`
4. Transform: Apply transformers `["kimi-tool-calls"]`
5. Forward: Send to upstream with model `moonshotai/Kimi-K2.5-TEE`

### 3. Transformer System

Transformers modify requests and responses. They are pluggable and can be applied per-model.

#### Transformer Interface

```go
// RequestTransformer transforms incoming requests before sending to upstream
type RequestTransformer interface {
    Name() string
    TransformRequest(body []byte, model string) ([]byte, error)
}

// ResponseTransformer transforms upstream responses before sending to client
type ResponseTransformer interface {
    Name() string
    TransformResponse(body []byte, isStreaming bool) ([]byte, error)
}
```

For streaming responses, a more specialized interface:

```go
// StreamingTransformer handles chunk-by-chunk transformation
type StreamingTransformer interface {
    ResponseTransformer
    // TransformStream processes a single SSE chunk
    // Returns: (wasModified, modifiedChunk, shouldKeepOriginal)
    TransformStream(chunk []byte) (modified bool, newChunk []byte, keepChunk bool)
}
```

#### Built-in Transformers

##### 3.1 Kimi Tool Calls Fixer

**Purpose**: Convert raw tool call tokens emitted by Kimi K2.5 into proper OpenAI `tool_calls` format.

**Tokens handled**:
- `<|tool_calls_section_begin|>` - Start of tool calls section
- `<|tool_call_begin|>` - Start of individual tool call
- `<|tool_call_argument_begin|>` - Start of tool arguments
- `<|tool_call_end|>` - End of individual tool call
- `<|tool_calls_section_end|>` - End of tool calls section

**Non-streaming transformation**:
1. Parse tokens from `reasoning_content` and `content` fields
2. Remove tokens from the text fields
3. Create proper `tool_calls` array with:
   - `id`: Generated unique call ID
   - `type`: "function"
   - `function.name`: Function name (stripped of namespace prefix)
   - `function.arguments`: JSON arguments string
4. Set `finish_reason` to "tool_calls"

**Streaming transformation**:
1. Accumulate chunks in a state machine
2. Detect token boundaries even across chunk boundaries
3. When section ends, emit proper tool_call delta chunks
4. Emit `finish_reason: "tool_calls"` chunk

##### 3.2 Anthropic to OpenAI Converter

**Purpose**: Convert Anthropic-format messages to OpenAI chat completions format.

**Request transformation** (input: Anthropic `/messages` format → output: OpenAI `/chat/completions`):

| Anthropic Field | OpenAI Field | Notes |
|-----------------|--------------|-------|
| `model` | `model` | Pass through |
| `messages` | `messages` | Convert content blocks |
| `system` | `messages[0].content` (system) | Convert system prompts |
| `tools` | `tools` | Convert tool definitions |
| `tool_choice` | `tool_choice` | Pass through |
| `max_tokens` | Not applicable | Anthropic-specific |
| `temperature` | `temperature` | Pass through |
| `top_p` | `top_p` | Pass through |
| `stream` | `stream` | Pass through |

**Content block conversion**:
- `type: "text"` → OpenAI content string
- `type: "image"` → `type: "image_url"` with base64 data
- `type: "tool_use"` → Not included in request (handled via tools)
- `type: "tool_result"` → `type: "tool"` in messages

### 4. HTTP Handlers

#### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/v1/chat/completions` | OpenAI chat completions |
| POST | `/v1/anthropic/messages` | Anthropic messages (converted) |
| GET | `/v1/models` | List models (aggregated from all upstreams) |
| GET | `/health` | Health check |

#### /v1/models Implementation

The `/v1/models` endpoint aggregates models from all configured upstreams by querying each upstream's `/models` endpoint and combining the results.

**Response format** (OpenAI-compatible):
```json
{
  "object": "list",
  "data": [
    {
      "id": "chutes.ai/moonshotai/kimi-k2.5",
      "object": "model",
      "created": 1700000000,
      "owned_by": "moonshotai"
    },
    {
      "id": "minimax.io/abab6.5s-chat",
      "object": "model",
      "created": 1700000000,
      "owned_by": "minimax"
    }
  ]
}
```

**Model ID format**: `<provider>/<original_model_id>`

**Notes**:
- If an upstream doesn't support `/models` endpoint, configured model patterns are used as fallback
- Models are prefixed with their provider to ensure uniqueness

#### Handler Flow

```
Request ──▶ Parse Body ──▶ Route to Upstream
               │                  │
               │                  ▼
               │            Apply Request Transformers
               │                  │
               │                  ▼
               │            Forward to Upstream
               │                  │
               │                  ▼
               │            Apply Response Transformers
               │                  │
               ▼                  ▼
            Return Response to Client
```

### 5. Upstream Client

The upstream client handles HTTP communication with backend providers.

**Features**:
- Per-upstream HTTP clients with configured timeouts
- Automatic retry on transient failures
- Connection pooling
- Request/response logging

**Authentication**:
- Client-provided API keys are **ignored**
- Upstream's configured `api_key` is always used
- `Authorization: Bearer <api_key>` header added automatically

## Data Structures

### Config Structure

```go
type Config struct {
    Port        int               `json:"port"`
    LogLevel    string            `json:"log_level"`
    Timeout     string            `json:"timeout"`
    Transformers map[string]TransformerConfig `json:"transformers"`
    Upstreams   []UpstreamConfig  `json:"upstreams"`
}

type UpstreamConfig struct {
    Name     string         `json:"name"`
    Provider string         `json:"provider"`
    Endpoint string         `json:"endpoint"`
    APIKey   string         `json:"api_key"`
    Models   []ModelConfig `json:"models"`
}

type ModelConfig struct {
    Pattern            string   `json:"pattern"`
    Transformers       []string `json:"transformers"`
    RequestTransformers []string `json:"request_transformers"`
}

type TransformerConfig map[string]interface{}
```

### Request/Response Types

#### OpenAI Chat Completions

**Request**:
```json
{
  "model": "chutes.ai/moonshotai/Kimi-K2.5-TEE",
  "messages": [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello"}
  ],
  "stream": false
}
```

**Response**:
```json
{
  "id": "chatcmpl-xxx",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "chutes.ai/moonshotai/Kimi-K2.5-TEE",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello!"
      },
      "finish_reason": "stop"
    }
  ]
}
```

#### Anthropic Messages

**Request**:
```json
{
  "model": "claude-3-opus-20240229",
  "max_tokens": 1024,
  "messages": [
    {"role": "user", "content": "Hello"}
  ]
}
```

**Transformed to OpenAI**:
```json
{
  "model": "claude-3-opus-20240229",
  "messages": [
    {"role": "user", "content": "Hello"}
  ]
}
```

## Error Handling

### Error Response Format

All errors follow OpenAI's error format:

```json
{
  "error": {
    "message": "Error description",
    "type": "invalid_request_error",
    "param": "model",
    "code": "invalid_model"
  }
}
```

### Error Types

| Type | Code | Description |
|------|------|-------------|
| `invalid_request_error` | invalid_model | Model not found in any upstream |
| `invalid_request_error` | invalid_upstream | Upstream provider not configured |
| `invalid_request_error` | transform_error | Request/response transformation failed |
| `upstream_error` | upstream_timeout | Upstream request timed out |
| `upstream_error` | upstream_error | Upstream returned an error |
| `server_error` | internal_error | Internal proxy error |

## Metrics

AI-Proxy provides comprehensive metrics tracking for monitoring performance and usage.

### Metrics Endpoint

| Method | Path | Description |
|--------|------|-------------|
| GET | `/metrics` | Returns all metrics in JSON format |

### Latency Metrics

Latency is tracked for every request and reported in milliseconds:

**Per-Request Metrics:**
- `current_ms`: Latency of the most recent request
- `rolling_avg_ms`: Average latency across all requests
- `last_10_avg_ms`: Average latency of the last 10 requests
- `total_count`: Total number of requests

**Available at three levels:**
1. **Global**: All requests across all providers and models
2. **Per Provider**: Aggregated by upstream provider
3. **Per Model**: Detailed per-model metrics

### TPS (Tokens Per Second) Counters

Token usage is tracked for both input and output:

**Tracked per:**
- **Provider**: Total tokens per upstream provider
- **Model**: Detailed token usage per model

**Metrics include:**
- `input_tokens`: Total input tokens processed
- `output_tokens`: Total output tokens generated
- `requests`: Total number of requests
- `last_updated_unix`: Timestamp of last update

### Metrics Response Format

```json
{
  "global_latency": {
    "current_ms": 1234,
    "rolling_avg_ms": 1150.5,
    "last_10_avg_ms": 1200.3,
    "total_count": 42
  },
  "models": [
    {
      "model": "moonshotai/Kimi-K2.5-TEE",
      "provider": "chutes.ai",
      "latency": {
        "current_ms": 1234,
        "rolling_avg_ms": 1150.5,
        "last_10_avg_ms": 1200.3,
        "total_count": 15
      },
      "tps": {
        "input_tokens": 15000,
        "output_tokens": 8500,
        "requests": 15,
        "last_updated_unix": 1704067200
      }
    }
  ],
  "providers": [
    {
      "provider": "chutes.ai",
      "latency": {
        "current_ms": 1234,
        "rolling_avg_ms": 1150.5,
        "last_10_avg_ms": 1200.3,
        "total_count": 42
      },
      "tps": {
        "input_tokens": 45000,
        "output_tokens": 28000,
        "requests": 42,
        "last_updated_unix": 1704067200
      }
    }
  ]
}
```

## Environment Variable API Keys

API keys can be loaded from environment variables for better security and flexibility.

### Configuration Format

Instead of hardcoding API keys, use the `ENV:` prefix:

```json
{
  "upstreams": [
    {
      "name": "chutes",
      "provider": "chutes.ai",
      "endpoint": "https://api.chutes.ai/v1",
      "api_key": "ENV:CHUTES_API_KEY",
      "models": [...]
    }
  ]
}
```

### Environment Variable Loading

At startup, the proxy will resolve environment variable references:

```bash
export CHUTES_API_KEY="sk-chutes-xxx"
./ai-proxy serve
```

**Notes:**
- If the environment variable is not set, the original value is used as-is
- This allows mixing static and dynamic API key configurations
- The `ENV:` prefix is case-insensitive

## Logging

- **Log levels**: debug, info, warn, error
- **Log format**: `timestamp [level] message`
- **Structured logging**: JSON format for machine parsing (optional)

### Key Log Events

- Request received: `method path model`
- Upstream selected: `provider endpoint model`
- Transformation applied: `transformer_name direction`
- Error: `error_type error_message`

## Testing Strategy

### Unit Tests

- Configuration loading and merging
- Router model matching
- Transformer logic (token parsing, format conversion)

### Integration Tests

- End-to-end proxy with mock upstream
- Streaming response handling
- Error propagation

### Test Patterns

See `/workspaces/kimi-k2.5-fix-proxy/claude-proxy/tests/` for reference test payloads.

## Future Enhancements

1. **Additional transformers**:
   - OpenAI to Anthropic response conversion
   - Token counting and rate limiting
   - Response caching

2. **Load balancing**:
   - Round-robin across multiple upstreams for same provider
   - Health checking and failover

3. **Metrics**:
   - Request latency histograms
   - Token usage tracking
   - Error rate monitoring

4. **Security**:
   - API key rotation
   - Request validation
   - Rate limiting per client

## Reference Implementation

The original Python implementation is available at:
`/workspaces/kimi-k2.5-fix-proxy/kimi-k2.5-tool-fix-proxy/`

Key differences from Python version:
- Uses Cobra for CLI instead of environment variables
- Uses Gin for HTTP server instead of FastAPI
- Pluggable transformer architecture
- Multi-upstream support
- Anthropic message conversion
