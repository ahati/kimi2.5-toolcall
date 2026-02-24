# AI-Proxy Agent Guidelines

This file provides guidance for AI agents working on the AI-Proxy codebase.

## Build Commands

```bash
# Build the binary
go build -o ai-proxy

# Run the server
go run . serve
./ai-proxy serve

# Install dependencies
go mod tidy
go mod download

# Clean build
go clean && go build -o ai-proxy
```

## Test Commands

```bash
# Run all tests
go test ./...

# Run tests with coverage
go test -cover ./...

# Run tests for a specific package
go test ./internal/proxy
go test ./internal/metrics

# Run a single test function
go test -run TestFunctionName ./internal/proxy

# Run tests with verbose output
go test -v ./...

# Run tests and generate coverage report
go test -coverprofile=coverage.out ./...
go tool cover -html=coverage.out -o coverage.html
```

## Lint Commands

```bash
# Format code
go fmt ./...

# Vet code for common issues
go vet ./...

# Install and run golint (if not already installed)
go install golang.org/x/lint/golint@latest
golint ./...

# Run all checks
go fmt ./... && go vet ./... && go test ./...
```

## Code Style Guidelines

### Imports

Group imports in three sections with blank lines between:
1. Standard library packages
2. Third-party packages
3. Local project packages

```go
import (
	"encoding/json"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/spf13/cobra"

	"ai-proxy/internal/config"
	"ai-proxy/internal/metrics"
)
```

### Naming Conventions

- **Packages**: Lowercase, single word (e.g., `metrics`, `handler`, `proxy`)
- **Types**: PascalCase for exported (e.g., `Handler`, `Metrics`), camelCase for internal
- **Functions**: PascalCase for exported, camelCase for unexported
- **Variables**: camelCase (e.g., `modelName`, `proxyConfig`)
- **Constants**: PascalCase or camelCase (avoid ALL_CAPS)
- **Interfaces**: Use `-er` suffix (e.g., `ResponseTransformer`, `RequestTransformer`)
- **Acronyms**: Keep uppercase (e.g., `HTTPClient`, `APIKey`, `URL`)

### Error Handling

Always check errors and handle them explicitly:

```go
// Early return pattern
resp, err := httpClient.Do(req)
if err != nil {
	c.JSON(http.StatusBadGateway, NewErrorResponse("upstream_error", err.Error()))
	return
}
defer resp.Body.Close()

// Error wrapping for context
if err := validateConfig(cfg); err != nil {
	return fmt.Errorf("failed to validate config: %w", err)
}
```

### Struct Tags

Use `mapstructure` for config structs and `json` for API structs:

```go
type Config struct {
	Port     int           `mapstructure:"port"`
	Timeout  time.Duration `mapstructure:"timeout"`
}

type ErrorResponse struct {
	Error ErrorDetail `json:"error"`
}
```

### Comments

- Export all public types and functions with doc comments
- Start with the name of the item being documented
- Use complete sentences

```go
// RouteResult contains the result of a routing decision.
type RouteResult struct {
	Upstream *config.UpstreamConfig
}

// Route determines which upstream provider should handle a request.
func (p *Proxy) Route(model string) (*RouteResult, error) {
```

## Project Structure

```
/workspaces/kimi-k2.5-fix-proxy/ai-proxy/
├── main.go                    # Entry point
├── cmd/                       # CLI commands
│   ├── config/               # Config generation command
│   └── server/               # Server command
├── internal/                  # Internal packages
│   ├── config/               # Configuration management
│   ├── handler/              # HTTP handlers
│   ├── metrics/              # Metrics collection
│   ├── proxy/                # Proxy routing logic
│   ├── transformer/          # Request/response transformers
│   └── upstream/             # Upstream provider clients
└── docs/                     # Documentation
```

## Testing Guidelines

- Test files should be named `<package>_test.go`
- Place tests in the same package or `package <name>_test` for black-box testing
- Use table-driven tests for multiple test cases
- Use `t.Parallel()` for parallel test execution
- Mock external dependencies using interfaces

Example test pattern:

```go
func TestRoute(t *testing.T) {
	tests := []struct {
		name    string
		model   string
		wantErr bool
	}{
		{"valid model", "chutes.ai/model", false},
		{"invalid model", "invalid", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// test implementation
		})
	}
}
```

## Common Patterns

### Handler Pattern

Handlers receive dependencies via constructor:

```go
type Handler struct {
	proxy   *proxy.Proxy
	config  *config.Config
	metrics *metrics.Metrics
}

func New(p *proxy.Proxy, cfg *config.Config, m *metrics.Metrics) *Handler {
	return &Handler{proxy: p, config: cfg, metrics: m}
}
```

### Error Types

Define custom errors as exported variables:

```go
type ProxyError struct {
	Code    string
	Message string
}

func (e *ProxyError) Error() string {
	return e.Message
}

var (
	ErrUpstreamNotFound = &ProxyError{Code: "upstream_not_found", Message: "..."}
)
```

## Dependencies

Key dependencies in this project:
- `github.com/gin-gonic/gin` - HTTP web framework
- `github.com/spf13/cobra` - CLI framework
- `github.com/spf13/viper` - Configuration management
- `github.com/google/uuid` - UUID generation

Go version: 1.21+
