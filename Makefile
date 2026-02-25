# AI-Proxy Makefile
# Install binary and config to standard Unix locations

PREFIX ?= $(HOME)/.local
XDG_CONFIG_DIR ?= $(HOME)/.config
OPENCODE_CONFIG_DIR = $(XDG_CONFIG_DIR)/opencode

BINDIR = $(PREFIX)/bin
CONFIGDIR = $(XDG_CONFIG_DIR)/ai-proxy
CONFIGFILE = $(CONFIGDIR)/ai-proxy-config.json

BINARY = ai-proxy
GO ?= go

.PHONY: all build install install-opencode-config uninstall clean help

all: build

build:
	$(GO) build -o $(BINARY)

install:
	@echo "Installing $(BINARY) to $(BINDIR)..."
	@mkdir -p $(BINDIR)
	@cp $(BINARY) $(BINDIR)/$(BINARY)
	@chmod 755 $(BINDIR)/$(BINARY)
	@echo "Installing config to $(CONFIGFILE)..."
	@mkdir -p $(CONFIGDIR)
	@cp ai-proxy-config.json $(CONFIGFILE)
	@chmod 644 $(CONFIGFILE)
	@echo "Installation complete!"
	@echo ""
	@echo "Next steps:"
	@echo "1. Edit $(CONFIGFILE) to update your API keys and provider settings"
	@echo "2. Set API keys as environment variables (see README)"
	@echo "3. Run: $(BINDIR)/$(BINARY) serve"

install-opencode-config:
	@echo "Installing opencode config to $(OPENCODE_CONFIG_DIR)..."
	@mkdir -p $(OPENCODE_CONFIG_DIR)
	@cp opencode-config.json $(OPENCODE_CONFIG_DIR)/opencode.json
	@chmod 644 $(OPENCODE_CONFIG_DIR)/opencode.json
	@echo "Opencode config installed to $(OPENCODE_CONFIG_DIR)/opencode.json"

uninstall:
	@echo "Uninstalling $(BINARY)..."
	@rm -f $(BINDIR)/$(BINARY)
	@echo "Removing config from $(CONFIGFILE)..."
	@rm -f $(CONFIGFILE)
	@rm -rf $(CONFIGDIR)
	@echo "Uninstallation complete!"

clean:
	@echo "Cleaning build artifacts..."
	@rm -f $(BINARY)
	@$(GO) clean

help:
	@echo "AI-Proxy Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  build                 - Build the binary"
	@echo "  install               - Build and install to $(PREFIX) (default: $$HOME/.local)"
	@echo "  install-opencode-config - Install opencode config to $$HOME/.config/opencode"
	@echo "  uninstall             - Remove binary and config"
	@echo "  clean                 - Remove build artifacts"
	@echo "  help                  - Show this help message"
	@echo ""
	@echo "Environment variables:"
	@echo "  PREFIX       - Installation prefix (default: \$$HOME/.local)"
	@echo "  XDG_CONFIG_DIR - Config directory (default: \$$HOME/.config)"
	@echo ""
	@echo "Example:"
	@echo "  make install PREFIX=/opt/ai-proxy"
