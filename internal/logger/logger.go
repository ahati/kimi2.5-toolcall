package logger

import (
	"fmt"
	"os"
	"strings"
	"sync"
)

type Level int

const (
	DebugLevel Level = iota
	InfoLevel
	WarnLevel
	ErrorLevel
)

type Logger struct {
	level  Level
	colors bool
	mu     sync.RWMutex
}

var (
	defaultLogger *Logger
	once          sync.Once
)

func parseLevel(level string) Level {
	switch strings.ToLower(level) {
	case "debug":
		return DebugLevel
	case "info":
		return InfoLevel
	case "warn", "warning":
		return WarnLevel
	case "error":
		return ErrorLevel
	default:
		return InfoLevel
	}
}

func (l Level) String() string {
	switch l {
	case DebugLevel:
		return "DEBUG"
	case InfoLevel:
		return "INFO"
	case WarnLevel:
		return "WARN"
	case ErrorLevel:
		return "ERROR"
	default:
		return "UNKNOWN"
	}
}

func isTerminal() bool {
	// Check if stderr is a terminal
	fileInfo, err := os.Stderr.Stat()
	if err != nil {
		return false
	}
	return fileInfo.Mode()&os.ModeCharDevice != 0
}

func Init(level string) {
	once.Do(func() {
		defaultLogger = &Logger{
			level:  parseLevel(level),
			colors: isTerminal(),
		}
	})
}

func SetLevel(level string) {
	if defaultLogger != nil {
		defaultLogger.mu.Lock()
		defaultLogger.level = parseLevel(level)
		defaultLogger.mu.Unlock()
	}
}

func Debugf(format string, args ...interface{}) { logf(DebugLevel, format, args...) }
func Infof(format string, args ...interface{})  { logf(InfoLevel, format, args...) }
func Warnf(format string, args ...interface{})  { logf(WarnLevel, format, args...) }
func Errorf(format string, args ...interface{}) { logf(ErrorLevel, format, args...) }

func logf(level Level, format string, args ...interface{}) {
	if defaultLogger == nil {
		defaultLogger = &Logger{level: InfoLevel, colors: isTerminal()}
	}

	defaultLogger.mu.RLock()
	configuredLevel := defaultLogger.level
	colors := defaultLogger.colors
	defaultLogger.mu.RUnlock()

	if level < configuredLevel {
		return
	}

	prefix := ""
	if colors {
		switch level {
		case DebugLevel:
			prefix = "\033[90m[DEBUG]\033[0m " // Gray
		case InfoLevel:
			prefix = "\033[32m[INFO]\033[0m " // Green
		case WarnLevel:
			prefix = "\033[33m[WARN]\033[0m " // Yellow
		case ErrorLevel:
			prefix = "\033[31m[ERROR]\033[0m " // Red
		}
	} else {
		prefix = fmt.Sprintf("[%s] ", level.String())
	}

	fmt.Fprintf(os.Stderr, "%s%s\n", prefix, fmt.Sprintf(format, args...))
}
