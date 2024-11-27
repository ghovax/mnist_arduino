.PHONY: setup run clean install dev

# Python virtual environment
VENV := venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

# Flask application settings
FLASK_APP := app.py
PORT := 5050
HOST := 0.0.0.0

# Default target
all: setup

# Create virtual environment and install dependencies
setup: $(VENV)/bin/activate

$(VENV)/bin/activate:
	python3 -m venv $(VENV)
	$(PIP) install -r requirements.txt

# Run the Flask application
run: setup
	$(PYTHON) $(FLASK_APP)

# Run in development mode with debug
dev: setup
	FLASK_ENV=development $(PYTHON) $(FLASK_APP)

# Clean build artifacts and generated files
clean:
	rm -rf build/
	rm -rf artifacts/
	rm -rf __pycache__/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Clean everything including venv
clean-all: clean
	rm -rf $(VENV)

# Install and configure arduino-cli
install:
	@if ! command -v arduino-cli >/dev/null 2>&1; then \
		curl -fsSL https://raw.githubusercontent.com/arduino/arduino-cli/master/install.sh | sh; \
		arduino-cli core update-index; \
		arduino-cli core install arduino:mbed_nano; \
	fi

# Train MNIST model
train:
	$(PYTHON) mnist_compile.py

# Create required directories
init:
	mkdir -p artifacts
	mkdir -p build

# Help target
help:
	@echo "Available targets:"
	@echo "  setup      - Create virtual environment and install dependencies"
	@echo "  run        - Run the Flask application"
	@echo "  dev        - Run in development mode with debug enabled"
	@echo "  clean      - Remove build artifacts and generated files"
	@echo "  clean-all  - Remove everything including virtual environment"
	@echo "  install    - Install and configure arduino-cli"
	@echo "  train      - Train the MNIST model"
	@echo "  init       - Create required directories"