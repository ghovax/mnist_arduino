.PHONY: setup run clean install

# Python virtual environment
VENV := venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

# Default target
all: setup

# Create virtual environment and install dependencies
setup: $(VENV)/bin/activate

$(VENV)/bin/activate:
	python3 -m venv $(VENV)
	$(PIP) install -r requirements.txt

# Run the main program
run: setup
	$(PYTHON) main.py

# Clean build artifacts and images
clean:
	rm -rf build/
	rm -f image.png
	rm -rf __pycache__/

# Clean everything including venv
clean-all: clean
	rm -rf $(VENV)

# Install arduino-cli if not present
install:
	@if ! command -v arduino-cli >/dev/null 2>&1; then \
		curl -fsSL https://raw.githubusercontent.com/arduino/arduino-cli/master/install.sh | sh; \
		arduino-cli core update-index; \
		arduino-cli core install arduino:mbed_nano; \
	fi 