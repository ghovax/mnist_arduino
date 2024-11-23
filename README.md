# Arduino Camera Interface

Python interface for capturing images from an Arduino Nano 33 BLE's camera module.

## Prerequisites

- Python 3.7+
- Arduino Nano 33 BLE
- Arduino Camera module OV7675 (or any other compatible module)
- TinyML Shield (optional, for more convenient connections)

## Quick Start

1. First clean any previous builds and images: `make clean`

2. Install arduino-cli and required cores: `make install`

3. Set up Python environment and dependencies: `make setup`

4. Run the program: `make run`

## Make Commands

- `make install` - Install arduino-cli and required board support
- `make setup` - Create Python environment and install dependencies
- `make run` - Run the camera interface program
- `make clean` - Remove build artifacts and generated images
- `make clean-all` - Remove everything including virtual environment

## Expected Output

When running successfully, the program will:
1. Find and connect to your Arduino
2. Compile and flash the camera sketch
3. Capture an image and save it as `image.png` in the current directory

## Troubleshooting

If no Arduino is found, check:
- USB connection
- Whether the Arduino shows up in `arduino-cli board list`
- Your user has permission to access the USB port