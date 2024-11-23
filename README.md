# Arduino camera interface

This Python utility interfaces with an Arduino Nano 33 BLE to capture and process images from an attached camera module.

## Features

- Automatic Arduino detection and connection
- Smart compilation - only recompiles when sketch changes
- Image capture and processing
- Converts captured images to transparent PNGs with dark pixels preserved

## Prerequisites

- Python 3.9+
- arduino-cli installed and configured
- Arduino Nano 33 BLE board
- Required Python packages (see requirements.txt)

## Installation

1. Clone this repository
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Basic usage:
```bash
python main.py
```

### Command line arguments

- `--force-compile`: Forces recompilation of the Arduino sketch even if unchanged (recommended if the image is not as expected, or even if an error occurs)
  ```bash
  python main.py --force-compile
  ```

### How it works

1. The program first looks for a connected Arduino Nano 33 BLE
2. It checks if the Arduino sketch has been modified since last run
   - A cached copy of the sketch is kept in the build directory
   - Only recompiles if the sketch has changed or `--force-compile` is used
3. If needed, compiles and uploads the sketch to the Arduino
4. Captures an image from the camera
5. Processes the image to create a transparent PNG where:
   - Dark pixels are preserved
   - Light pixels become transparent

### Output files

- `image.png`: Raw grayscale image from camera
- `processed_image.png`: Processed image with transparency

## Troubleshooting

If you encounter issues:
1. Ensure Arduino is properly connected
2. Check if arduino-cli is installed and configured
3. Try forcing a recompilation with `--force-compile`
4. Check the serial port permissions
5. If connection fails, try unplugging and reconnecting the Arduino
6. Make sure no other programs are using the Arduino's serial port
