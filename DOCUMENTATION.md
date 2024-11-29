# Arduino Camera MNIST System Documentation

## Overview
This system integrates an Arduino Nano 33 BLE with an OV7670 camera module to capture images, process them, and use a trained MNIST model to recognize handwritten digits. The system consists of a Flask backend server that communicates with the Arduino and processes images, and can be integrated with a frontend interface.

## System Architecture

### Core Components
1. Flask Backend Server (app.py)
2. Arduino with Camera Module (camera.ino)
3. MNIST Model (mnist_cnn_model.keras)
4. Serial Communication Interface

## Hardware Requirements
- Arduino Nano 33 BLE
- OV7670 Camera Module
- USB Connection to Host Computer

## Software Requirements
- Python 3.x
- arduino-cli
- Required Python packages:
  - Flask
  - PySerial
  - NumPy
  - PIL (Python Imaging Library)
  - TensorFlow

## Detailed Component Documentation

### 1. Global Variables and Setup
- `serial_connection`: Maintains the global serial connection to Arduino
- `serial_lock`: Thread-safe lock for serial communication
- `log_queue`: Queue for storing logging messages
- `log_lock`: Thread-safe lock for logging operations

### 2. Command Line Arguments
The application accepts the following command line arguments:
- `--force-compile`: Forces recompilation of Arduino sketch regardless of changes

### 3. Arduino Communication Functions

#### Port Detection
`find_arduino_port()`
- Purpose: Automatically detects the Arduino's serial port
- Process:
  - Scans all available serial ports
  - Identifies Arduino by "Arduino" or "usbmodem" in port description
  - Tests connection before confirming
- Returns: Port device path or None

#### Serial Connection Management
`get_serial_connection()`
- Purpose: Manages serial connection to Arduino
- Features:
  - Thread-safe connection handling
  - Automatic port detection
  - Connection testing
  - Buffer clearing
- Settings:
  - Baud rate: 115200
  - Timeout: 2 seconds
  - Write timeout: 1 second

### 4. Arduino Sketch Management

#### Compilation System
`compile_sketch()`
- Handles Arduino sketch compilation
- Uses arduino-cli
- Creates build directory
- Returns compilation success status

#### Flashing System
`flash_arduino(port, hex_file, timeout=30)`
- Uploads compiled code to Arduino
- Verifies hex file existence
- Handles upload errors
- Times out after 30 seconds by default

#### Change Detection System
- `has_sketch_changed()`: Detects modifications to Arduino code
- `get_file_hash()`: Generates SHA-256 hash of sketch
- `update_cached_sketch()`: Maintains sketch version control

### 5. Image Processing Pipeline

#### Capture System
`acquire_image(force_compile=False)`
- Orchestrates image capture process
- Handles:
  - Sketch compilation if needed
  - Arduino communication
  - Error recovery
- Returns: (success_boolean, image_object)

#### Raw Data Processing
`receive_image_data(arduino_connection, width=176, height=144, timeout=5)`
- Handles raw image data reception
- Features:
  - Timeout protection
  - Buffer management
  - Data verification
- Image specifications:
  - Width: 176 pixels
  - Height: 144 pixels
  - Format: Grayscale

#### Image Enhancement
`threshold_image(input_image, threshold: int = 100)`
- Processes raw camera input
- Creates transparency mask
- Enhances digit visibility
- Returns RGBA format image

### 6. MNIST Integration

#### Digit Recognition System
`evaluate_digit(input_image)`
- Loads and uses MNIST model
- Image preprocessing:
  - Resizing to 28x28
  - Normalization
  - Color inversion
- Returns:
  - Predicted digit
  - Confidence score
  - Probability distribution

### 7. Web API Endpoints

#### Capture Endpoint
`POST /capture`
- Triggers image capture
- Processes image
- Performs digit recognition
- Returns:
  - Base64 encoded image
  - Recognition results
  - Confidence scores
  - System logs

#### Connection Endpoint
`POST /connect`
- Establishes Arduino connection
- Performs connection testing
- Returns connection status

### 8. Logging System

#### Web Console Handler
- Custom logging implementation
- Features:
  - Timestamp recording
  - Log level tracking
  - Queue-based message handling
- Thread-safe operation

### 9. Error Handling

#### Recovery Systems
- Serial connection recovery
- Arduino reconnection
- Resource cleanup
- Exception handling

#### Resource Management
- Automatic connection cleanup
- Memory management
- Thread safety
- Port release

## Usage Instructions

### Initial Setup
1. Install required software:
   ```bash
   pip install -r requirements.txt
   ```
2. Install arduino-cli
3. Connect Arduino Nano 33 BLE
4. Ensure MNIST model file is present

### Running the System
1. Start the Flask server:
   ```bash
   python app.py
   ```
2. Connect to Arduino through the API
3. Begin capturing and processing images

### Troubleshooting
- Check Arduino connection if capture fails
- Verify serial port permissions
- Monitor system logs for errors
- Ensure proper lighting for camera

## Best Practices
- Regular testing of camera alignment
- Proper lighting conditions
- Regular Arduino sketch updates
- System logs monitoring

## Security Considerations
- Serial port access control
- API endpoint protection
- Resource limitation
- Error handling

## Performance Optimization
- Thread-safe operations
- Efficient image processing
- Memory management
- Connection pooling

## Maintenance
- Regular sketch updates
- System log review
- Connection testing
- Model updates

This documentation provides a comprehensive overview of the system's architecture, functionality, and usage. For specific implementation details, refer to the source code comments.