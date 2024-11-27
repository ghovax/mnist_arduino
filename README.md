# Arduino Camera MNIST Digit Recognition

A web-based application that interfaces with an Arduino Nano 33 BLE and camera module to capture, process, and recognize handwritten digits using a trained MNIST CNN model.

## Features

- Real-time camera capture and digit recognition
- Web-based interface with live console output
- Interactive probability visualization
- Automatic Arduino connection management
- Image preprocessing pipeline
- Smart sketch compilation system
- Real-time status monitoring

## Prerequisites

- Python 3.9+
- Arduino Nano 33 BLE board
- Compatible camera module
- arduino-cli installed and configured
- Modern web browser

## Installation

1. Clone this repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Initialize required directories:
```bash
make init
```

3. Create and activate a virtual environment:
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Unix or MacOS:
source venv/bin/activate
```

4. Install required Python packages:
```bash
make setup
```

5. Install arduino-cli (if not already installed):
```bash
make install
```

## Usage

### Available Make Commands

```bash
make help        # Show all available commands
make setup       # Create virtual environment and install dependencies
make run         # Run the Flask application
make dev         # Run in development mode with debug enabled
make clean       # Remove build artifacts and generated files
make clean-all   # Remove everything including virtual environment
make install     # Install and configure arduino-cli
make train       # Train the MNIST model
make init        # Create required directories
```

### Running the Application

1. Start the application:
```bash
make run    # Regular mode
# or
make dev    # Development mode with debug enabled
```
The server will start on `http://localhost:5050`

2. Access the web interface:
   - Open your browser and navigate to `http://localhost:5050`
   - Click "Connect to Arduino" to establish connection
   - Use "Capture Image" to take pictures and process them

## Web Interface Features

- **Connection Status**: Real-time Arduino connection status indicator
- **Console Output**: Live logging of system operations
- **Image Display**: Shows captured images in real-time
- **Probability Histogram**: Visual representation of digit predictions

## Image Processing Pipeline

1. Raw image capture from Arduino camera
2. Grayscale conversion and thresholding
3. MNIST format conversion (28x28 pixels)
4. CNN model prediction
5. Probability distribution visualization

## Project Structure
```
├── app.py              # Main Flask application
├── mnist_compile.py    # MNIST model training script
├── requirements.txt    # Python dependencies
├── templates/         
│   ├── index.html     # Main web interface
├── artifacts/         # Generated images and data
└── camera/           # Arduino sketch files
```

## Model Training and Testing

### Training the Model
To retrain the MNIST model:
```bash
make train
```

### Testing the Model
The application includes a dedicated web interface for testing the MNIST model:

1. Start the testing interface:
```bash
python mnist_compile_app.py
```
The test server will start on `http://localhost:5100`

2. Access the testing interface:
   - Open your browser and navigate to `http://localhost:5100`
   - Use the drawing canvas to draw digits
   - Click "Predict" to see the model's prediction
   - Click "Clear" to reset the canvas

### Testing Features
- **Interactive Drawing Canvas**: Draw digits using your mouse
- **Real-time Prediction**: Instant digit recognition
- **Probability Visualization**: Bar chart showing confidence for each digit
- **Image Processing Preview**: 
  - Original grayscale conversion
  - MNIST format (28x28) preview
- **Confidence Score**: Displays prediction confidence percentage

This testing interface is particularly useful for:
- Verifying model accuracy
- Understanding the preprocessing pipeline
- Testing model performance with different writing styles
- Debugging recognition issues

## Troubleshooting

1. **Arduino Connection Issues**
   - Verify the Arduino is properly connected
   - Check if the correct port is detected
   - Try unplugging and reconnecting the device
   - Ensure no other program is using the serial port

2. **Image Capture Problems**
   - Verify camera module connection
   - Check lighting conditions
   - Try forcing a sketch recompilation:
     ```bash
     python app.py --force-compile
     ```
   - Examine console logs for errors

3. **Recognition Issues**
   - Ensure proper image positioning
   - Check image preprocessing results in artifacts/
   - Verify model loading success in console

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- TensorFlow team for MNIST model support
- Arduino community for hardware interface
- Flask team for web framework