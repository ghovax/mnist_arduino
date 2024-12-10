# Arduino MNIST Recognition

A real-time handwritten digit recognition system using Arduino Nano 33 BLE Sense's camera and TensorFlow for digit classification.

## Features

- Real-time image capture using Arduino Nano 33 BLE Sense camera
- TensorFlow-based MNIST model for digit recognition
- Next.js web interface for real-time visualization
- Flask backend for Arduino communication and image processing
- Live probability distribution visualization
- Real-time system status and error reporting

## Prerequisites

- Arduino Nano 33 BLE Sense
- Python 3.9+
- Node.js 18+
- arduino-cli

## Installation

1. Clone the repository
2. Install dependencies with `make install`
3. Run the development server with `make run`
4. Navigate to the directory `interface` and check out the [README](interface/README.md) for information on running the Next.js web interface
5. Connect to the Arduino and start recognizing digits at the link displayed in the terminal!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
