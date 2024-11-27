import serial
import serial.tools.list_ports
import time
import subprocess
import os
import logging
import sys
import numpy as np
from PIL import Image
import hashlib
import shutil
import argparse
import tensorflow as tf
from flask import Flask, jsonify
import json
from datetime import datetime
from threading import Lock
from io import BytesIO
import base64
from queue import Queue
import threading

# Simplify logging to only essential messages
logger = logging.getLogger()

# Initialize Flask app
app = Flask(__name__)

# Add these after Flask app initialization
serial_connection = None
serial_lock = Lock()
log_queue = Queue()
log_lock = Lock()


def setup_argument_parser():
    """Configure command line argument parser."""
    parser = argparse.ArgumentParser(description="Arduino camera image capture utility")
    parser.add_argument(
        "--force-compile",
        action="store_true",
        help="Force recompilation of Arduino sketch even if unchanged",
    )
    return parser


# Find the port where Arduino is connected.
def find_arduino_port():
    """Find the port where Arduino is connected."""
    available_ports = list(serial.tools.list_ports.comports())

    for port_info in available_ports:
        if "Arduino" in port_info.description or "usbmodem" in port_info.device:
            try:
                test_connection = serial.Serial(
                    port_info.device, baudrate=115200, timeout=0.1
                )
                test_connection.close()
                return port_info.device
            except serial.SerialException:
                continue

    logging.error("No Arduino found on any available ports")
    return None


# Compile the Arduino sketch.
def compile_sketch():
    """Compile the Arduino sketch."""
    sketch_path = "camera/camera.ino"
    start_time = time.time()
    build_path = os.path.join(os.getcwd(), "build")
    os.makedirs(build_path, exist_ok=True)

    cmd = [
        "arduino-cli",
        "compile",
        "--fqbn",
        "arduino:mbed_nano:nano33ble",
        "--build-path",
        build_path,
        "camera",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logging.error(f"Compilation failed: {result.stderr.strip()}")
        return False
    return True


# Flash the Arduino with the compiled hex file.
def flash_arduino(port, hex_file, timeout=30):
    """Flash the Arduino with the provided hex file."""
    try:
        if not os.path.exists(hex_file):
            logging.error("Cannot find hex file")
            return False

        cmd = [
            "arduino-cli",
            "upload",
            "-p",
            port,
            "--fqbn",
            "arduino:mbed_nano:nano33ble",
            "--input-dir",
            os.path.dirname(hex_file),
            "camera",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logging.error(f"Flash failed: {result.stderr}".split("\n")[0])
            return False
        return True
    except Exception as e:
        logging.error(f"Flash failed: {str(e)}")
        return False


def receive_image_data(arduino_connection, width=176, height=144, timeout=5):
    """Receive grayscale image data from Arduino."""
    expected_bytes = width * height
    image_buffer = bytearray(expected_bytes)
    received_bytes = 0
    start_time = time.time()

    while received_bytes < expected_bytes:
        if time.time() - start_time > timeout:
            logging.error("Timeout waiting for image data")
            return None

        data_chunk = arduino_connection.read(min(4096, expected_bytes - received_bytes))
        if not data_chunk:
            continue

        chunk_size = len(data_chunk)
        image_buffer[received_bytes : received_bytes + chunk_size] = data_chunk
        received_bytes += chunk_size

    try:
        image_array = np.frombuffer(image_buffer, dtype=np.uint8)
        return image_array.reshape((height, width))
    except Exception as error:
        logging.error(f"Failed to process image data: {str(error)}")
        return None


def get_file_hash(file_path):
    """Calculate SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def has_sketch_changed():
    """Check if the Arduino sketch has changed since last compilation."""
    sketch_path = "camera/camera.ino"
    cached_sketch = "build/camera.ino.cached"

    # If no cached version exists, sketch has effectively changed
    if not os.path.exists(cached_sketch):
        return True

    current_hash = get_file_hash(sketch_path)
    cached_hash = get_file_hash(cached_sketch)

    return current_hash != cached_hash


def update_cached_sketch():
    """Update the cached version of the sketch."""
    sketch_path = "camera/camera.ino"
    cached_sketch = "build/camera.ino.cached"
    os.makedirs(os.path.dirname(cached_sketch), exist_ok=True)
    shutil.copy2(sketch_path, cached_sketch)


# Add this new function to manage the serial connection
def get_serial_connection():
    """Get or create serial connection to Arduino."""
    global serial_connection

    with serial_lock:
        if serial_connection is not None and serial_connection.is_open:
            return serial_connection

        port = find_arduino_port()
        if not port:
            return None

        try:
            connection = serial.Serial(
                port=port,
                baudrate=115200,
                timeout=2,
                write_timeout=1,
                inter_byte_timeout=None,
            )

            if not connection.is_open:
                connection.open()

            if connection.is_open:
                logging.info("Serial connection established!")
                connection.reset_input_buffer()
                connection.reset_output_buffer()
                time.sleep(0.5)
                serial_connection = connection
                return connection

        except serial.SerialException as e:
            logging.error(f"Connection failed: {str(e)}")
            try:
                connection.close()
            except:
                pass
            return None

    return None


# Modify acquire_image function
def acquire_image(force_compile=False):
    """Acquire image from Arduino camera."""
    global serial_connection

    try:
        needs_flashing = has_sketch_changed() or force_compile

        if needs_flashing:
            sketch_path = "camera/camera.ino"
            reason = "forced by the user" if force_compile else "sketch changed"
            logging.warning(
                f"Recompiling the Arduino sketch at '{sketch_path}' ({reason})..."
            )
            if not compile_sketch():
                return False, None
            update_cached_sketch()

            hex_file = "build/camera.ino.with_bootloader.hex"
            port = find_arduino_port()
            if not port or not flash_arduino(port, hex_file, timeout=30):
                return False, None

            with serial_lock:
                if serial_connection is not None:
                    try:
                        serial_connection.close()
                    except:
                        pass
                    serial_connection = None

        arduino = get_serial_connection()
        if not arduino:
            return False, None

        with serial_lock:
            try:
                arduino.reset_input_buffer()
                arduino.reset_output_buffer()

                bytes_written = arduino.write(b"c")
                if bytes_written != 1:
                    logging.error(
                        f"Failed to send capture command (wrote {bytes_written} bytes)"
                    )
                    return False, None

                image_data = receive_image_data(arduino, timeout=5)

                if image_data is not None:
                    # Remove the last column of pixels
                    image_data = image_data[:, :-2]
                    image = Image.fromarray(image_data, mode="L")
                    return True, image

                return False, None

            except Exception as error:
                logging.error(f"Error during capture: {str(error)}")
                try:
                    arduino.close()
                except:
                    pass
                serial_connection = None
                return False, None

    except Exception as error:
        logging.error(f"Error: {str(error)}")
        return False, None


def threshold_image(input_image, threshold: int = 100):
    """Process the image to make lighter pixels transparent and keep darker pixels."""
    input_array = np.array(input_image)
    height, width = input_array.shape
    output_array = np.zeros((height, width, 4), dtype=np.uint8)
    output_array[..., 0:3] = 0
    output_array[..., 3] = np.where(input_array < threshold, 255, 0)

    processed_image = Image.fromarray(output_array, mode="RGBA")
    return processed_image


def evaluate_digit(input_image):
    """Evaluate the processed image using MNIST model to predict the digit."""
    try:
        # Load and compile the model with metrics
        model = tf.keras.models.load_model("mnist_cnn_model.keras")

        # Load and preprocess the image
        processed_image = input_image.convert("RGBA")

        # Create a white background
        background = Image.new("RGBA", processed_image.size, (255, 255, 255, 255))
        # Paste the image using itself as mask
        background.paste(processed_image, mask=processed_image)
        # Now convert to grayscale
        processed_image = background.convert("L")
        processed_image = processed_image.resize((28, 28), Image.Resampling.BICUBIC)

        # Convert to numpy array and normalize
        image_array = np.array(processed_image, dtype=np.float32)
        # Invert the colors to get white digit on black background
        image_array = 255 - image_array
        image_array = image_array / 255.0

        # Reshape for model input (add batch and channel dimensions)
        image_array = image_array.reshape(
            (1, 28, 28, 1)
        )  # Ensure correct shape (batch, height, width, channels)

        # Get predictions
        probabilities = model.predict(image_array, verbose=0)  # Reduce TF logging noise
        predicted_digit = int(np.argmax(probabilities[0]))
        confidence = float(probabilities[0][predicted_digit])

        # Convert probabilities to list
        all_probabilities = [float(probability) for probability in probabilities[0]]

        return predicted_digit, confidence, all_probabilities

    except Exception as error:
        logging.error(f"Failed to evaluate image: {str(error)}")
        return None, None, None


class WebConsoleHandler(logging.Handler):
    def emit(self, record):
        with log_lock:
            log_entry = {
                "level": record.levelname,
                "message": self.format(record),
                "timestamp": datetime.now().isoformat(),
            }
            log_queue.put(log_entry)


# Modify the logging setup
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger()
web_handler = WebConsoleHandler()
logger.addHandler(web_handler)


@app.route("/capture", methods=["POST"])
def capture():
    """Handle image capture request."""
    global serial_connection
    logs = []

    try:
        # Collect any pending logs
        while not log_queue.empty():
            logs.append(log_queue.get_nowait())

        with serial_lock:
            if serial_connection is None or not serial_connection.is_open:
                return jsonify(
                    {
                        "success": False,
                        "errorMessage": "Not connected to Arduino",
                        "logs": logs,
                    }
                )

        success, image = acquire_image(force_compile=False)

        # Collect any new logs from the acquisition
        while not log_queue.empty():
            logs.append(log_queue.get_nowait())

        if not success or image is None:
            return jsonify(
                {
                    "success": False,
                    "errorMessage": "Failed to acquire image from Arduino",
                    "logs": logs,
                }
            )

        thresholded_image = threshold_image(image, threshold=100)
        predicted_digit, confidence, probabilities = evaluate_digit(thresholded_image)

        # Collect any final logs
        while not log_queue.empty():
            logs.append(log_queue.get_nowait())

        if predicted_digit is None:
            return jsonify(
                {
                    "success": False,
                    "errorMessage": "Failed to evaluate image",
                    "logs": logs,
                }
            )

        buffered = BytesIO()
        image.save(buffered, format="PNG")
        encoded_image = base64.b64encode(buffered.getvalue()).decode()

        return jsonify(
            {
                "success": True,
                "originalImage": f"data:image/png;base64,{encoded_image}",
                "predictedDigit": predicted_digit,
                "confidence": confidence,
                "probabilities": probabilities,
                "logs": logs,
            }
        )

    except Exception as e:
        logging.error(f"Error during capture: {str(e)}")
        # Collect any error logs
        while not log_queue.empty():
            logs.append(log_queue.get_nowait())
        return jsonify({"success": False, "errorMessage": str(e), "logs": logs})


# Add cleanup function and register it to run on shutdown
def cleanup():
    """Clean up resources on shutdown."""
    global serial_connection
    with serial_lock:
        if serial_connection is not None:
            try:
                serial_connection.close()
                logging.info("Closed serial connection")
            except:
                pass
            serial_connection = None


import atexit

atexit.register(cleanup)


@app.route("/connect", methods=["POST"])
def connect_arduino():
    """Handle Arduino connection request."""
    global serial_connection

    try:
        with serial_lock:
            # Close existing connection if any
            if serial_connection is not None and serial_connection.is_open:
                try:
                    serial_connection.close()
                except:
                    pass
                serial_connection = None

            port = find_arduino_port()
            if not port:
                return jsonify(
                    {
                        "success": False,
                        "errorMessage": "Arduino not found. Please check the connection.",
                    }
                )

            try:
                connection = serial.Serial(
                    port=port,
                    baudrate=115200,
                    timeout=2,
                    write_timeout=1,
                    inter_byte_timeout=None,
                )

                if not connection.is_open:
                    connection.open()

                if connection.is_open:
                    logging.info(f"Serial connection established on port {port}")
                    connection.reset_input_buffer()
                    connection.reset_output_buffer()
                    time.sleep(0.5)  # Give the Arduino time to reset
                    serial_connection = connection

                    return jsonify({"success": True, "portName": port})
                else:
                    raise serial.SerialException("Failed to open port")

            except serial.SerialException as e:
                logging.error(f"Connection failed: {str(e)}")
                try:
                    if connection:
                        connection.close()
                except:
                    pass
                return jsonify(
                    {"success": False, "errorMessage": f"Failed to connect: {str(e)}"}
                )

    except Exception as e:
        logging.error(f"Connection error: {str(e)}")
        return jsonify({"success": False, "errorMessage": str(e)})


if __name__ == "__main__":
    # Run the app on port 5050
    app.run(host="0.0.0.0", port=5050)
