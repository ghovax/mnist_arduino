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
from flask import Flask, render_template, jsonify, send_file, Response
import base64
from io import BytesIO
from queue import Queue
import json
from datetime import datetime
from threading import Lock

# Setup basic logging configuration
logging.basicConfig(level=logging.INFO)

# Initialize Flask app
app = Flask(__name__)

# Ensure artifacts directory exists
os.makedirs("artifacts", exist_ok=True)


# After Flask initialization, add this custom logging handler
class WebConsoleHandler(logging.Handler):
    def __init__(self, max_messages=1000):
        super().__init__()
        self.messages = Queue(maxsize=max_messages)

    def emit(self, record):
        try:
            message = {
                "timestamp": datetime.fromtimestamp(record.created).strftime(
                    "%H:%M:%S"
                ),
                "level": record.levelname,
                "message": self.format(record),
            }
            if self.messages.full():
                self.messages.get()  # Remove oldest message if queue is full
            self.messages.put(message)
        except Exception:
            self.handleError(record)

    def get_messages(self):
        return list(self.messages.queue)


# Initialize the web console handler
web_handler = WebConsoleHandler()
web_handler.setFormatter(logging.Formatter("%(message)s"))
logging.getLogger().addHandler(web_handler)

# Add these after Flask app initialization
serial_connection = None
serial_lock = Lock()


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
    ports = list(serial.tools.list_ports.comports())
    logging.info(f"Found {len(ports)} serial ports")

    for port in ports:
        logging.info(f"Currently checking port {port.device} ({port.description})")
        if "Arduino" in port.description or "usbmodem" in port.device:
            logging.info(
                f"Found Arduino to connect to: {port.description} on {port.device}"
            )

            # Test if port is actually available
            try:
                test_connection = serial.Serial(
                    port.device, baudrate=115200, timeout=0.1
                )
                test_connection.close()
                logging.info("Port is available for connection")
                return port.device
            except serial.SerialException as e:
                logging.warning(f"Port found but not available: {str(e)}")
                continue
        else:
            logging.warning(f"No Arduino found on port {port.device}")

    port_list = [f"{p.device}" for p in ports]
    logging.error(f"No Arduino found. Available ports: {port_list}")
    return None


# Compile the Arduino sketch.
def compile_sketch():
    """Compile the Arduino sketch."""
    sketch_path = "camera/camera.ino"
    logging.info(
        f"Compiling sketch at '{sketch_path}', please wait... it may take up to a few seconds"
    )
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

    elapsed_time = time.time() - start_time

    if result.returncode == 0:
        logging.info(f"Compilation successful (took {elapsed_time:.1f}s)")
        return True
    else:
        error_msg = result.stderr.replace("\n", " ").strip()
        logging.error(f"Compilation failed after {elapsed_time:.1f}s: {error_msg}")
        return False


# Flash the Arduino with the compiled hex file.
def flash_arduino(port, hex_file, timeout=30):
    """Flash the Arduino with the provided hex file."""
    try:
        if not os.path.exists(hex_file):
            logging.error(
                f"Cannot find hex file: '{hex_file}', please compile the sketch first"
            )
            return False

        logging.info(
            f"Flashing Arduino with '{hex_file}'... please wait, the operation will timeout after {timeout}s"
        )
        start_time = time.time()

        cmd = [
            "arduino-cli",
            "upload",
            "-p",
            port,
            "--fqbn",
            "arduino:mbed_nano:nano33ble",
            "--input-dir",
            os.path.dirname(hex_file),  # Use the build directory for input
            "camera",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        elapsed_time = time.time() - start_time

        if result.returncode == 0:
            logging.info(f"Successfully flashed (took {elapsed_time:.1f}s)")
            return True
        else:
            error_message = result.stderr.split("\n")[0]
            logging.error(f"Flash failed after {elapsed_time:.1f}s: {error_message}")
            return False
    except Exception as e:
        elapsed_time = time.time() - start_time
        logging.error(f"Flash failed after {elapsed_time:.1f}s: {str(e)}")
        return False


def receive_image_data(arduino, width=176, height=144, timeout=5):
    """Receive grayscale image data from Arduino."""
    logging.info(f"Waiting for image data... timeout: {timeout}s")

    expected_bytes = width * height
    raw_data = bytearray(expected_bytes)  # Pre-allocate buffer
    received_bytes = 0
    start_time = time.time()

    while received_bytes < expected_bytes:
        if time.time() - start_time > timeout:
            logging.error(f"Timeout after {timeout}s waiting for image data")
            return None

        # Read larger chunks of data
        chunk = arduino.read(min(4096, expected_bytes - received_bytes))
        if not chunk:
            continue

        chunk_size = len(chunk)
        raw_data[received_bytes : received_bytes + chunk_size] = chunk
        received_bytes += chunk_size

        # Log progress for large transfers
        if received_bytes % 8192 == 0:
            logging.info(
                f"Received {received_bytes}/{expected_bytes} bytes ({(received_bytes/expected_bytes)*100:.1f}%)"
            )

    try:
        # Convert raw bytes to numpy array and reshape to 2D
        image_data = np.frombuffer(raw_data, dtype=np.uint8)
        image_data = image_data.reshape((height, width))
        return image_data
    except Exception as e:
        logging.error(f"Failed to process image data: {str(e)}")
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

            # Close existing connection after flashing
            with serial_lock:
                if serial_connection is not None:
                    try:
                        serial_connection.close()
                    except:
                        pass
                    serial_connection = None

        # Get or create serial connection
        arduino = get_serial_connection()
        if not arduino:
            return False, None

        with serial_lock:
            try:
                # Flush any existing data
                arduino.reset_input_buffer()
                arduino.reset_output_buffer()

                # Send capture command
                bytes_written = arduino.write(b"c")
                if bytes_written != 1:
                    logging.error(
                        f"Failed to send capture command (wrote {bytes_written} bytes)"
                    )
                    return False, None

                logging.info("Capture command sent successfully")

                # Receive and process image
                image_data = receive_image_data(arduino, timeout=5)

                if image_data is not None:
                    image = Image.fromarray(image_data, mode="L")
                    image_path = "artifacts/acquired_image.png"
                    image.save(image_path)
                    logging.info(f"Grayscale acquired image saved to '{image_path}'")
                    return True, image

                return False, None

            except Exception as e:
                logging.error(f"Error during capture: {str(e)}")
                # Close connection on error
                try:
                    arduino.close()
                except:
                    pass
                serial_connection = None
                return False, None

    except Exception as e:
        logging.error(f"Error: {str(e)}")
        return False, None


def threshold_image(image, threshold: int = 100):
    """
    Process the image to make lighter pixels transparent and keep darker pixels.

    Args:
        image_path (str): Path to the input image
        output_path (str): Path to save the processed image
        threshold (int): Pixel values below this will be kept (0-255), default 100
    """
    # Read the image
    img_array = np.array(image)

    # Create an RGBA array (same width and height, but 4 channels)
    height, width = img_array.shape
    rgba_array = np.zeros((height, width, 4), dtype=np.uint8)

    # Set RGB channels to black (0)
    rgba_array[..., 0:3] = 0

    # Set alpha channel based on threshold
    # Pixels darker than threshold will be visible (alpha = 255)
    # Pixels lighter than threshold will be transparent (alpha = 0)
    rgba_array[..., 3] = np.where(img_array < threshold, 255, 0)

    # Create PIL image and save
    processed_img = Image.fromarray(rgba_array, mode="RGBA")
    output_path = "artifacts/thresholded_image.png"
    processed_img.save(output_path, format="PNG")
    logging.info(f"Thresholded the image and saved it to '{output_path}'")

    return processed_img


def evaluate_digit(image):
    """Evaluate the processed image using MNIST model to predict the digit."""
    try:
        # Load and compile the model with metrics
        model = tf.keras.models.load_model("mnist_cnn_model.keras")

        # Load and preprocess the image
        img = image.convert("RGBA")  # Convert to RGBA first
        # Create a white background
        background = Image.new("RGBA", img.size, (255, 255, 255, 255))
        # Paste the image using itself as mask (this handles transparency)
        background.paste(img, mask=img)
        # Now convert to grayscale
        img = background.convert("L")
        img = img.resize((28, 28), Image.Resampling.BICUBIC)  # Resize to MNIST format

        # Convert to numpy array and normalize
        img_array = np.array(img, dtype=np.float32)
        # Invert the colors to get white digit on black background
        img_array = 255 - img_array
        img_array = img_array / 255.0  # Normalize to [0,1]

        # Reshape for model input (add batch and channel dimensions)
        img_array = img_array.reshape(1, 28, 28, 1)

        # Save the preprocessed image for debugging
        debug_img = (img_array[0, :, :, 0] * 255).astype(np.uint8)
        debug_img_path = "artifacts/debug_mnist_input.png"
        Image.fromarray(debug_img).save(debug_img_path)

        # Get predictions and probabilities
        probabilities = model.predict(img_array)
        predicted_digit = int(probabilities.argmax(axis=1)[0])
        confidence = float(probabilities[0][predicted_digit])

        # Convert probabilities to list
        all_probabilities = [float(p) for p in probabilities[0]]

        return predicted_digit, confidence, all_probabilities

    except Exception as e:
        logging.error(f"Failed to evaluate image: {str(e)}")
        return None, None, None


@app.route("/")
def index():
    """Render the main page."""
    return render_template("index.html")


@app.route("/capture", methods=["POST"])
def capture():
    """Handle image capture request."""
    global serial_connection

    try:
        with serial_lock:
            if serial_connection is None or not serial_connection.is_open:
                return jsonify({"success": False, "error": "Not connected to Arduino"})

        # Capture image
        success, image = acquire_image(force_compile=False)

        if not success or image is None:
            return jsonify(
                {"success": False, "error": "Failed to acquire image from Arduino"}
            )

        # Process the acquired image
        thresholded_image = threshold_image(image, threshold=100)

        # Evaluate the processed image
        predicted_digit, confidence, probabilities = evaluate_digit(thresholded_image)

        if predicted_digit is None:
            return jsonify({"success": False, "error": "Failed to evaluate image"})

        # Convert images to base64 for sending to frontend
        def img_to_base64(img):
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return f"data:image/png;base64,{img_str}"

        # Load the MNIST debug image
        mnist_debug_img = Image.open("artifacts/debug_mnist_input.png")

        return jsonify(
            {
                "success": True,
                "original_image": img_to_base64(image),
                "threshold_image": img_to_base64(thresholded_image),
                "mnist_image": img_to_base64(mnist_debug_img),
                "predicted_digit": predicted_digit,
                "confidence": confidence,
                "probabilities": probabilities,
            }
        )

    except Exception as e:
        logging.error(f"Error during capture: {str(e)}")
        return jsonify({"success": False, "error": str(e)})


# Add this new route to fetch log messages
@app.route("/logs")
def get_logs():
    """Server-sent events handler for log messages."""

    def generate():
        last_message_count = 0
        while True:
            messages = web_handler.get_messages()
            if len(messages) > last_message_count:
                new_messages = messages[last_message_count:]
                last_message_count = len(messages)
                yield f"data: {json.dumps(new_messages)}\n\n"
            time.sleep(0.1)  # Small delay to prevent busy-waiting

    return Response(generate(), mimetype="text/event-stream")


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
                        "error": "Arduino not found. Please check the connection.",
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

                    return jsonify({"success": True, "port": port})
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
                    {"success": False, "error": f"Failed to connect: {str(e)}"}
                )

    except Exception as e:
        logging.error(f"Connection error: {str(e)}")
        return jsonify({"success": False, "error": str(e)})


if __name__ == "__main__":
    # Run the app on port 5050
    app.run(debug=True, host="0.0.0.0", port=5050)
