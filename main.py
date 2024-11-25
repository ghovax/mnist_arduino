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
import matplotlib.pyplot as plt

# Setup basic logging configuration
logging.basicConfig(
    level=logging.INFO,
)


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
                test_connection = serial.Serial(port.device, baudrate=115200, timeout=1)
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


def receive_image_data(arduino, width=175, height=144, timeout=10):
    """Receive grayscale image data from Arduino."""
    logging.info(
        f"Waiting for image data... please wait, the operation will timeout after {timeout}s"
    )

    # Calculate expected bytes (1 byte per pixel for grayscale)
    expected_bytes = width * height
    received_bytes = 0
    raw_data = bytearray()

    # Set timeout for entire operation (10 seconds)
    start_time = time.time()

    while received_bytes < expected_bytes:
        if time.time() - start_time > timeout:
            logging.error(f"Timeout after {timeout}s waiting for image data")
            return None

        # Read data in chunks
        chunk = arduino.read(min(1024, expected_bytes - received_bytes))
        if not chunk:  # No data received
            logging.error("No data received from Arduino, aborting")
            return None

        raw_data.extend(chunk)
        received_bytes = len(raw_data)

    if received_bytes != expected_bytes:
        logging.error(
            f"Received incomplete data: {received_bytes} vs {expected_bytes} bytes"
        )
        return None

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


def acquire_image(force_compile=False):
    try:
        port = find_arduino_port()
        if not port:
            return False  # Return False if no Arduino found

        needs_flashing = has_sketch_changed() or force_compile

        if needs_flashing:
            sketch_path = "camera/camera.ino"
            reason = "forced by the user" if force_compile else "sketch changed"
            logging.warning(
                f"Recompiling the Arduino sketch at '{sketch_path}' ({reason})..."
            )
            if not compile_sketch():
                return False
            update_cached_sketch()

            hex_file = "build/camera.ino.with_bootloader.hex"
            if not flash_arduino(port, hex_file, timeout=60):
                return False
        else:
            logging.warning("Sketch unchanged, skipping compilation and flash")

        logging.info("Attempting to clear serial port...")
        try:
            # Force close any existing connections
            subprocess.run(
                ["killall", "-9", "screen"], stderr=subprocess.DEVNULL
            )  # Kill any screen sessions
            time.sleep(0.5)
        except:
            pass

        try:
            logging.info("Attempting to connect...")

            # Increase timeout for more reliable data transfer
            arduino = serial.Serial(
                port=port,
                baudrate=115200,
                timeout=5,
                write_timeout=2,
                inter_byte_timeout=1,
            )

            logging.info("Serial object created, checking if open...")

            if not arduino.is_open:
                logging.warning("Port not open, attempting to open...")
                arduino.open()

            if arduino.is_open:
                logging.info("Serial connection established!")
                arduino.reset_input_buffer()
                arduino.reset_output_buffer()

                # Add delay after connection to let Arduino stabilize
                logging.warning("Waiting for Arduino to initialize...")
                time.sleep(1)
            else:
                raise serial.SerialException("Failed to open port")

        except serial.SerialException as e:
            logging.error(f"Connection failed: {str(e)}")
            try:
                arduino.close()
            except:
                pass
            return False

        # Flush any existing data
        arduino.reset_input_buffer()
        arduino.reset_output_buffer()

        # Send command and verify it was sent
        bytes_written = arduino.write(b"c")
        if bytes_written != 1:
            logging.error(
                f"Failed to send capture command (wrote {bytes_written} bytes)"
            )
            arduino.close()
            return False

        logging.info("Capture command sent successfully")

        # Receive and process image with longer timeout
        image_data = receive_image_data(arduino, timeout=10)

        # Always close the connection
        arduino.close()
        image_path = "artifacts/image.png"
        if image_data is not None:
            # Save the grayscale image
            image = Image.fromarray(image_data, mode="L")  # 'L' mode for grayscale
            image.save(image_path)
            logging.info(f"Grayscale acquired image saved to '{image_path}'")
            return True
        else:
            return False

    except KeyboardInterrupt:
        logging.warning("Closing connection...")
        if "arduino" in locals():
            arduino.close()
        return False
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        if "arduino" in locals():
            arduino.close()
        return False


def threshold_image(image_path: str, output_path: str, threshold: int = 100):
    """
    Process the image to make lighter pixels transparent and keep darker pixels.

    Args:
        image_path (str): Path to the input image
        output_path (str): Path to save the processed image
        threshold (int): Pixel values below this will be kept (0-255), default 100
    """
    # Read the image
    img = Image.open(image_path).convert("L")
    img_array = np.array(img)

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
    processed_img.save(output_path, format="PNG")
    logging.info(f"Thresholded the image and saved it to '{output_path}'")

    return processed_img


def evaluate_digit(image_path):
    """Evaluate the processed image using MNIST model to predict the digit."""
    try:
        # Load and compile the model with metrics
        model = tf.keras.models.load_model("mnist_model.keras")

        # Load and preprocess the image
        img = Image.open(image_path).convert("RGBA")  # Convert to RGBA first
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

        # Save the preprocessed image for debugging and reopen it just to be sure (redundancy is good)
        debug_img = (img_array[0, :, :, 0] * 255).astype(np.uint8)
        debug_img_path = "artifacts/debug_mnist_input.png"
        Image.fromarray(debug_img).save(debug_img_path)

        img = Image.open(debug_img_path).convert("L")
        image_array = np.array(img)
        # Get predictions and probabilities
        probabilities = model.predict(image_array.reshape(1, 28, 28))
        pred = probabilities.argmax(axis=1)
        plt.imshow(image_array, cmap="gray")
        plt.title(
            f"Predicted the digit {pred[0]} with confidence {probabilities[0][pred[0]]:.2%}"
        )
        plt.show()

    except Exception as e:
        logging.error(f"Failed to evaluate image: {str(e)}")


def main():
    parser = setup_argument_parser()
    args = parser.parse_args()

    # Only process image if acquisition was successful
    if acquire_image(force_compile=args.force_compile):
        # Process the acquired image to make light pixels transparent
        threshold_image(
            "artifacts/image.png", "artifacts/processed_image.png", threshold=100
        )

        # Evaluate the processed image
        evaluate_digit("artifacts/processed_image.png")
    else:
        logging.warning("Skipping image processing due to acquisition failure")


if __name__ == "__main__":
    main()
