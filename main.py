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

# ANSI escape codes for colors
class Colors:
    HEADER = '\033[95m'
    INFO = '\033[94m'
    SUCCESS = '\033[92m'
    WARNING = '\033[93m'
    ERROR = '\033[91m'
    RESET = '\033[0m'

class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors and prefixes"""
    
    def format(self, record):
        # Save original levelname
        orig_levelname = record.levelname
        
        # Add color and prefix based on level
        level_colors = {
            logging.ERROR: (Colors.ERROR, "ERROR"),
            logging.WARNING: (Colors.WARNING, "WARNING"), 
            logging.DEBUG: (Colors.SUCCESS, "SUCCESS"),
            logging.CRITICAL: (Colors.HEADER, "ARDUINO"),
        }
        color, name = level_colors.get(record.levelno, (Colors.INFO, "INFO"))
        record.levelname = f"{color}[{name}]{Colors.RESET}"
            
        # Format the message
        result = super().format(record)
        
        # Restore original levelname
        record.levelname = orig_levelname
        return result

def setup_logging():
    """Configure logging format and handlers."""
    logger = logging.getLogger('arduino_flash')
    logger.setLevel(logging.DEBUG)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.DEBUG)
    
    formatter = ColoredFormatter('%(levelname)s %(message)s')
    console.setFormatter(formatter)
    
    logger.addHandler(console)
    return logger

logger = setup_logging()

def setup_argument_parser():
    """Configure command line argument parser."""
    parser = argparse.ArgumentParser(description='Arduino camera image capture utility')
    parser.add_argument('--force-compile', action='store_true',
                       help='Force recompilation of Arduino sketch even if unchanged')
    return parser

# Find the port where Arduino is connected.
def find_arduino_port():
    """Find the port where Arduino is connected."""
    ports = list(serial.tools.list_ports.comports())
    logger.info(f"Found {len(ports)} serial ports")
    
    for port in ports:
        logger.info(f"Currently checking port {port.device} ({port.description})")
        if "Arduino" in port.description or "usbmodem" in port.device:
            logger.debug(f"Found Arduino to connect to: {port.description} on {port.device}")
            
            # Test if port is actually available
            try:
                test_connection = serial.Serial(port.device, baudrate=115200, timeout=1)
                test_connection.close()
                logger.debug("Port is available for connection")
                return port.device
            except serial.SerialException as e:
                logger.warning(f"Port found but not available: {str(e)}")
                continue
        else:
            logger.warning(f"No Arduino found on port {port.device}")
            
    port_list = [f"{p.device}" for p in ports]
    logger.error(f"No Arduino found. Available ports: {port_list}")
    return None

# Compile the Arduino sketch.
def compile_sketch():
    """Compile the Arduino sketch."""
    logger.info("Compiling sketch...")
    start_time = time.time()
    
    build_path = os.path.join(os.getcwd(), "build")  # Create build directory in current working directory
    os.makedirs(build_path, exist_ok=True)  # Ensure build directory exists
    
    cmd = [
        "arduino-cli",
        "compile",
        "--fqbn", "arduino:mbed_nano:nano33ble",
        "--build-path", build_path,
        "camera"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    elapsed_time = time.time() - start_time
    
    if result.returncode == 0:
        logger.debug(f"Compilation successful (took {elapsed_time:.1f}s)")
        return True
    else:
        error_msg = result.stderr.replace('\n', ' ').strip()
        logger.error(f"Compilation failed after {elapsed_time:.1f}s: {error_msg}")
        return False

# Flash the Arduino with the compiled hex file.
def flash_arduino(port, hex_file, timeout=30):
    """Flash the Arduino with the provided hex file."""
    try:
        if not os.path.exists(hex_file):
            logger.error(f"Cannot find {hex_file}, please compile the sketch first")
            return False

        logger.info(f"Flashing Arduino... please wait, it may take up to {timeout}s")
        start_time = time.time()

        cmd = [
            "arduino-cli",
            "upload",
            "-p", port,
            "--fqbn", "arduino:mbed_nano:nano33ble",
            "--input-dir", os.path.dirname(hex_file),  # Use the build directory for input
            "camera"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        elapsed_time = time.time() - start_time
        
        if result.returncode == 0:
            logger.debug(f"Successfully flashed (took {elapsed_time:.1f}s)")
            return True
        else:
            error_message = result.stderr.split('\n')[0]
            logger.error(f"Flash failed after {elapsed_time:.1f}s: {error_message}")
            return False
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"Flash failed after {elapsed_time:.1f}s: {str(e)}")
        return False

def receive_image_data(arduino, width=176, height=144, timeout=10):
    """Receive grayscale image data from Arduino."""
    logger.info(f"Waiting for image data... please wait, it may take up to {timeout}s")
    
    # Calculate expected bytes (1 byte per pixel for grayscale)
    expected_bytes = width * height
    received_bytes = 0
    raw_data = bytearray()
    
    # Set timeout for entire operation (10 seconds)
    start_time = time.time()
    
    while received_bytes < expected_bytes:
        if time.time() - start_time > timeout:
            logger.error(f"Timeout after {timeout}s waiting for image data")
            return None
            
        # Read data in chunks
        chunk = arduino.read(min(1024, expected_bytes - received_bytes))
        if not chunk:  # No data received
            logger.error("No data received from Arduino, aborting")
            return None
            
        raw_data.extend(chunk)
        received_bytes = len(raw_data)
    
    if received_bytes != expected_bytes:
        logger.error(f"Received incomplete data: {received_bytes} vs {expected_bytes} bytes")
        return None
    
    try:
        # Convert raw bytes to numpy array and reshape to 2D
        image_data = np.frombuffer(raw_data, dtype=np.uint8)
        image_data = image_data.reshape((height, width))
        return image_data
    except Exception as e:
        logger.error(f"Failed to process image data: {str(e)}")
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
            reason = "forced by the user" if force_compile else "sketch changed"
            logger.warning(f"Recompiling the Arduino sketch ({reason})...")
            if not compile_sketch():
                return False
            update_cached_sketch()
            
            hex_file = "build/camera.ino.with_bootloader.hex"
            if not flash_arduino(port, hex_file, timeout=60):
                return False
        else:
            logger.warning("Sketch unchanged, skipping compilation and flash")

        logger.info("Attempting to clear serial port...")
        try:
            # Force close any existing connections
            subprocess.run(['killall', '-9', 'screen'], stderr=subprocess.DEVNULL)  # Kill any screen sessions
            time.sleep(0.5)
        except:
            pass

        try:
            logger.info("Attempting to connect...")
            
            # Increase timeout for more reliable data transfer
            arduino = serial.Serial(
                port=port,
                baudrate=115200,
                timeout=5,
                write_timeout=2,
                inter_byte_timeout=1
            )
            
            logger.info("Serial object created, checking if open...")
            
            if not arduino.is_open:
                logger.warning("Port not open, attempting to open...")
                arduino.open()
            
            if arduino.is_open:
                logger.debug("Serial connection established!")
                arduino.reset_input_buffer()
                arduino.reset_output_buffer()
                
                # Add delay after connection to let Arduino stabilize
                logger.warning("Waiting for Arduino to initialize...")
                time.sleep(1)
            else:
                raise serial.SerialException("Failed to open port")
                
        except serial.SerialException as e:
            logger.error(f"Connection failed: {str(e)}")
            try:
                arduino.close()
            except:
                pass
            return False
        
        # Flush any existing data
        arduino.reset_input_buffer()
        arduino.reset_output_buffer()
                
        # Send command and verify it was sent
        bytes_written = arduino.write(b'c')
        if bytes_written != 1:
            logger.error(f"Failed to send capture command (wrote {bytes_written} bytes)")
            arduino.close()
            return False
        
        logger.debug("Capture command sent successfully")
        
        # Receive and process image with longer timeout
        image_data = receive_image_data(arduino, timeout=10)
        
        # Always close the connection
        arduino.close()
        
        if image_data is not None:
            # Save the grayscale image
            image = Image.fromarray(image_data, mode='L')  # 'L' mode for grayscale
            image.save('image.png')
            logger.debug("Grayscale image saved as 'image.png'")
            return True
        else:
            return False

    except KeyboardInterrupt:
        logger.warning("Closing connection...")
        if 'arduino' in locals():
            arduino.close()
        return False
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        if 'arduino' in locals():
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
    img = Image.open(image_path).convert('L')
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
    processed_img = Image.fromarray(rgba_array, mode='RGBA')
    processed_img.save(output_path, format='PNG')
    logger.debug("Processed image saved as 'processed_image.png'")
    
    return processed_img

def main():
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Only process image if acquisition was successful
    if acquire_image(force_compile=args.force_compile):
        # Process the acquired image to make light pixels transparent
        threshold_image('image.png', 'processed_image.png', threshold=100)
    else:
        logger.warning("Skipping image processing due to acquisition failure")

if __name__ == "__main__":
    main()

