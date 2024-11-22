import serial
import serial.tools.list_ports
import time
import subprocess
import os
import logging
import sys
import numpy as np
from PIL import Image
import select
import cv2

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

def find_arduino_port():
    """Find the port where Arduino is connected."""
    ports = list(serial.tools.list_ports.comports())
    for port in ports:
        if "Arduino" in port.description or "usbmodem" in port.device:
            logger.debug(f"Found Arduino to connect to: {port.description} on {port.device}")
            return port.device
            
    port_list = [f"{p.device}: {p.description}" for p in ports]
    logger.error(f"No Arduino found. Available ports: {', '.join(port_list)}")
    return None

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

def flash_arduino(port, hex_file):
    """Flash the Arduino with the provided hex file."""
    try:
        if not os.path.exists(hex_file):
            logger.error(f"Cannot find {hex_file}")
            return False

        logger.info("Flashing Arduino...")
        start_time = time.time()

        cmd = [
            "arduino-cli",
            "upload",
            "-p", port,
            "--fqbn", "arduino:mbed_nano:nano33ble",
            "--input-dir", os.path.dirname(hex_file),  # Use the build directory for input
            "camera"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
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

def read_camera_data(port):
    """Read a single image frame from the Arduino and save it."""
    logger.info("Capturing image...")
    
    frame_width = 176
    frame_height = 144
    frame_size = frame_width * frame_height
    
    try:
        # First read any debug messages until we get to the actual image data
        while True:
            line = port.readline().decode('utf-8', errors='ignore').strip()
            if line:
                logger.critical(f"{line}")
            if "Sending image data..." in line:
                break
        
        time.sleep(0.1)  # Wait for Arduino to prepare data
        port.reset_input_buffer()  # Clear any leftover data
        
        # Read the raw frame data
        logger.info("Receiving image data from Arduino...")
        raw_data = port.read(frame_size)
        logger.debug(f"Successfully received {len(raw_data)} bytes of image data")
            
        if len(raw_data) != frame_size:
            logger.error(f"Incomplete image data: received {len(raw_data)} of {frame_size} bytes")
            return
            
        # Convert to numpy array and reshape
        frame = np.frombuffer(raw_data, dtype=np.uint8)
        frame = frame.reshape((frame_height, frame_width))
        
        # Save the grayscale image
        cv2.imwrite("image.png", frame)
        logger.debug("Saved image to file")
        
        # Read the final message
        while True:
            line = port.readline().decode('utf-8', errors='ignore').strip()
            if line:
                logger.critical(f"{line}")
                if "Image sent successfully!" in line:
                    break
            
    except Exception as e:
        logger.error(f"Error capturing image: {e}")
        port.close()
        time.sleep(1)
        port.open()

def main():
    try:
        port = find_arduino_port()
        if not port:
            return

        if not compile_sketch():
            return

        hex_file = "build/camera.ino.with_bootloader.hex"
        
        if flash_arduino(port, hex_file):
            # Use higher baudrate for image data
            arduino = serial.Serial(port=port, baudrate=115200, timeout=5)
            time.sleep(2)  # Wait for Arduino to reset
            
            logger.info("Arduino connected! Press Ctrl+C to exit")
            logger.info("Press the TinyML Shield button to capture an image...")
            
            # Clear any startup messages
            while arduino.in_waiting:
                line = arduino.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    logger.critical(f"{line}")
            
            while True:
                # Check for any debug messages
                if arduino.in_waiting:
                    line = arduino.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        logger.critical(f"{line}")
                        if "Button pressed!" in line:
                            read_camera_data(arduino)
                
                time.sleep(0.1)  # Small delay to prevent tight CPU usage
                    
    except KeyboardInterrupt:
        logger.info("Closing connection...")
        if 'arduino' in locals():
            arduino.close()
    except Exception as e:
        logger.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
