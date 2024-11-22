import serial
import serial.tools.list_ports
import time
import subprocess
import sys
import os

def find_arduino_port():
    """Find the port where Arduino is connected."""
    ports = list(serial.tools.list_ports.comports())
    print("\nScanning ports...")
    if not ports:
        print("No ports found!")
        return None
        
    for port in ports:
        print(f"Found port: {port.device}")
        print(f"   Description: {port.description}")
        print(f"   Hardware ID: {port.hwid}")
        print("---")
        
        if "Arduino" in port.description or "usbmodem" in port.device:
            return port.device
    return None

def flash_arduino(port, hex_file):
    """Flash the Arduino with the provided hex file."""
    try:
        cmd = [
            "avrdude",
            "-p", "atmega328p",          # Processor type (for Uno/Nano)
            "-c", "arduino",             # Programmer type
            "-P", port,                  # Port
            "-b", "115200",             # Baud rate
            "-U", f"flash:w:{hex_file}:i"  # Flash operation
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print("Successfully flashed Arduino!")
            return True
        else:
            print(f"Error flashing Arduino: {result.stderr}")
            return False
    except Exception as e:
        print(f"Error during flashing: {e}")
        return False

def main():
    # Try to find Arduino port automatically
    port = find_arduino_port()
    if not port:
        print("Arduino not found! Please check connection and try again.")
        print("Available ports:")
        for p in serial.tools.list_ports.comports():
            print(f"  - {p.device}: {p.description}")
        return

    print(f"Found Arduino on port: {port}")
    
    # Basic blink program hex file
    hex_file = "build/blink.ino.hex"  # Update this line
    
    if flash_arduino(port, hex_file):
        try:
            arduino = serial.Serial(port=port, baudrate=9600, timeout=1)
            time.sleep(2)  # Wait for the serial connection to initialize
            
            print("Connected successfully to Arduino yassss!")
            print("Press Ctrl+C to exit...")
            
            while True:
                print("Arduino is running...")
                time.sleep(1)
                
        except serial.SerialException as e:
            print(f"Error opening serial port: {e}")
            print("Try running: sudo chmod 666", port)
        except KeyboardInterrupt:
            print("\nClosing connection to Arduino...")
            arduino.close()
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
