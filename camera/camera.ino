#include <Arduino_OV767X.h>

// Camera settings
#define IMAGE_WIDTH 176
#define IMAGE_HEIGHT 144
#define BYTES_PER_PIXEL 1
#define BYTES_PER_FRAME (IMAGE_WIDTH * IMAGE_HEIGHT * BYTES_PER_PIXEL)

const int LED_PIN = 13;

// Buffer for one frame
uint8_t frame_buffer[BYTES_PER_FRAME];

void setup() {
  // Initialize serial communication
  Serial.begin(115200);
  while (!Serial) {
    ; // Wait for serial port to connect
  }

  // Initialize the camera in grayscale mode
  if (!Camera.begin(QCIF, GRAYSCALE, 1)) {
    // Fail silently, not a big deal
    while (1);
  }
}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read();
    
    if (command == 'c') {  // Capture command
      // Capture frame into buffer
      Camera.readFrame(frame_buffer);
      
      // Send the frame over serial
      Serial.write(frame_buffer, BYTES_PER_FRAME);
      
      // Flush the serial buffer
      Serial.flush();
    }
  }
} 