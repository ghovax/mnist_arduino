#include <Arduino_OV767X.h>

// Camera settings
#define IMAGE_WIDTH 176
#define IMAGE_HEIGHT 144
#define BYTES_PER_PIXEL 1
#define ACTUAL_WIDTH 175  // Ignore the last column
#define BYTES_PER_FRAME (ACTUAL_WIDTH * IMAGE_HEIGHT * BYTES_PER_PIXEL)

const int LED_PIN = 13;

// Buffer for one frame
uint8_t frame_buffer[IMAGE_WIDTH * IMAGE_HEIGHT];
uint8_t cleaned_buffer[BYTES_PER_FRAME];

void setup() {
  // Initialize serial communication
  Serial.begin(115200);
  while (!Serial) {
    ; // Wait for serial port to connect
  } 

  pinMode(LED_PIN, OUTPUT);

  // Initialize the camera in grayscale mode
  if (!Camera.begin(QCIF, GRAYSCALE, 1)) {
    // Fail silently, not a big deal
    while (1);
  }
}

void loop() {
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    
    if (cmd == 'c') {  // Capture command
      // Capture frame into buffer
      Camera.readFrame(frame_buffer);
      
      // Copy frame data excluding the last column
      for (int y = 0; y < IMAGE_HEIGHT; y++) {
        for (int x = 0; x < ACTUAL_WIDTH; x++) {
          cleaned_buffer[y * ACTUAL_WIDTH + x] = frame_buffer[y * IMAGE_WIDTH + x];
        }
      }
      
      // Send the cleaned frame over serial
      Serial.write(cleaned_buffer, BYTES_PER_FRAME);
      
      // Flush the serial buffer
      Serial.flush();
    }
  }

  digitalWrite(LED_PIN, HIGH);
} 