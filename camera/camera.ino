#include <Arduino_OV767X.h>

// Camera settings
#define IMAGE_WIDTH 176
#define IMAGE_HEIGHT 144
#define BYTES_PER_PIXEL 1
#define BYTES_PER_FRAME (IMAGE_WIDTH * IMAGE_HEIGHT * BYTES_PER_PIXEL)

const int LED_PIN = 13;
const int BUTTON_PIN = 2;  // TinyML Shield button is on pin 2

// Buffer for one frame
uint8_t frame_buffer[BYTES_PER_FRAME];

void setup() {
  // Initialize serial communication
  Serial.begin(115200);
  while (!Serial) {
    ; // Wait for serial port to connect
  } 

  pinMode(LED_PIN, OUTPUT);
  pinMode(BUTTON_PIN, INPUT_PULLUP);  // Configure button pin with internal pull-up resistor 
  digitalWrite(LED_PIN, LOW);  // Start with LED on

  // Initialize the camera in grayscale mode
  if (!Camera.begin(QCIF, GRAYSCALE, 1)) {
    Serial.println("Failed to initialize camera!");
    while (1);
  }
  
  // Optionally set the camera parameters after initialization
  // Camera.setBrightness(3);
  // Camera.setContrast(2);
}

void loop() {
  // Read button state (LOW when pressed because of pull-up resistor)
  if (digitalRead(BUTTON_PIN) == LOW) {
    digitalWrite(LED_PIN, HIGH);   // Turn LED on when button is pressed
  } else {
    digitalWrite(LED_PIN, LOW);    // Turn LED off when button is released
  }

  if (Serial.available() > 0) {
    char cmd = Serial.read();
    
    if (cmd == 'c') {  // Capture command
      // Capture frame into buffer
      Camera.readFrame(frame_buffer);
      
      // Send the entire frame over serial
      Serial.write(frame_buffer, BYTES_PER_FRAME);
      
      // Flush the serial buffer
      Serial.flush();
    }
  }
} 