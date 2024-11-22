#include <Arduino_OV767X.h>

// Configure camera settings
const int IMAGE_WIDTH = 176;
const int IMAGE_HEIGHT = 144;
const int BYTES_PER_PIXEL = 1;  // Using grayscale
const int LED_PIN = LED_BUILTIN;  // Built-in LED for visual feedback
const int BUTTON_PIN = 13;  // TinyML Shield button

// Debouncing variables
unsigned long lastDebounceTime = 0;
unsigned long debounceDelay = 250;    // Increase debounce delay
bool lastButtonState = HIGH;
bool buttonState = HIGH;

void setup() {
  Serial.begin(115200);
  while (!Serial) {
    ; // Wait for serial port to connect
  }
  
  pinMode(LED_PIN, OUTPUT);
  pinMode(BUTTON_PIN, INPUT_PULLUP);  // Configure button with pull-up resistor
  digitalWrite(LED_PIN, LOW);
  
  Serial.println("Camera initializing...");
  
  // Initialize the camera
  if (!Camera.begin(QCIF, GRAYSCALE, 1)) {
    Serial.println("Failed to initialize camera!");
    while (1);
  }
  
  Serial.println("Camera initialized successfully!");
}

void loop() {
  // Read button with debouncing
  int reading = digitalRead(BUTTON_PIN);

  // If the button state changed, reset debounce timer
  if (reading != lastButtonState) {
    lastDebounceTime = millis();
  }

  // Only act on button press if enough time has passed
  if ((millis() - lastDebounceTime) > debounceDelay) {
    // If button state has changed
    if (reading != buttonState) {
      buttonState = reading;
      
      // Only trigger on button press (LOW due to pull-up)
      if (buttonState == LOW) {
        Serial.println("Button pressed!");
        digitalWrite(LED_PIN, HIGH);  // Visual feedback
        
        // Buffer for one image frame
        byte pixels[IMAGE_WIDTH * IMAGE_HEIGHT];
        
        // Capture frame
        Serial.println("Capturing frame...");
        Camera.readFrame(pixels);
        
        // Send the image data
        Serial.println("Sending image data...");
        delay(100);  // Give Python time to prepare for binary data
        Serial.write((uint8_t*)pixels, IMAGE_WIDTH * IMAGE_HEIGHT);  // Send raw bytes
        Serial.flush();
        delay(100);  // Wait for data to be sent
        
        digitalWrite(LED_PIN, LOW);
        Serial.println("\nImage sent successfully!");
      }
    }
  }
  
  lastButtonState = reading;
  delay(10);  // Small delay to prevent tight loop
} 