const int LED_PIN = 13;
const int BUTTON_PIN = 2;  // TinyML Shield button is on pin 2

void setup() {
  pinMode(LED_PIN, OUTPUT);
  pinMode(BUTTON_PIN, INPUT_PULLUP);  // Configure button pin with internal pull-up resistor
}

void loop() {
  // Read button state (LOW when pressed because of pull-up resistor)
  if (digitalRead(BUTTON_PIN) == LOW) {
    digitalWrite(LED_PIN, HIGH);  // Turn LED on when button is pressed
  } else {
    digitalWrite(LED_PIN, LOW);   // Turn LED off when button is released
  }
}