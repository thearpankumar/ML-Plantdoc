// Define the analog pin where the sensor is connected
const int sensorPin = A0;  // Assuming the sensor is connected to analog pin A0
int sensorValue = 0;

void setup() {
  // Start serial communication at 9600 baud rate
  Serial.begin(9600);
}

void loop() {
  // Read the value from the sensor
  sensorValue = analogRead(sensorPin);
  
  // Convert the sensor value to a percentage (0% - 100%)
  int moisturePercentage = map(sensorValue, 0, 1023, 0, 100);
  
  // Send the moisture level as a percentage to the serial port
  Serial.println(moisturePercentage);
  
  // Wait for a second before taking the next reading
  delay(1000);
}
