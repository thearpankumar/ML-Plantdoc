import serial
import time

# Set up the serial connection (adjust the port and baud rate)
arduino = serial.Serial(port='COM5', baudrate=9600, timeout=.1)

def read_moisture():
    if arduino.in_waiting > 0:
        # Read the line of data from the serial port
        data = arduino.readline().decode('utf-8').rstrip()
        return data
    return None

while True:
    moisture_level = read_moisture()
    if moisture_level:
        # Print the moisture level to the console
        print(f"Soil Moisture Level: {moisture_level}%")
        
        # Open the file in append mode and write just the number to the file
        with open('moisture_levels.txt', 'a') as file:
            file.write(moisture_level + '\n')
            
    time.sleep(1)
