import serial #Serial imported for Serial communication
import time #Required to use delay functions

ArduinoSerial = serial.Serial("COM7",9600) #Create Serial port object called arduinoSerialData
time.sleep(2) #wait for 2 secounds for the communication to get established

 #read the serial data and print it as line
#while True:

num = str(ArduinoSerial.readline())
print(num)
print(num[1:-4])

