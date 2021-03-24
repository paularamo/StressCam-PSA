import serial
import time, sys
import datetime
import subprocess

i=0
time_found=False
response=''
SERIAL_PORT="/dev/ttyACM0"
ser=serial.Serial(SERIAL_PORT, baudrate = 9600, timeout = 15)

print(ser)

while time_found==False:
        ser.write('AT+CCLK?\r')
        response = ser.readline()
        
        while "CCLK:" not in response:
                response=ser.readline()
                time.sleep(0.2)
                ++i
                if i==200:
                        break
        if "CCLK:" in response:
                time_found=True
                print(response)
        else:
                time.sleep(20)

split_date=response.split("\"")
date_time_str = split_date[1]
date_time_str = date_time_str.replace(",", " ")
date_time_str = date_time_str.replace("-", ".")
date_time_obj = datetime.datetime.strptime(date_time_str, '%y/%m/%d %H:%M:%S.%f')
subprocess.call(['sudo', 'date', '-s', date_time_obj.isoformat(" ")])
