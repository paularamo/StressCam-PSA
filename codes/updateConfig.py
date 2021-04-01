#!/usr/bin/python3
import os
import sys
sys.path.append("/home/pi/.local/lib/python3.7/site-packages")
sys.path.append("/home/pi")

import json
import time
import datetime
import subprocess
#from Hologram.HologramCloud import HologramCloud

####Device Parameters##############################
#Modify the config.json file w/ correct parameters#
###################################################
with open('/home/pi/ML_Corn/config.json') as f:           #
    device_params = json.load(f)                  #
                                                  #
devicekey = device_params['Hologram ID']          #
cameraID = device_params['Camera ID']             #
crop = device_params['Crop']                 #
treatment = device_params['Treatment']            #
testML = device_params['test ML']    #
mode = device_params['mode']
time_zone = device_params['timezone']

###close????
###################################################

print('Update config.json - SMS') #welcome message for console

from Hologram.HologramCloud import HologramCloud
credentials = {'devicekey':devicekey} #'6r)^]p]Q'} #Hologram device key from hologram.io
hologram = HologramCloud(credentials, network='cellular',authentication_type='csrpsk') #Connect to Hologram CLoud, change network to cellular to connect to LTE


#hologram=HologramCloud(dict(), network='cellular') #setup hologram cloud
hologram.enableSMS #tell hologram to listen for sms

recv=None #if we dont set initial variables the loop breaks
cmd=None

while True: #start the loop
    recv = hologram.popReceivedSMS() #this works, makes the recv variable equal an incoming message
    if recv is not None: #works, run the following code when a message comes in
        print( 'SMS From: ', 0)#recv.sender) #print the sender of the sms
        cmd = recv.message #'start;MiQ==Idm;Rasp-ID000;bare;corn;true'#recv.message #works, makes the cmd variable the payload of the message.
        cmd_splitted = cmd.split(";")
        print (cmd_splitted) #works, prints only what the sms message is.
        if cmd_splitted[0] == 'Start' or cmd_splitted[0] == 'start': #Checks for the first word
            print ('Start command received') #let the console show we know the SMS had the word start and we will process it
            os.system('/home/pi/wittypi/start_schedule.sh')
            time.sleep(2)

            f = open('/home/pi/ML_Corn/config.json', 'w+')           #
            device_params['Hologram ID']  =  cmd_splitted[1]
            device_params['Camera ID'] = cmd_splitted[2]
            device_params['Crop'] = cmd_splitted[4]
            device_params['Treatment'] = cmd_splitted[3]
            device_params['test ML'] = cmd_splitted[5]
            device_params['mode'] = cmd_splitted[0]

            f.write(json.dumps(device_params))#, f)
            f.close()

        elif cmd_splitted[0] == 'Stop' or cmd_splitted[0] == 'stop':
            print ('Stop command received') #let the console show we know the SMS had the word start and we will process it
            os.system('/home/pi/wittypi/stop_schedule.sh')
            subprocess.call (["sudo","shutdown","+15"])#Shutdown in 15 minutes
            time.sleep(2)
            f = open('/home/pi/ML_Corn/config.json', 'w+')           #
            device_params['mode'] = cmd_splitted[0]
            #jsonfile = open('/home/pi/ML_Corn/config.json', 'w')
            f.write(json.dumps(device_params))#, f)
            f.close()
        elif cmd_splitted[0] == 'Shutdown' or cmd_splitted[0] == 'shutdown':
            print ('Shutdown command received') #let the console show we know the SMS had the word start and we will process it
            os.system('/home/pi/wittypi/shutdown_schedule.sh')
            time.sleep(2)
            f = open('/home/pi/ML_Corn/config.json', 'w+')           #
            device_params['mode'] = cmd_splitted[0]
            #jsonfile = open('/home/pi/ML_Corn/config.json', 'w')
            f.write(json.dumps(device_params))#, f)
            f.close()
            subprocess.call (["sudo","shutdown","+15"])#Shutdown in 15 minutes
            #time.sleep(2)
        elif cmd_splitted[0] == 'Default' or cmd_splitted[0] == 'default':
            print ('Default command received') #let the console show we know the SMS had the word start and we will process it
            os.system('/home/pi/wittypi/default_schedule.sh')
            #subprocess.call (["sudo","shutdown","+15"])#Shutdown in 10 minutes
            time.sleep(2)
            f = open('/home/pi/ML_Corn/config.json', 'w+')           #
            device_params['mode'] = cmd_splitted[0]
            #jsonfile = open('/home/pi/ML_Corn/config.json', 'w')
            f.write(json.dumps(device_params))#, f)
            f.close()
        elif cmd_splitted[0] == 'setup' or cmd_splitted[0] == 'Setup':
            print ('Default command received') #let the console show we know the SMS had the word start and we will process it
            os.system('/home/pi/wittypi/default_schedule.sh')
            #subprocess.call (["sudo","shutdown","+15"])#Shutdown in 10 minutes
            time.sleep(2)
            cli_setup = "sudo" + " " + "timedatectl" + " " + "set-timezone" + " " + cmd_splitted[6]
            print(cli_setup)
            #subprocess.call("sudo", "timedatectl", "set-timezone", str(cmd_splitted[6]))
            os.system(cli_setup)
            f = open('/home/pi/ML_Corn/config.json', 'w+')           #
            device_params['Hologram ID']  =  cmd_splitted[1]
            device_params['Camera ID'] = cmd_splitted[2]
            device_params['Crop'] = cmd_splitted[4]
            device_params['Treatment'] = cmd_splitted[3]
            device_params['test ML'] = cmd_splitted[5]
            device_params['mode'] = cmd_splitted[0]
            device_params['timezone'] = cmd_splitted[6]
            #jsonfile = open('/home/pi/ML_Corn/config.json', 'w')
            f.write(json.dumps(device_params))#, f)
            f.close()
    if recv is None: #Runs the following code when there is no message
        print ('No Command received, keep the previous configuration')#Prints timestamp, handy for troubleshooting
        break