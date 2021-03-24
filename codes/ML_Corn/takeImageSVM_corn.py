#!/usr/bin/python3
import sys
sys.path.append("/home/pi/.local/lib/python3.7/site-packages")
sys.path.append("/home/pi")
from picamera import PiCamera
from time import sleep
import datetime
import gpiozero as gpz
import json
from skimage.io import imread #read images
from skimage.transform import resize
from PIL import Image
import cv2

import psutil
#SVM dependancies:
import numpy as np
from sklearn import svm
from skimage.feature import hog
import numpy as np
import pickle
from PIL import Image
import cv2
import random
import os
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
###################################################



disk = psutil.disk_usage('/')
disk_percent_used = disk.percent
disk_free = disk.free / 2**30


from Hologram.HologramCloud import HologramCloud
credentials = {'devicekey':devicekey} #'6r)^]p]Q'} #Hologram device key from hologram.io
hologram = HologramCloud(credentials, network='cellular',authentication_type='csrpsk') #Connect to Hologram CLoud, change network to cellular to connect to LTE

sum_RSSI = 0.0
sum_quality = 0.0
num_samples = 5
camera = PiCamera()
curr = datetime.datetime.now()
currDate = curr.strftime("%d_%m_%Y")
currTime = curr.strftime("%H_%M_%S")
timeStamp = curr.strftime("%d_%m_%Y_%H_%M_%S")
file = '/home/pi/images/' + curr.strftime("%d_%m_%Y_%H_%M_%S") + '.jpg'

camera.resolution=(2592,1944)
#camera.rotation = 180
camera.capture(file)
#for i in range(num_samples):
#    signal_strength = hologram.network.signal_strength
#    print('Signal strength: ' + signal_strength)
#    rssi, qual = signal_strength.split(',')
#    sum_RSSI = sum_RSSI +int(rssi)
#    sum_quality = sum_quality +int(qual)
#    time.sleep(2)
#print('Average RSSI' + str(sum_RSSI/num_samples))
#print('Average quality' + str(sum_quality/num_samples))
cpu = gpz.CPUTemperature()
cpu_temp = cpu.temperature
print("CPU Temperature:", cpu_temp)
################################################
#IMPORTANT: If you want to utilize the SVM ML model on images taken in the field
#           change the "Make pred on testset" value to: false  instead of true
if testML:
    #Test Corn Images:
    street = os.listdir('/home/pi/ML_Corn/CornTestImages')
    randomImage = random.randrange(len(street)) #pick a random test image
    file = '/home/pi/ML_Corn/CornTestImages/'+street[randomImage]
    with open('/home/pi/ML_Corn/testImageLabels.json') as f:
        correct_labels = json.load(f)
    actual_label = correct_labels[street[randomImage]]
    im = imread(file)
else:
    im = cv2.imread(file)
im = Image.fromarray(im,"RGB")
print("Resizing image...")
im_final = im.resize((324,216))#Model was trained on 324(w)x216(h) images
print("Captured and Resized Image!")
########Convert RGB Image to LUX representation #############
print("Converting Image to LUX format...")
img0 = np.asarray(im_final)
LUX = np.zeros((img0.shape[0],img0.shape[1], 3),dtype=np.float32 )
row = img0.shape[0]-1 #img0 = image we are manipulating. row = height of image. columns = width of image
columns = img0.shape[1]-1
for i in range(row):
    for j in range(columns):

      r = img0[i][j][2]#cvGet2D(r_plane,i,j)
      g = img0[i][j][1]
      b = img0[i][j][0]

      l = ((((r+1)**0.3)) * (((g+1)**0.6)) * (((b+1)**0.1)) - 1).astype(np.float32) #r.val[0] = r,need to replace power function
      M = 256
      if r>l:
        u = ((M/2)*((r+1)/(l+1))).astype(np.float32)
        x = ((M/2)*((b+1)/(l+1))).astype(np.float32)
      else:
        u = (M-((M/2)*((l+1)/(r+1)))).astype(np.float32)
        x = (M-((M/2)*((l+1)/(b+1)))).astype(np.float32)

      LUX[i][j][0] = int(l)
      LUX[i][j][1] = int(u)
      LUX[i][j][2] = int(x)
print("Finished converting image to LUX!")
######Extract Histogram of Gradients from LUX Image ##########
print("Extracting HOG Features...")
pixels_per_cell = (9,9)
cells_per_block = (24,36)
fd = hog(LUX, orientations=9, pixels_per_cell=pixels_per_cell,cells_per_block=cells_per_block,  block_norm='L2', transform_sqrt=False, visualize= False, multichannel=True) #16,16
######Load the Model and make a prediction#############
print("Loading SVM model and making prediction...")
svm_file = '/home/pi/ML_Corn/svm_tuned_modelFINAL.sav'
loaded_model = pickle.load(open(svm_file, 'rb'))
fd = fd.reshape(1,-1) #required if only one sample
result = loaded_model.predict_proba(fd)
print(result.shape)
print("Evaulating Image: ", street[randomImage])
print("Predicted Water Stress Probabilites:",result)
print("Actual Label", actual_label)

waterStressLevel = int(np.argmax(result))+1

print("Predicted WS:", waterStressLevel)

if testML:
    data = {
                "DEV_ID": devicekey,
                "CAM_ID": cameraID,
                "CPU_TEMP": cpu_temp,
                "DATE":currDate,
                "TIME":currTime,
                #"SD":disk_percent_used,
                "SD_free":disk_free,
                "CROP": crop,
                "TREATMENT": treatment,
                "FILE":street[randomImage],
                "P_WS_1": result[0][0],
                "P_WS_2": result[0][1],
                "P_WS_3": result[0][2],
                "Predicted_WS":waterStressLevel,
                "Actual_WS": actual_label,
                #Need to add Actual WS level if testing
    #            "Av_RSSI":sum_RSSI/num_samples,
    #            "Av_qual":sum_quality/num_samples,
            }
else:
    data = {
                "DEV_ID": devicekey,
                "CAM_ID": cameraID,
                "CPU_TEMP": cpu_temp,
                "DATE":currDate,
                "TIME":currTime,
                #"SD":disk_percent_used,
                "SD_free":disk_free,
                "CROP": crop,
                "TREATMENT": treatment,
                "FILE":timeStamp,
                "P_WS_1": result[0][0],
                "P_WS_2": result[0][1],
                "P_WS_3": result[0][2],
                "Predicted_WS":waterStressLevel,
                #Need to add Actual WS level if testing
    #            "Av_RSSI":sum_RSSI/num_samples,
    #            "Av_qual":sum_quality/num_samples,
            }
recv = hologram.sendMessage(data) # Send message to hologram cloud
print("Recieved Code:",recv)
print("0 Means Succesful Transmission")
sleep(20)