#!/usr/bin/python3
#!/home/pi/.local/lib/python3.7/site-packages

import sys
sys.path.append("/home/pi/.local/lib/python3.7/site-packages")
sys.path.append("/home/pi")
from picamera import PiCamera
from time import sleep
import datetime
import gpiozero as gpz
import json
#from skimage.io import imread #read images
#from skimage.transform import resize
from PIL import Image

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
with open('/home/pi/ML_Corn/config.json') as f:   #
    device_params = json.load(f)                  #
                                                  #
devicekey = device_params['Hologram ID']          #
cameraID = device_params['Camera ID']             #
crop = device_params['Crop']                      #
treatment = device_params['Treatment']            #
testML = device_params['test ML']
mode = device_params['mode']
time_zone = device_params['timezone']
farm = device_params['farm']
rep = device_params['rep']
version = device_params['ver']
###################################################

tz_splitted = time_zone.split("/")

from Hologram.HologramCloud import HologramCloud
credentials = {'devicekey':devicekey} #'6r)^]p]Q'} #Hologram device key from hologram.io
try:
    hologram = HologramCloud(credentials, network='cellular',authentication_type='csrpsk') #Connect to Hologram CLoud, change network to cellular to connect to LTE
except:
    print("Could not connect or open socket to Hologram Cloud");
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
camera.rotation = 180
camera.capture(file)
camera.close()

cpu = gpz.CPUTemperature()
cpu_temp = cpu.temperature
print("CPU Temperature:", cpu_temp)


disk = psutil.disk_usage('/')
disk_percent_used = disk.percent
disk_free = disk.free / 2**30
################################################
#IMPORTANT: If you want to utilize the SVM ML model on images taken in the field
#            change the "test ML" value to: false  instead of true
if testML=="true":
    #Test Corn Images:
    if (crop == "corn"):
        street = os.listdir('/home/pi/ML_Corn/CornTestImages')
        randomImage = random.randrange(len(street)) #pick a random test image
        file = '/home/pi/ML_Corn/CornTestImages/'+street[randomImage]
        with open('/home/pi/ML_Corn/testImageLabels.json') as f:
            correct_labels = json.load(f)
        actual_label = correct_labels[street[randomImage]]
        im = cv2.imread(file)
    else: #test soybean images
        testImages = np.load("/home/pi/ML_Corn/soybeanTestImages.npy")
        testLabels = np.load("/home/pi/ML_Corn/soybeanTestLabels.npy")
        randomImage = random.randrange(len(testLabels)) #pick a random test image
        actual_label = np.argmax(testLabels[randomImage])#one-hot encoded
        im = testImages[randomImage]
else:#read image just captured by camera
    im = cv2.imread(file)
###SVM model for Corn, TFLite model for soybeans
if (crop == "corn"):
    im = Image.fromarray(im,"RGB")
    #print("Resizing image...")
    im_final = im.resize((324,216))#Model was trained on 324(w)x216(h) images
    #print("Captured and Resized Image!")
    ########Convert RGB Image to LUX representation #############
    #print("Converting Image to LUX format...")
    img0 = np.asarray(im_final)
    LUX = np.zeros((img0.shape[0],img0.shape[1], 3),dtype=np.float32 )
    row = img0.shape[0]-1 #img0 = image we are manipulating. row = height of image. columns = width of image
    columns = img0.shape[1]-1
    for i in range(row):
        for j in range(columns):

          r = img0[i][j][2]
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
    #print("Finished converting image to LUX!")
    ######Extract Histogram of Gradients from LUX Image ##########
    #print("Extracting HOG Features...")
    pixels_per_cell = (9,9)
    cells_per_block = (24,36)
    fd = hog(LUX, orientations=9, pixels_per_cell=pixels_per_cell,cells_per_block=cells_per_block,  block_norm='L2', transform_sqrt=False, visualize= False, multichannel=True) #16,16
    ######Load the Model and make a prediction#############
    #print("Loading SVM model and making prediction...")
    svm_file = '/home/pi/ML_Corn/svm_tuned_modelFINAL.sav'
    loaded_model = pickle.load(open(svm_file, 'rb'))
    fd = fd.reshape(1,-1) #required if only one sample
    result = loaded_model.predict_proba(fd)
    #print("Evaluating Image: ", file)
    #print("Predicted Water Stress Probabilites:",result)
    waterStressLevel = int(np.argmax(result))+1
    #print("Predicted WS:", waterStressLevel)
    try:
        print("Actual WS: ", actual_label)
    except NameError:
        pass
else: #DL model for soybeans
    #TFlite Dependancies: don't import unless in Soybeans
    from tensorflow import lite as tflite
    import tensorflow as tf
    #im = Image.fromarray(im,"RGB")
    im_final = cv2.resize(im,(224,224))#Model was trained on 224x224 images
    #Preprocess Image for Mobilenet_V2
    #print(im_final)
    im_final=tf.keras.applications.mobilenet_v2.preprocess_input(im_final)
    # Load TFLite model and allocate tensors.
    im_final= np.array(im_final,dtype=np.float32)
    #print("allocating tensors")
    interpreter = tflite.Interpreter(model_path="/home/pi/ML_Corn/soybeanTFLiteModel.tflite")
    interpreter.allocate_tensors()
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    Xtest = np.array(im_final, dtype = np.float32)

    #test model
    input_data = np.expand_dims(Xtest,axis=0)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    #print("invoking interpreter")
    interpreter.invoke()
    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    results = np.squeeze(output_data)
    #print(results)
    waterStressLevel = int(np.argmax(results))
    percentConfident = results[waterStressLevel]*100
    #print("Predicted WS Level: ", waterStressLevel)
    #print("Confidence: ",percentConfident)
    try:
        print("Actual WS: ", actual_label)
    except NameError:
        pass
if (crop =="corn"):
    if testML=="true":
        data = {
                    "DEV_ID": devicekey,
                    #"CAM_ID": cameraID,
                    "CPU_TEMP": cpu_temp,
                    "DATE":currDate,
                    "TIME":currTime,
                    "SD_free":round(disk_free,2),
                    "FARM":farm,
                    "REP":rep,
                    "CROP": crop,
                    "TRT": treatment,
                    "MODE": mode,
                    "ZONE": tz_splitted[1],
                    "FILE":street[randomImage],
                    "P_WS_1": round(result[0][0],4),
                    "P_WS_2": round(result[0][1],4),
                    "P_WS_3": round(result[0][2],4),
                    "P_WS":waterStressLevel,
                    "A_WS": actual_label,
                    "VER": version,
                }
    else:
        data = {
                    "DEV_ID": devicekey,
                    #"CAM_ID": cameraID,
                    "CPU_TEMP": cpu_temp,
                    "DATE":currDate,
                    "TIME":currTime,
                    "SD_free":round(disk_free,2),
                    "FARM":farm,
                    "REP":rep,
                    "CROP": crop,
                    "TRT": treatment,
                    "MODE": mode,
                    "ZONE": tz_splitted[1],
                    "FILE":timeStamp,
                    "P_WS_1": round(result[0][0],4),
                    "P_WS_2": round(result[0][1],4),
                    "P_WS_3": round(result[0][2],4),
                    "P_WS":waterStressLevel,
                    "VER": version,
                }
else:#different payloads for Soybean cams
    if testML=="true":
        data = {
                    "DEV_ID": devicekey,
                    #"CAM_ID": cameraID,
                    "CPU_TEMP": cpu_temp,
                    "DATE":currDate,
                    "TIME":currTime,
                    "SD_free":round(disk_free,2),
                    "FARM":farm,
                    "REP":rep,
                    "CROP": crop,
                    "TRT": treatment,
                    "MODE": mode,
                    "ZONE": tz_splitted[1],
                    "FILE":"Test Image",
                    "P_WS_0": round(float(results[0]),4),
                    "P_WS_1": round(float(results[1]),4),
                    "P_WS_2": round(float(results[2]),4),
                    "P_WS_3": round(float(results[3]),4),
                    "P_WS_4": round(float(results[4]),4),
                    "P_WS_5": round(float(results[5]),4),
                    "P_WS":waterStressLevel,
                    "A_WS": int(actual_label),
                    "VER": version,
                }
    else:
        data = {
                    "DEV_ID": devicekey,
                    #"CAM_ID": cameraID,
                    "CPU_TEMP": cpu_temp,
                    "DATE":currDate,
                    "TIME":currTime,
                    "SD_free":round(disk_free,2),
                    "FARM":farm,
                    "REP":rep,
                    "CROP": crop,
                    "TRT": treatment,
                    "MODE": mode,
                    "ZONE": tz_splitted[1],
                    "FILE":timeStamp,
                    "P_WS_0": round(float(results[0]),4),
                    "P_WS_1": round(float(results[1]),4),
                    "P_WS_2": round(float(results[2]),4),
                    "P_WS_3": round(float(results[3]),4),
                    "P_WS_4": round(float(results[4]),4),
                    "P_WS_5": round(float(results[5]),4),
                    "P_WS":waterStressLevel,
                    "VER": version,
                }
#print(data)
with open('data.txt', 'a') as outfile:
            json.dump(data, outfile)
            outfile.write('\n')

try:
    recv = hologram.sendMessage(data) # Send message to hologram cloud
    print("Recieved Code:",recv)
    print("0 Means Succesful Transmission")
except:
    print("Not network")


sleep(20)