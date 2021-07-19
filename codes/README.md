# StressCam

The StressCam is a project with this specific goal: To find a On-Farm solution to detect Water stress in real-time and relate to other measurements on the field. Keywords: open source, computer vision, machine learning, edge computing.

![](https://lh6.googleusercontent.com/7CYahv4pL-UyMMPPmljDtzjVqGhKxL244pjyrUobLfCUVoLKhSitLrKBxk6yv_Pf3WdrxyGQ9OclLobWjuPSFBjaQHQDmfh0GnVLu4YFEjaFftxPOg1SHJ6OlW8DR2ySmQujDIdS)
Figure 1. StressCam Gen3

The StressCam is a low cost camera that can be installed in the field in a several ways. The camera consists of a raspberry pi, a PiCamera, a power management system and real time clock, and a solar power system. The idea of this project is to obtain a low cost camera (~U$200) that can be installed in the field and with which we can manage the crop, specifically in the management of water stress.

## Hardware

XXXXXXXXXXXXX AKHIL XXXXXX Hardware description

1) WITTY PI:  Witty Pi is small extension board that can adds realtime clock and power management to the Raspberry Pi. It is the component which is responsible for time syncronization and scheduling.

2) RASPBERRY PI: Used to store all the codes(Memory card connected to it) and connect all the other components. The memory card is 64 gb in size and stores the images taken by   the camera.

3) HOLOGRAM DEVICE: The component is uded to connect to internet. It has a sim card which scans and connects to the desired network. This device is used to send and receive messages.

4) POWER BANK: Used to provide power to the raspberry pi board and witty pi board.

5) SOLAR PANEL: Used to charge the power bank.It is oriented exactly in the direction of sun to receive maximum solar input. 

![](https://lh5.googleusercontent.com/N2H0_MO4PmXhJ8P6JGCTe2nUMajob00ZQ6D4R9IX0nlp051khI73B4iDbEphEmGPOLRfs5xxlFEaVW1YWIUXHLcEAsT46O_HFPPFVrF1zo8oNumjRA4aHHfqpbMwh6JQS9AN2tiv)

## Operations Modes

XXXXXXXXXXXXX AKHIL XXXXXX General description of operation modes

The camera has 4 operation modes start,setup,stop and shutdown. Each and every mode has its on schedule and used for different purposes. 

![](https://lh3.googleusercontent.com/c-DIDslRGLf1w9noavTf1WWEy3sW4QI2ZKYUXA04UJIrBAuuwiwpIS7r3sFw22wi47Kri7KsWrWSWBxAARF2zQd0NYrtwhlgdgsMgUVVC-TmhLfmja1WC1ny3HAPeFdmp2Qozy37)

## Pipeline

XXXXXXXXXXXXX AKHIL XXXXXX General description of the pipeline


System2RTC
===============================================================================
This code is used to syncronize the time for all the components used. The time is taken from the hologram device and then this time stamp is syncronized 
by the witty pi device and eventually runs on the scheduled syncronized time scale.
Also  this code provides useful utility functions such as
•	Setting start up ,shutdown times.
•	Clearing start up, shutdown times.
•	I2c read
•	I2c write
•	Getting temperature data. etc


Run.sh
===============================================================================
This program runs the complete set up and one of the most important command of the stress cam. It does various functions like:
Reading the date-time-zonetime with cellular modem.

Sycn RTC with RPi date.

Hear any SMS pending for configuration.

Takes pictures on the RPi and run ML for soybean or corn.

Send a payload with camera performance and water stress level.


Updateconfig
===============================================================================
This programs function is to convert the defualt time schedule which we install in the cameras to the required start time schedule when we run the start message using the 
config.json file which contains many parameters like device id,hologram id etc. It updates the RTC and witty pi log and syncronizes the camera to function with the 
given start schedule. Whenever a start or default message is sent the following parameters get updated:
```         device_params['Hologram ID']  =  cmd_splitted[1]
            device_params['Camera ID'] = cmd_splitted[2]
            device_params['Crop'] = cmd_splitted[4]
            device_params['Treatment'] = cmd_splitted[3]
            device_params['test ML'] = cmd_splitted[5]
            device_params['mode'] = cmd_splitted[0]
            device_params['timezone'] = cmd_splitted[6]
            device_params['farm'] = cmd_splitted[7]
            device_params['rep'] = cmd_splitted[8]
```

updatedate
===============================================================================
This command updates the date in the Hologram as well as the RTC ,witty pi log. Sometimes when we unplug the hologram before it recives a signal connectivity the hologram sets its default date eventually syncronizing with the witty pi and we end up loosing the time stamp. But this program makes sure that the date is set up back to tthe correct stamp,
the if statement makes sure that we are  in the correct year.
