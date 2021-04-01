#!/bin/bash
# file: run.sh

# This script 
# 1. Read the date-time-zonetime with cellular modem
# 2. Sycn RTC with RPi date
# 3. Hear any SMS pending for configuration
# 4. Takes pictures on the RPi and run ML for soybean or corn
# 5. Send a payload with camera performance and water stress level.

echo "Running run.sh "


sudo python /home/pi/updateDate.py
sudo /home/pi/System2RTC.sh
sudo python3 /home/pi/updateConfig.py
sudo python3 /home/pi/ML_Corn/takeImages.py
ps -aef | grep python
