># BADIC SCRIPTS TO IMPLEMENT TRAINING USING OBJECT DETECTION MODELS

**Author: Adedamola Ayodeji Sode**   
**Email: adedamola.sode@gmail.com**

# BRIEF DESCRIPTION 
This project follows the tutorial of <a href = "https://www.youtube.com/channel/UCHXa4OpASJEwrHrLeIzw7Yg">Nicholas Renotte</a> on youtube, specifically using this video: <a href ="https://www.youtube.com/watch?v=IOI0o3Cxv9Q&ab_channel=NicholasRenotte">Real Time Face Mask Detection with Tensorflow and Python | Custom Object Detection w/ MobileNet SSD</a>.

# DEPENDENCIES REQUIRED TO BUILD AND RUN
All dependencies used are in their latest versions as of 20-04-22, and their corrsponding libraries and dependencies which follow suit:
1. Tensorflow, with CUDA and CuDNN for GPU acceleration
2. Object-detection-api
3. OpenCV
4. Numpy

# CONFIGURATION
1. Create a Project folder with this given directory tree:

<img src="/directory.png" alt="directory tree" width="" height="">

2. Insert your train and test images and corresponsing .xml labels in the respective train and test folders in the /workspace/images directory

3. Run the run_me file and make adjustments where needed in the pipeline configuration file or set.py module in scripts as needed (Best done with your preferred IDE)

# EXECUTING THE PROGRAM

1. Clone the git 
```bash
git clone  https://github.com/AsimovNo9/ScrewDetection.git
```
2. Run the run_me.py file using your choice of IDE or your commandline 
```bash
python ./run_me.py
```




