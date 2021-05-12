# Real-Time-AI - Spring 2021 ECGR 4090  
## FINAL PROJECT
Any necessary code review is within the "detect.py" file in FinalProjectVehicle/Source
Run "./runMe.sh" with any necessary argument changes based on configuration
PLEASE DOWLOAD THE TRAINED MODEL BEFORE ATTEMPTING TO RUN AT: https://drive.google.com/file/d/1jL-aVRtBkP5Io93YWTxxlax0islc-sKG/view?usp=sharing
PLACE "best.pt" within the "FinalProjectVehicle/Source/weights" directory to ensure proper functionality.
When cloning into the repository ensure file directory structure is spared.
Similarly, modify the "./runMe.sh" --source argument to be "0" for a webcam or "XXXXXX.mp4" so long as "XXXXX.mp4" is within the "./runMe.sh" directory


## HW0
#### Problem 1 
Running of pretrained resnet101 model as in hw0p1.py. Five different images were passed into the model and the accuracies of which can be observed my running. In order to run clone repository and extract maintaining same file structure. Modify line 21 of hw0p1.py to be desired image i.e "randomResNetPics/fan.jpg". Large portions of code cloned from  https://github.com/deep-learning-with-pytorch/dlwpt-code/blob/master/p1ch2/2_pre_trained_networks.ipynb  

#### Problem 2
Running of pretrained resnetgen to input horse image, and ouput zebra image. Run similar to Problem #1, and modify line 93 to differnet horse input image options to observe output. Large portions of code cloned from https://github.com/deep-learning-with-pytorch/dlwpt-code/blob/master/p1ch2/3_cyclegan.ipynb

#### Problem 3
Observing the computational complexity and number of parameters of various models using ptflops. Included in hw0p1.py and hw0p2.py  
https://pypi.org/project/ptflops/

#### Problem 4
Recreating same image prediction as completed in Problem #1, but using the mobilenet model. Similarly, measuring the computational complexity and number of parameters as well. Included in hw0p4.py. https://pytorch.org/hub/pytorch_vision_mobilenet_v2/ used as reference. 

## HW1 - Further instructions are included within the HW1 directory README.txt
#### Problem 1 
YOLO.v5 was run across every model size to measure execution time, and using an image of an intersection across each model size to examine output. This solution is included in the "HW1/hw1p1.py" using a batch size of 1, and varying the model size, or using the provided intersection image, similarly modifying the model size. Code Adapted from https://pytorch.org/hub/ultralytics_yolov5/.  

#### Problem 2 
YOLO.v5 inference was run on the NVIDIA Jetson Nano with varying model sizes (s,m,l,x) and of varying batch sizes (1,8,16). The execution time was similarly observed is in Problem 1. his solution is included in the "HW1/hw1p2.py" using varying batch size, and varying the model size. Code Adapted from https://pytorch.org/hub/ultralytics_yolov5/  

### Problem 3  
Real-time Inference was ran on the NVIDIA Jetson Nano using the YOLO.v5s model. The camera was read from directly using the "detect.py" framwork provided by https://github.com/ultralytics/yolov5https://github.com/ultralytics/yolov5. The "detect.py" was slightly modified to also calculate the FPS and display to terminal output.

## HW2  Further instructions are included within the HW2 directory README.txt
#### Problem 1
Solution included in "hw2.py".  

#### Problem 2  
Solution included in "hw2.py."  

## HW3  
#### Problem 1  
Solution included in varying hw1p1.py or similar files.  

#### Problem 2
Solution included in "live_demo.py".  

#### Problem 3 
Solution included in "hw3p3.py".  

## HW4  
#### Problem 1  
Solution included in "hw4p1ab.ipynb".  

#### Problem 2
Solutions included in the "hw4p2a.ipynb" and "hw4p2bc.ipynb".

