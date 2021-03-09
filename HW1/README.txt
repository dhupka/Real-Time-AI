Problem 1 :
Solution is in hw1p1.py within the HW1 directory. Run by modifying line 11 for different model sizes. Similarly, modify line 25 to be "intersection" and modify the model sizes again to run inference on the intersection image over varying model sizes. Analysis upon data is within "Hupka_HW1_Report.pdf"

Problem 2: 
Solution is in hw1p2y. 2.1 ran the YOLO.v5s model on a batch size of 1,8,16 this can be run by changing line 25 to be "imgs1, imgs8, or imgs16" respectively. Ensure that file structure is spared as HW1/input/"XXXX" where "XXXX" are the input images, and that "hw1p2.py" is being run from the HW1 directory. Analysis upon data is within "Hupka_HW1_Report.pdf".

Problem 3:
Jetcam was not utilized in implementation due to difficulty in versioning and enabling jetcam to recognize USB camera. Attempts were made to modify jetcam source and rebuild setup.py with python3 to enable functionality, but this fix did not work. Thus, the camera was read directly using the "detect.py" framework provided by YOLO.v5 code was adapted from such https://github.com/ultralytics/yolov5. "detect.py" was modified to also measure the FPS of the inferrence by calculalting 1/inference time (t2-t1) and printing this value to the output terminal. "detect.py" can be run using a webcam by running "python3 detect.py --source 0" from the "/HW1/yolov5" directory. A link to a demonstration of this real-time inference is here: 
