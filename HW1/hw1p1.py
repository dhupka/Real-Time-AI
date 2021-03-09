#Damian Hupka
#ECGR 4090 - Real-Time AI Spring 2021
#03/09/2021
#Homework #1
#Code adapted from: https://pytorch.org/hub/ultralytics_yolov5/
import torch
import time

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5x', pretrained=True)

# Images
dir = 'input/'
imgs1 = [dir + ('firetruck.jpeg')]
imgs8 = [dir + f for f in ('ac.jpeg', 'bicycle.jpeg', 'bread.jpeg', 'bus.jpeg', 
 'candle.jpeg', 'computer.jpeg', 'dogs.jpeg', 'eggs.jpeg')]  
imgs16 = [dir + f for f in ('ac.jpeg', 'bicycle.jpeg', 'bread.jpeg', 'bus.jpeg', 
 'candle.jpeg', 'computer.jpeg', 'dogs.jpeg', 'eggs.jpeg', 'firetruck.jpeg', 'pencil.jpg',
 'salad.jpeg', 'soap.jpeg', 'toothbrush.jpeg', 'wagon.jpeg', 'table.jpeg')] 

# Inference
start = time.time()         #Beginning of measuring execution time
results = model(imgs16)
stop = time.time()          #Ending of measuring execution time
duration = stop - start

# Results
results.print()
results.show()  # or .show()
print(duration)

# Data
#print(results.xyxy[0])  # print img1 predictions (pixels)
#                   x1           y1           x2           y2   confidence        class
# tensor([[7.50637e+02, 4.37279e+01, 1.15887e+03, 7.08682e+02, 8.18137e-01, 0.00000e+00],
#         [9.33597e+01, 2.07387e+02, 1.04737e+03, 7.10224e+02, 5.78011e-01, 0.00000e+00],
#         [4.24503e+02, 4.29092e+02, 5.16300e+02, 7.16425e+02, 5.68713e-01, 2.70000e+01]])