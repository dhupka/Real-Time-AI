import torch
import time

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Images
dir = '/HW1/'
imgs = [dir + ('intersection(nyc_street)'))]  # batched list of images

# Inference
#start = time.time()         #Beginning of measuring execution time
results = model(imgs)
#stop = time.time()          #Ending of measuring execution time
#duration = stop - start

# Results
results.print()  
results.show()  # or .show()
#print(duration)

# Data
print(results.xyxy[0])  # print img1 predictions (pixels)
#                   x1           y1           x2           y2   confidence        class
# tensor([[7.50637e+02, 4.37279e+01, 1.15887e+03, 7.08682e+02, 8.18137e-01, 0.00000e+00],
#         [9.33597e+01, 2.07387e+02, 1.04737e+03, 7.10224e+02, 5.78011e-01, 0.00000e+00],
#         [4.24503e+02, 4.29092e+02, 5.16300e+02, 7.16425e+02, 5.68713e-01, 2.70000e+01]])