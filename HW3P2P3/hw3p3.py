from trt_pose.parse_objects import ParseObjects
from trt_pose.draw_objects import DrawObjects
import PIL.Image
import torchvision.transforms as transforms
import cv2
import time
from torch2trt import TRTModule
import torch2trt
import torch
import trt_pose.models
import json
import trt_pose.coco
import os
import sys
import numpy as np

yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

with open('human_pose.json', 'r') as f:
    human_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(human_pose)


num_parts = len(human_pose['keypoints'])
num_links = len(human_pose['skeleton'])

# model = trt_pose.models.densenet121_baseline_att(
#     num_parts, 2 * num_links).cuda().eval()

model = trt_pose.models.resnet18_baseline_att(
    num_parts, 2 * num_links).cuda().eval()


MODEL_WEIGHTS = 'resnet18_baseline_att_224x224_A_epoch_249.pth'
# MODEL_WEIGHTS = 'densenet121_baseline_att_256x256_B_epoch_160.pth'

model.load_state_dict(torch.load(MODEL_WEIGHTS))


WIDTH = 224
HEIGHT = 224

data = torch.zeros((1, 3, HEIGHT, WIDTH)).cuda()

# with torch.cuda.device(0):
#     model_trt = torch2trt.torch2trt(model, [data], fp16_mode=True, max_workspace_size=1 << 25)


OPTIMIZED_MODEL = 'resnet18_baseline_att_224x224_A_epoch_249_trt.pth'
# OPTIMIZED_MODEL = 'densenet121_baseline_att_256x256_B_epoch_160_trt.pth'

#torch.save(model_trt.state_dict(), OPTIMIZED_MODEL)


pose_model_trt = TRTModule()
pose_model_trt.load_state_dict(torch.load(OPTIMIZED_MODEL))


# t0 = time.time()
# torch.cuda.current_stream().synchronize()
# for i in range(50):
#     y = model(data)
# torch.cuda.current_stream().synchronize()
# t1 = time.time()

# print(50.0 / (t1 - t0))


mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
device = torch.device('cuda')


def preprocess(image):
    global device
    device = torch.device('cuda')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]


parse_objects = ParseObjects(topology)
draw_objects = DrawObjects(topology)


if __name__ == '__main__':
    #v_f = sys.argv[1]
    cap = cv2.VideoCapture('test2.mp4')

    while(cap.isOpened()):
        start_time = time.time()
        ret, image = cap.read()
        if ret:
            image = cv2.resize(image,(224,224))
            # image = cv2.resize(image,(256,256))
            data = preprocess(image)
            cmap, paf = pose_model_trt(data)
            results = yolo_model(image, size=224)
            image_results = results.render()
            time_end = time.time()
            cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
            counts, objects, peaks = parse_objects(cmap, paf)
            # print(counts)
            # print(objects)
            draw_objects(image_results[0], counts, objects, peaks)
            cv2.putText(image_results[0], "FPS: " + str(float(1.0 / (time_end - start_time))), (0, 224), cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 0, 0,), 2, cv2.LINE_AA)
            cv2.imshow('res', image_results[0])
            cv2.waitKey(1)