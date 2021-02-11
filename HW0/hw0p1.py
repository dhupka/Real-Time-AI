#Damian Hupka
#ECGR 4090 Real-Time AI
#02/11/2021

from torchvision import models 
from torchvision import transforms
from PIL import Image
import torch
from ptflops import get_model_complexity_info

resnet = models.resnet101(pretrained=True)

preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])

img = Image.open("randomResNetPics/waterbottle.jpeg")
img_t = preprocess(img)
batch_t=torch.unsqueeze(img_t,0)

resnet.eval()
out = resnet(batch_t)

with open('imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]
_, index = torch.max(out, 1)
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
print(labels[index[0]],percentage[index[0]].item())
_, indices = torch.sort(out, descending=True)
print([(labels[idx],percentage[idx].item()) for idx in indices[0][:5]])


#P3 for problem 2
# with torch.cuda.device(0):
#   macs, params = get_model_complexity_info(resnet, (3, 224, 224), as_strings=True,
#                                            print_per_layer_stat=True, verbose=True)
#   print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
#   print('{:<30}  {:<8}'.format('Number of parameters: ', params))