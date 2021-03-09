#Damian Hupka
#ECGR 4090 - Real-Time AI Spring 2021
#03/09/2021
#Homework #2

import torch
import imageio
from PIL import Image
from torchvision.transforms import transforms
#Problem 1
#1.a Tensor from list (range(9)): size, offset, stride?
temp = torch.tensor(list(range(9)))
print(temp.storage())
temp = temp.type(torch.FloatTensor)
print(temp.storage())

#Size
print("Size of tensor is: " + str(len(temp)))
#Offset
print("Offset of tensor is: " + str(temp.storage_offset()))
#Stride
print("Stride of tensor is:" + str(temp.stride()))

#1.b cosine & square root, are these functions in torch
#Yes, both of these functions exist as "torch.cos()" and 
#"torch.sqrt()" respectively.

#1.c Apply element-wise cosine & square to temp, why error?
#Will error if tensor type is not float on PyTorch version 1.7.0
print(torch.cos(temp))
print(torch.sqrt(temp))

#1.d How to make 1.c work?
#Cast to type Float

#1.e Is there a version of the function that operates in place?
print(torch.cos_(temp))
print(torch.sqrt_(temp))

#Problem 2
#2.a take RGB pics

#2.b Load images, convert to tensors
r1 = Image.open('rgbPics/r1.jpg').convert('RGB')
r2 = Image.open('rgbPics/r2.jpg').convert('RGB')
r3 = Image.open('rgbPics/r3.jpg').convert('RGB')
g1 = Image.open('rgbPics/g1.jpg').convert('RGB')
g2 = Image.open('rgbPics/g2.jpg').convert('RGB')
b1 = Image.open('rgbPics/b1.jpg').convert('RGB')
b2 = Image.open('rgbPics/b2.jpg').convert('RGB')
b3 = Image.open('rgbPics/b3.jpg').convert('RGB')
ten = transforms.ToTensor()
r1t = ten(r1)
r2t = ten(r2)
r3t = ten(r3)
g1t = ten(g1)
g2t = ten(g2)
b1t = ten(b1)
b2t = ten(b2)
b3t = ten(b3)

#2.c Use mean() method to get brightness of each image on tensor
m_r1 = r1t.mean()
m_r2 = r2t.mean()
m_r3 = r3t.mean()
m_g1 = g1t.mean()
m_g2 = g2t.mean()
m_b1 = b1t.mean()
m_b2 = b2t.mean()
m_b3 = b3t.mean()
print("Brightness Red 1: " + str (m_r1))
print("Brightness Red 2: " + str (m_r2))
print("Brightness Red 3: " + str (m_r3))
print("Brightness Green 1: " + str (m_g1))
print("Brightness Green 2: " + str (m_g2))
print("Brightness Blue 1: " + str (m_b1))
print("Brightness Blue 2: " + str (m_b2))
print("Brightness Blue 3: " + str (m_b3))
print('\n')

#2.d Take mean of each channel of images, can I identify RGB only from avgs?
m_red_r1 = r1t[0, :, :].mean()
m_green_r1 = r1t[1, :, :].mean()
m_blue_r1 = r1t[2, :, :].mean()
print("Red Mean (red 1):" + str(m_red_r1))
print("Green Mean (red 1):" + str(m_green_r1))
print("Blue Mean (red 1):" + str(m_blue_r1))
print('\n')

m_red_r2 = r2t[0, :, :].mean()
m_green_r2 = r2t[1, :, :].mean()
m_blue_r2 = r2t[2, :, :].mean()
print("Red Mean (red 2):" + str(m_red_r2 ))
print("Green Mean (red 2):" + str(m_green_r2))
print("Blue Mean (red 2):" + str(m_blue_r2 ))
print('\n')

m_red_r3 = r3t[0, :, :].mean()
m_green_r3 = r3t[1, :, :].mean()
m_blue_r3 = r3t[2, :, :].mean()
print("Red Mean (red 3):" + str(m_red_r3))
print("Green Mean (red 3):" + str(m_green_r3))
print("Blue Mean (red 3):" + str(m_blue_r3))
print('\n')

m_red_g1 = g1t[0, :, :].mean()
m_green_g1 = g1t[1, :, :].mean()
m_blue_g1 = g1t[2, :, :].mean()
print("Red Mean (green 1):" + str(m_red_g1))
print("Green Mean (green 1):" + str(m_green_g1))
print("Blue Mean (green 1):" + str(m_blue_g1))
print('\n')

m_red_g2 = g2t[0, :, :].mean()
m_green_g2 = g2t[1, :, :].mean()
m_blue_g2 = g2t[2, :, :].mean()
print("Red Mean (green 2):" + str(m_red_g2))
print("Green Mean (green 2):" + str(m_green_g2))
print("Blue Mean (green 2):" + str(m_blue_g2))
print('\n')

m_red_b1 = b1t[0, :, :].mean()
m_green_b1 = b1t[1, :, :].mean()
m_blue_b1 = b1t[2, :, :].mean()
print("Red Mean (blue 1):" + str(m_red_b1))
print("Green Mean (blue 1):" + str(m_green_b1))
print("Blue Mean (blue 1):" + str(m_blue_b1))
print('\n')

m_red_b2 = b2t[0, :, :].mean()
m_green_b2 = b2t[1, :, :].mean()
m_blue_b2 = b2t[2, :, :].mean()
print("Red Mean (blue 2):" + str(m_red_b2))
print("Green Mean (blue 2):" + str(m_green_b2))
print("Blue Mean (blue 2):" + str(m_blue_b2))
print('\n')

m_red_b3 = b3t[0, :, :].mean()
m_green_b3 = b3t[1, :, :].mean()
m_blue_b3 = b3t[2, :, :].mean()
print("Red Mean (blue 3):" + str(m_red_b3))
print("Green Mean (blue 3):" + str(m_green_b3))
print("Blue Mean (blue 3):" + str(m_blue_b3))
print('\n')