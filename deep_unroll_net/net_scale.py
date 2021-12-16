import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from correlation_package import Correlation

from package_core.net_basics import *
from forward_warp_package import *
from reblur_package import *
from net_pwc import *
from image_proc import *

def nearest_pow2(num):
    n = 0
    for i in range(10):
        if pow(2,i) > num:
            n = i-1
            break
    return n

def generate_2D_grid(H, W):
    x = torch.arange(0, W, 1).float().cuda() 
    y = torch.arange(0, H, 1).float().cuda()

    xx = x.repeat(H, 1)
    yy = y.view(H, 1).repeat(1, W)
    
    grid = torch.stack([xx, yy], dim=0) 

    return grid
    

class FlowWarper(nn.Module):
    def __init__(self, w, h):
        super(FlowWarper, self).__init__()
        x = np.arange(0,w)
        y = np.arange(0,h)
        gx, gy = np.meshgrid(x,y)
        self.w = w
        self.h = h
        self.grid_x = torch.autograd.Variable(torch.Tensor(gx), requires_grad=False).cuda()
        self.grid_y = torch.autograd.Variable(torch.Tensor(gy), requires_grad=False).cuda()

    def forward(self, img, uv):
        u = uv[:,0,:,:]
        v = uv[:,1,:,:]
        X = self.grid_x.unsqueeze(0).expand_as(u) + u
        Y = self.grid_y.unsqueeze(0).expand_as(v) + v
        X = 2*(X/self.w - 0.5)
        Y = 2*(Y/self.h - 0.5)
        grid_tf = torch.stack((X,Y), dim=3)
        img_tf = torch.nn.functional.grid_sample(img, grid_tf)
        return img_tf
    

class UNet_Esti_Scale(nn.Module):

    def __init__(self):
		    super(UNet_Esti_Scale, self).__init__()
      
		    ## Encoder 
		    self.conv1 = nn.Conv2d(10,32,7,1,3,bias=False)
		    self.conv2 = nn.Conv2d(32,32,7,1,3,bias=False)
		    self.relu = nn.LeakyReLU(0.1,True)
		    self.avgpool1 = nn.AvgPool2d(kernel_size=7,stride=2, padding=3)

		    self.conv3 = nn.Conv2d(32,64,5,1,2,bias=False)
		    self.conv4 = nn.Conv2d(64,64,5,1,2,bias=False)
		    self.avgpool2 = nn.AvgPool2d(kernel_size=5,stride=2, padding=2)

		    self.conv5 = nn.Conv2d(64,128,3,1,1,bias=False)
		    self.conv6 = nn.Conv2d(128,128,3,1,1,bias=False)
		    self.avgpool3 = nn.AvgPool2d(kernel_size=3,stride=2, padding=1)

		    self.conv7 = nn.Conv2d(128,256,3,1,1,bias=False)
		    self.conv8 = nn.Conv2d(256,256,3,1,1,bias=False)
		    self.avgpool4 = nn.AvgPool2d(kernel_size=3,stride=2, padding=1)

		    self.conv9 = nn.Conv2d(256,512,3,1,1,bias=False)
		    self.conv10 = nn.Conv2d(512,512,3,1,1,bias=False)
		    self.avgpool5 = nn.AvgPool2d(kernel_size=3,stride=2, padding=1)

		    self.conv11 = nn.Conv2d(512,512,3,1,1,bias=False)
		    self.conv12 = nn.Conv2d(512,512,3,1,1,bias=False)

		    ## Decoder
		    self.upsample2D = nn.Upsample(scale_factor=2, mode='bilinear')

		    self.conv13 = nn.Conv2d(512,512,3,1,1,bias=False)
		    self.conv14 = nn.Conv2d(512,512,3,1,1,bias=False)

		    self.conv15 = nn.Conv2d(512,256,3,1,1,bias=False)
		    self.conv16 = nn.Conv2d(256,256,3,1,1,bias=False)

		    self.conv17 = nn.Conv2d(256,128,3,1,1,bias=False)
		    self.conv18 = nn.Conv2d(128,128,3,1,1,bias=False)

		    self.conv19 = nn.Conv2d(128,64,3,1,1,bias=False)
		    self.conv20 = nn.Conv2d(64,64,3,1,1,bias=False)

		    self.conv21 = nn.Conv2d(64,32,3,1,1,bias=False)
		    self.conv22 = nn.Conv2d(32,32,3,1,1,bias=False)

		    self.conv23 = nn.Conv2d(32,6,3,1,1,bias=False)
		
		    #self.tanh = nn.Tanh()#
		    self.sigmoid = nn.Sigmoid()

    def forward(self, im_rs0, F_0_1, im_rs1, F_1_0):
		    sources = []

		    X = torch.cat([im_rs0, F_0_1, im_rs1, F_1_0], 1)

		    ## Encoder
		    X = self.conv1(X)
		    X = self.relu(X)
		    X = self.conv2(X)
		    X = self.relu(X)
		    #print(X.size())
		    sources.append(X)

		    X = self.avgpool1(X)
		    X = self.conv3(X)
		    X = self.relu(X)
		    X = self.conv4(X)
		    X = self.relu(X)
		    #print(X.size())
		    sources.append(X)

		    X = self.avgpool2(X)
		    X = self.conv5(X)
		    X = self.relu(X)
		    X = self.conv6(X)
		    X = self.relu(X)
		    #print(X.size())
		    sources.append(X)

		    X = self.avgpool3(X)
		    X = self.conv7(X)
		    X = self.relu(X)
		    X = self.conv8(X)
		    X = self.relu(X)
		    #print(X.size())
		    sources.append(X)

		    X = self.avgpool4(X)
		    X = self.conv9(X)
		    X = self.relu(X)
		    X = self.conv10(X)
		    X = self.relu(X)
		    #print(X.size())
		    sources.append(X)

		    X = self.avgpool5(X)
		    X = self.conv11(X)
		    X = self.relu(X)
		    X = self.conv12(X)
		    X = self.relu(X)
		    #print(X.size())

		    ## Decoder
		    X = self.upsample2D(X)
		    X = self.conv13(X)
		    X = self.relu(X)
		    #print(X.size())
		    X = X + sources[-1]
		    X = self.conv14(X)
		    X = self.relu(X)

		    X = self.upsample2D(X)
		    X = self.conv15(X)
		    X = self.relu(X)
		    #print(X.size())
		    X = X + sources[-2]
		    X = self.conv16(X)
		    X = self.relu(X)

		    X = self.upsample2D(X)
		    X = self.conv17(X)
		    X = self.relu(X)
		    #print(X.size())
		    X = X + sources[-3]
		    X = self.conv18(X)
		    X = self.relu(X)

		    X = self.upsample2D(X)
		    X = self.conv19(X)
		    X = self.relu(X)
		    #print(X.size())
		    X = X + sources[-4]
		    X = self.conv20(X)
		    X = self.relu(X)

		    X = self.upsample2D(X)
		    X = self.conv21(X)
		    X = self.relu(X)
		    #print(X.size())
		    X = X + sources[-5]
		    X = self.conv22(X)
		    X = self.relu(X)

		    X = self.conv23(X)
		    ##X = self.relu(X)
		    #print(X.size())
		    out = X#self.tanh(X)
		    
		    out_processed = torch.cat((self.sigmoid(out[:,:2,:,:]),(out[:,2:,:,:])),1)

		    return out_processed