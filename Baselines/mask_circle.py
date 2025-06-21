import numpy as np
import cv2
import scipy.signal
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.transforms import Compose, ToTensor, Normalize
from PIL import Image
import math


def fix_image(image):
    w, h = image.size
    aspect_ratio = float(w)/float(h)
    if aspect_ratio >= 1.333 and aspect_ratio <= 1.334:
        result_im = image.resize((640, 480))
    elif aspect_ratio < 1.333:
        w_new = h * (4.0/3.0)
        w_pad = (w_new - w) / 2
        result_im = Image.new(image.mode, (int(w_new), h), 127)
        result_im.paste(image, (int(w_pad), 0))
        result_im = result_im.resize((640, 480))
    else:
        h_new = w * (3.0/4.0)
        h_pad = (h_new - h) / 2
        result_im = Image.new(image.mode, (w, int(h_new)), 127)
        result_im.paste(image, (0, int(h_pad)))
        result_im = result_im.resize((640, 480))
    return result_im

class ResBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.conv_res = nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False)
        
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)
        x = (x + res) * (1 / math.sqrt(2))
        return x

class NestedResUNetParam(nn.Module):
    def __init__(self, num_channels, num_params=6, width=32, resolution=(240, 320)):
        super().__init__()

        self.resolution = resolution
        nb_filter = [width, width*2, width*4, width*8, width*16]

        self.pool = nn.MaxPool2d(2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv0_0 = ResBlock(num_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = ResBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = ResBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = ResBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = ResBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = ResBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = ResBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = ResBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = ResBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = ResBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = ResBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = ResBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = ResBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = ResBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = ResBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])
        
        self.xyr_input = ResBlock(num_channels, nb_filter[0], nb_filter[0])      
        self.xyr_conv1 = ResBlock(nb_filter[0]*5, nb_filter[0]*4, nb_filter[0]*4)
        self.xyr_conv2 = ResBlock(nb_filter[0]*4, nb_filter[0]*3, nb_filter[0]*3)
        self.xyr_conv3 = ResBlock(nb_filter[0]*3, nb_filter[0]*2, nb_filter[0]*2)
        self.xyr_conv4 = ResBlock(nb_filter[0]*2, nb_filter[0], nb_filter[0])
        self.xyr_linear = nn.Sequential(
                              nn.Flatten(),
                              nn.Linear(nb_filter[0]*int(resolution[0]/16)*int(resolution[1]/16), int(resolution[0]/16)*int(resolution[1]/16)),
                              nn.ReLU(),
                              nn.Linear(int(resolution[0]/16)*int(resolution[1]/16), num_params)
                          )


    def forward(self, input):
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        
        xyr0 = self.xyr_input(input)
        xyr1 = self.xyr_conv1(torch.cat([xyr0, x0_1, x0_2, x0_3, x0_4], 1)) #240x320
        xyr2 = self.xyr_conv2(self.pool(xyr1)) #120x160
        xyr3 = self.xyr_conv3(self.pool(xyr2)) #60x80
        xyr4 = self.xyr_conv4(self.pool(xyr3)) #30x40
        xyr5 = self.xyr_linear(self.pool(xyr4)) #15x20
          
        return xyr5

class SharedAtrousConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.weights = nn.Parameter(torch.rand(out_channels, in_channels, 3, 3))
        nn.init.kaiming_normal_(self.weights, mode='fan_out', nonlinearity='relu')
        if bias:
            self.bias1 = nn.Parameter(torch.rand(out_channels))
            nn.init.zeros_(self.bias1)
            self.bias2 = nn.Parameter(torch.rand(out_channels))
            nn.init.zeros_(self.bias2)
        else:
            self.bias1 = None
            self.bias2 = None
        self.join_net = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1, stride=1, padding='same', bias=bias)
        nn.init.kaiming_normal_(self.join_net.weight, mode='fan_in', nonlinearity='relu')
    def forward(self, x):
        x1 = nn.functional.conv2d(x, self.weights, stride=1, padding='same', bias=self.bias1)
        x2 = nn.functional.conv2d(x, self.weights, stride=1, padding='same', dilation=2, bias=self.bias2)
        x3 = self.join_net(torch.cat([x1, x2], 1))
        return x3

class SharedAtrousResBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.conv_res = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.net = nn.Sequential(
            SharedAtrousConv2d(in_channels, middle_channels, bias=False),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            SharedAtrousConv2d(middle_channels, out_channels, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)
        x = (x + res) * (1 / math.sqrt(2))
        x = self.relu(x)
        return x
    
class Resize(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None, recompute_scale_factor=None, antialias=False):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.recompute_scale_factor = recompute_scale_factor
        self.antialias = antialias
    def forward(self, x):
        return torch.nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners, recompute_scale_factor=self.recompute_scale_factor)        
class NestedSharedAtrousResUNet(nn.Module):
    def __init__(self, num_classes, num_channels, width=32, resolution=(240, 320)):
        super().__init__()
        self.resolution = resolution
        nb_filter = [width, width*2, width*4, width*8, width*16]

        self.pool = Resize(scale_factor=0.5, mode='bilinear')
        self.up = Resize(scale_factor=2, mode='bilinear')

        self.conv0_0 = SharedAtrousResBlock(num_channels, nb_filter[0], nb_filter[0])
        self.conv1_0 = SharedAtrousResBlock(nb_filter[0], nb_filter[1], nb_filter[1])
        self.conv2_0 = SharedAtrousResBlock(nb_filter[1], nb_filter[2], nb_filter[2])
        self.conv3_0 = SharedAtrousResBlock(nb_filter[2], nb_filter[3], nb_filter[3])
        self.conv4_0 = SharedAtrousResBlock(nb_filter[3], nb_filter[4], nb_filter[4])

        self.conv0_1 = SharedAtrousResBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_1 = SharedAtrousResBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_1 = SharedAtrousResBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv3_1 = SharedAtrousResBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])

        self.conv0_2 = SharedAtrousResBlock(nb_filter[0]*2+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_2 = SharedAtrousResBlock(nb_filter[1]*2+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv2_2 = SharedAtrousResBlock(nb_filter[2]*2+nb_filter[3], nb_filter[2], nb_filter[2])

        self.conv0_3 = SharedAtrousResBlock(nb_filter[0]*3+nb_filter[1], nb_filter[0], nb_filter[0])
        self.conv1_3 = SharedAtrousResBlock(nb_filter[1]*3+nb_filter[2], nb_filter[1], nb_filter[1])

        self.conv0_4 = SharedAtrousResBlock(nb_filter[0]*4+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0]*4, num_classes, kernel_size=1)

    def forward(self, input):
        
        x0_0 = self.conv0_0(input)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3)], 1))
        
        output = self.final(torch.cat([x0_1, x0_2, x0_3, x0_4], 1))
        
        return output

class MaskCircleFinder(object):
    def __init__(self, mask_net_path = './nestedsharedatrousresunet-017-0.02902-maskIoU-0.938773.pth', circle_net_path = './nestedresunetparam-604-0.049-pupilIoU-0.907-irisIoU-0.947.pth', device=torch.device('cuda')):
        self.circle_net_path = circle_net_path
        self.mask_net_path = mask_net_path
        self.device = device
        self.NET_INPUT_SIZE = (320,240)
        self.circle_model = NestedResUNetParam(1, 6)
        try:
            self.circle_model.load_state_dict(torch.load(self.circle_net_path, map_location=self.device))
        except AssertionError:
                print("assertion error")
                self.circle_model.load_state_dict(torch.load(self.circle_net_path,
                    map_location = lambda storage, loc: storage))
        self.circle_model = self.circle_model.to(self.device)
        self.circle_model.eval()
        self.mask_model = NestedSharedAtrousResUNet(1, 1)
        try:
            self.mask_model.load_state_dict(torch.load(self.mask_net_path, map_location=self.device))
        except AssertionError:
                print("assertion error")
                self.mask_model.load_state_dict(torch.load(self.mask_net_path,
                    map_location = lambda storage, loc: storage))
        self.mask_model = self.mask_model.to(self.device)
        self.mask_model.eval()
        self.input_transform_mask = Compose([
            ToTensor(),
            #Normalize(mean=(0.5791223733793273,), std=(0.21176097694558188,))
            Normalize(mean=(0.5,), std=(0.5,))
        ])
        self.input_transform_circ = Compose([
            ToTensor(),
            Normalize(mean=(0.5791223733793273,), std=(0.21176097694558188,))
            #Normalize(mean=(0.5,), std=(0.5,))
        ])
    
    def getMask(self, image):
        w,h = image.size   
        # if image is not grayscale, convert it
        if image.mode == 'RGB':
            image = image.convert('L')

        image = cv2.resize(np.array(image), self.NET_INPUT_SIZE, cv2.INTER_LINEAR_EXACT) 
        if image.ndim == 3:
            image = image[:, :, 1]

        mask_logit_t = self.mask_model(Variable(self.input_transform_mask(image).unsqueeze(0).to(self.device)))[0]
        mask_t = torch.where(torch.sigmoid(mask_logit_t) > 0.5, 255, 0)
        mask = mask_t.cpu().numpy()[0]
        mask = np.uint8(mask)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST_EXACT)
        mask_pil = Image.fromarray(mask, 'L')
        #print('Mask Shape: ', mask.shape, 'Mask Size: ', mask.size)

        return mask_pil

    def circApprox(self, image):
        w,h = image.size
        # if image is not grayscale, convert it
        if image.mode == 'RGB':
            image = image.convert('L')

        image = cv2.resize(np.array(image), self.NET_INPUT_SIZE, cv2.INTER_LINEAR_EXACT)
        if image.ndim == 3:
            image = image[:, :, 1]
        
        w_mult = w/self.NET_INPUT_SIZE[0]
        h_mult = h/self.NET_INPUT_SIZE[1]

        inp_xyr_t = self.circle_model(Variable(self.input_transform_circ(image).unsqueeze(0).to(self.device)))

        #Circle params
        inp_xyr = inp_xyr_t.tolist()[0]
        pupil_x = int(inp_xyr[0] * w_mult)
        pupil_y = int(inp_xyr[1] * h_mult)
        pupil_r = int(inp_xyr[2] * max(w_mult, h_mult))
        iris_x = int(inp_xyr[3] * w_mult)
        iris_y = int(inp_xyr[4] * h_mult)
        iris_r = int(inp_xyr[5] * max(w_mult, h_mult))

        return np.array([pupil_x,pupil_y,pupil_r]), np.array([iris_x,iris_y,iris_r])
    
    def segment_and_circApprox(self, image, dataset='warsaw'):
        
        if dataset == 'qfire':
            image = fix_image(image)
        mask = self.getMask(image)   
        pupil_xyr, iris_xyr = self.circApprox(image);
        return mask, pupil_xyr, iris_xyr