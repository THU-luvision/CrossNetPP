import torch
import torch.nn as nn
import torch.nn.functional as F
from net_utils import *
from FlowNet_model import FlowNet
from FlowNet_model import FlowNet_dilation
from Backward_warp_layer import Backward_warp
from SupervisedFlowNet import SupervisedFlowNet


class MultiscaleWarpingNet8(nn.Module):

    def __init__(self, flownet_type = 'FlowNet_ori'):
        super(MultiscaleWarpingNet8, self).__init__()
        
        if flownet_type == 'FlowNet_ori':
            self.FlowNet_s1 = FlowNet(6)
            self.FlowNet_s2 = FlowNet(6)
        elif flownet_type == 'FlowNet_dilation':
            self.FlowNet_s1 = FlowNet_dilation(6)
            self.FlowNet_s2 = FlowNet_dilation(6)
        self.Backward_warp = Backward_warp()
        self.Encoder = Encoder(3)
        self.UNet_decoder_2 = UNet_decoder_2()

    def forward(self, buff,vimeo=False, require_flow=False, encoder_input = 'input_img1_SR',flow_visible=False ):

        if vimeo:
            input_img1_LR = buff['input_img1_LR'].cuda()
            input_img1_HR = buff['input_img1_HR'].cuda()
            input_img1_SR = buff['input_img1_SR'].cuda()
            input_img2_HR = buff['input_img2_HR'].cuda()
        else:
            input_img1_LR = torch.from_numpy(buff['input_img1_LR']).cuda()
            #input_img1_HR = torch.from_numpy(buff['input_img1_HR']).cuda()
            input_img1_SR = torch.from_numpy(buff[encoder_input]).cuda()
            input_img2_HR = torch.from_numpy(buff['input_img2_HR']).cuda()

        
        flow_s1 = self.FlowNet_s1(input_img1_LR, input_img2_HR)

        flow_s1_12_1 = flow_s1['flow_12_1']

        warp_img2_HR = self.Backward_warp(input_img2_HR, flow_s1_12_1)

        flow_s2 = self.FlowNet_s2(input_img1_LR, warp_img2_HR)
        flow_s2_12_1 = flow_s2['flow_12_1']
        flow_s2_12_2 = flow_s2['flow_12_2']
        flow_s2_12_3 = flow_s2['flow_12_3']
        flow_s2_12_4 = flow_s2['flow_12_4']

        SR_conv1, SR_conv2, SR_conv3, SR_conv4 = self.Encoder(input_img1_SR)
        HR2_conv1, HR2_conv2, HR2_conv3, HR2_conv4 = self.Encoder(warp_img2_HR)

        warp_s2_21_conv1 = self.Backward_warp(HR2_conv1, flow_s2_12_1)
        warp_s2_21_conv2 = self.Backward_warp(HR2_conv2, flow_s2_12_2)
        warp_s2_21_conv3 = self.Backward_warp(HR2_conv3, flow_s2_12_3)
        warp_s2_21_conv4 = self.Backward_warp(HR2_conv4, flow_s2_12_4)
        refSR_2 = self.UNet_decoder_2(SR_conv1, SR_conv2, SR_conv3, SR_conv4, warp_s2_21_conv1,warp_s2_21_conv2, warp_s2_21_conv3,warp_s2_21_conv4)

        if flow_visible:
            return warp_img2_HR,refSR_2,flow_s1,flow_s2
        else:
            if require_flow:
                return warp_img2_HR,refSR_2,flow_s1_12_1
            else:
                return warp_img2_HR,refSR_2


