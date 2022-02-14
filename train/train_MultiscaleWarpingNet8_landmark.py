import sys
import os
from optparse import OptionParser
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import time
import cPickle as pickle
import math
import random
import cv2

sys.path.insert(0,'./ref_utils/')
sys.path.insert(0,'./Model/')
from Model import MultiscaleWarpingNet8 

from LFDataset import LFDataset
from FlowerDataset import FlowerDataset
from SintelDataset import SintelDataset
from GigaDataset import GigaDataset
import matplotlib.pyplot as plt
import CustomLoss 
from sift_extractor import SiftExtractor


def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    
    if mse > 1000:
        return -100
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


def pre_align(buff,sift_extractor):

    input_img1_HR = np.array(buff['input_img1_HR'] * 255, dtype=np.uint8)
    input_img2_HR = np.array(buff['input_img2_HR'] * 255 ,dtype = np.uint8)

    B,C,H,W = input_img1_HR.shape
    for b in range(B):

        img1 = input_img1_HR[b].transpose(1,2,0)
        img2 = input_img2_HR[b].transpose(1,2,0)
        lm1,lm2 = sift_extractor.get_matched_landmark(img1,img2)
        if lm1 is None and lm2 is None:
            continue

        disparity = lm2 - lm1
        avg_dis = np.mean(disparity,axis=1)
        print (disparity.shape,np.mean(disparity,axis=0),np.mean(disparity,axis=1))

def pre_align2(buff):

    input_img1_LR = np.array(buff['input_img1_LR'] * 255, dtype=np.uint8)
    input_img2_HR = np.array(buff['input_img2_HR'] * 255 ,dtype = np.uint8)
    B,C,h,w = input_img1_LR.shape
    new_img2_HR = np.zeros(input_img2_HR.shape).astype(np.float32)
    for b in range(B):
        img1 = input_img1_LR[b].transpose(1,2,0)
        img2 = input_img2_HR[b].transpose(1,2,0)
        template = img2[h/4:h/4+h/2,w/4:w/4+h/2,:]
        res = cv2.matchTemplate(img1, template, cv2.TM_CCORR_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        left_top = max_loc
        padded_img2 = np.zeros((h*3,w*3,C)).astype(np.uint8)
        padded_img2[h:2*h,w:2*w,:] = img2
        dis_x = left_top[0] - w/4
        dis_y = left_top[1] - h/4
        cropped_img = padded_img2[h-dis_y:2*h-dis_y,w-dis_x:2*w-dis_x,:]
        new_img2_HR[b] = np.array(cropped_img.transpose(2,0,1),dtype=np.float32) / 255.0

    buff['input_img2_HR'] = new_img2_HR

    return buff


 
def gen_flow_label(sift_extractor, buff, flow):

    input_img1_HR = np.array(buff['input_img1_HR'] * 255, dtype=np.uint8)
    input_img2_HR = np.array(buff['input_img2_HR'] * 255 ,dtype = np.uint8)

    flow_label = flow.detach().cpu().numpy().copy()
    B,C,H,W = input_img1_HR.shape
    for b in range(B):

        img1 = input_img1_HR[b].transpose(1,2,0)
        img2 = input_img2_HR[b].transpose(1,2,0)
        lm1,lm2 = sift_extractor.get_matched_landmark(img1,img2)
        if lm1 is None and lm2 is None:
            continue

        disparity = lm2 - lm1
        for idx in range(disparity.shape[0]):
            flow_label[b,:,lm1[idx,1],lm1[idx,0]] = disparity[idx,:]
    
    return torch.from_numpy(flow_label)


def train_net(net, gpu=False, config={}):

    dataset_train = config['dataset_train']
    dataset_test = config['dataset_test'] 

    print('Starting training...')
    
    if config['optim'] == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=config['lr'], momentum=0.9, weight_decay=0.0005)
    elif config['optim'] == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr = config['lr'], weight_decay = 0.00005)    


    if config['loss'] == 'EuclideanLoss':
        criterion = CustomLoss.EuclideanLoss()
    elif config['loss'] == 'CharbonnierLoss':
        criterion = CustomLoss.CharbonnierLoss()
    elif config['loss'] == 'MSELoss':
        criterion = nn.MSELoss()
    else:
        print 'None loss type'
        sys.exit(0)

    sift_extractor = config['sift_extractor']
    criterion_warp = CustomLoss.EuclideanLoss()
    loss_count = np.zeros(3,dtype=np.float32)
    time_start = time.time()

    for iter_ in range(config['checkpoint'],config['max_iter']):


        if config['specified_view'] is None and config['view_mode']== 'specified':
            view = random.randint(1,7)
        else:
            view = config['specified_view'] 

        if config['dataset_name'] == 'Flower' or config['dataset_name'] == 'LFvideo': 
            buff = dataset_train.nextBatch_new(batchsize=config['batch_size'], shuffle=True, view_mode = config['view_mode'],specified_view=view, augmentation = True, crop_shape = config['train_data_crop_shape'],Dual = config['Dual'],checkpoint = config['checkpoint'])
        elif config['dataset_name'] == 'Sintel':
            buff = dataset_train.nextBatch_new(batchsize=config['batch_size'], shuffle=True, view_mode = config['view_mode'], augmentation = True, crop_shape = config['train_data_crop_shape'],Dual = config['Dual'])
        elif config['dataset_name'] == 'Giga':
            buff = dataset_train.nextBatch_new(batchsize=config['batch_size'], shuffle=False, view_mode = 'Random',specified_view=None, augmentation = False, crop_shape = config['train_data_crop_shape'],Dual = config['Dual'])
            buff = pre_align2(buff)
        label_img = buff['input_img1_HR']
        label_img = torch.from_numpy(label_img)

        if gpu:
            label_img = label_img.cuda()

        warp_img2_HR,fine_img1_SR,flow_s1_12_1 = net(buff,require_flow=True,encoder_input=config['encoder_input'])
        
        flow_label = gen_flow_label(sift_extractor,buff,flow_s1_12_1)
        if gpu:
            flow_label = flow_label.cuda()
        

        loss_1 = criterion_warp(warp_img2_HR, label_img)
        loss_2 = criterion_warp(flow_s1_12_1, flow_label)       
        loss_3 = criterion(fine_img1_SR, label_img)


        loss_count[0] += config['w1'] * loss_1.item()
        loss_count[1] += loss_2.item()
        loss_count[2] += config['w2'] * loss_3.item()


        loss = config['w1'] * loss_1 + loss_2 + config['w2'] * loss_3

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (iter_ + 1) % config['snapshot'] == 0:
            torch.save(net.state_dict(),
                       config['checkpoints_dir'] + 'CP{}.pth'.format(iter_ + 1))
            print('Checkpoint {} saved !'.format(iter_ + 1))
        
        
        if (iter_ + 1) % config['display'] == 0:
            time_end = time.time()    
            time_cost = time_end - time_start
            
            #------------------------------------------------
            pre_npy_2 = fine_img1_SR.data.cpu().numpy()
            label_img_npy = label_img.data.cpu().numpy()

            psnr_2 = 0
            for i in range(pre_npy_2.shape[0]):
                psnr_2 += psnr(pre_npy_2[i],label_img_npy[i]) / pre_npy_2.shape[0]
       
            loss_count = loss_count / config['display']
            print ('iter:%d    time: %.2fs / %diters   lr: %.8f   %s: %.3f  %.3f  %.3f  psnr: %.2f'%(iter_ + 1,time_cost,config['display'],config['lr'],config['loss'],loss_count[0],loss_count[1],loss_count[2], psnr_2))
            loss_count[:] = 0
            time_start = time.time()

        if (iter_ + 1) % config['step_size'] == 0:
            config['lr'] = config['lr'] * config['gamma']
            if config['optim'] == 'SGD':
                optimizer = optim.SGD(net.parameters(), lr=config['lr'] * config['gamma'], momentum=0.9, weight_decay=0.0005)
            elif config['optim'] == 'Adam':
                optimizer = optim.Adam(net.parameters(), lr = config['lr'], weight_decay = 0.00005)





def get_args():
    parser = OptionParser()

    parser.add_option('--batch_size', dest='batch_size', default=8,
                      type='int', help='batch size')
    parser.add_option('--lr', dest='lr', default=0.0001,
                      type='float', help='learning rate')
    parser.add_option('--gpu', action='store_true', dest='gpu',
                      default=True, help='use cuda')
    parser.add_option('--checkpoint_file', dest='load',
                      default=False, help='load file model')

    parser.add_option('--checkpoint',dest = 'checkpoint',default = 0,type = 'int',help = 'snapshot')
    parser.add_option('--scale', dest='scale', type='int',
                      default= 8 , help='downscaling factor of LR')

    parser.add_option('--loss',dest = 'loss',default='EuclideanLoss',help = 'loss type')

    parser.add_option('--dataset',dest = 'dataset',default = 'LFvideo',help = 'dataset type')

    parser.add_option('--gamma',dest = 'gamma',type = 'float', default = 0.2,help = 'lr decay')

    parser.add_option('--step_size',dest = 'step_size',type = 'float',default = 60000,help = 'step_size')

    parser.add_option('--max_iter',dest = 'max_iter',default = 1000000,type = 'int',help = 'max_iter')

    parser.add_option('--checkpoints_dir',dest = 'checkpoints_dir',default = './checkpoints/',help = 'checkpoints_dir')

    parser.add_option('--snapshot',dest = 'snapshot',default = 5000,type = 'float',help = 'snapshot')

    parser.add_option('--display',dest = 'display',default = 10,type = 'float',help = 'display')
    
    parser.add_option('--optim', dest = 'optim', default = 'SGD', help = 'optimizer type')  

    parser.add_option('--pretrained', dest = 'pretrained', default = None, help = 'optimizer type')   

    parser.add_option('--w1', dest = 'w1', default = 1.0, type = 'float',help = 'coarse weight')
    parser.add_option('--w2', dest = 'w2', default = 1.0, type = 'float',help = 'fine weight')
    parser.add_option('--gpu_id', dest = 'gpu_id', default = 0, type = 'int',help = 'gpu_id')
    parser.add_option('--view_mode', dest = 'view_mode', default = 'Random' , help = 'view_mode')
    parser.add_option('--specified_view', dest = 'specified_view', default = None,type='int' , help = 'view_mode')
    parser.add_option('--flownet_type', dest = 'flownet_type', default = 'FlowNet_ori' , help = 'FlowNet_ori or FlowNet_dilation')
    parser.add_option('--encoder_input', dest = 'encoder_input', default = 'input_img1_SR' , help = 'input_img1_SR or input_img1_LR')
    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':


    args = get_args()

    if args.gpu_id != 0:
        torch.cuda.set_device(args.gpu_id)

    print ('flownet_type', args.flownet_type)
    net = MultiscaleWarpingNet8(flownet_type = args.flownet_type)
    dataset_name = args.dataset
    scale = args.scale
    
    if dataset_name=='LFvideo':
        dataset_train = LFDataset(filename = './dataset/lf_video_dataset/train_x4_x8.h5', scale = scale)
        dataset_test = LFDataset(filename = './dataset/lf_video_dataset/test_x4_x8.h5', scale = scale)
        H,W = (320,512)

    elif dataset_name=='Flower':
        dataset_train = FlowerDataset(filename = './dataset/flower_dataset/train_x4_x8.h5', scale = scale)
        dataset_test = FlowerDataset(filename = './dataset/flower_dataset/test_x4_x8.h5', scale = scale)
        H,W = (320,512)

    elif dataset_name=='Sintel':
        dataset_train = SintelDataset(filename = './dataset/sintel_dataset/train_x4_x8.h5', scale = scale)
        dataset_test = SintelDataset(filename = './dataset/sintel_dataset/test_x4_x8.h5', scale = scale)
        H,W = (384,1024)

    elif dataset_name == 'Giga':

        dataset_train = GigaDataset(filename = './datasets/giga_dataset/20190220_to_20190304/train.h5')
        dataset_test = GigaDataset(filename = './datasets/giga_dataset/20190220_to_20190304/test.h5')
        H,W = (640,1024)

    config = {}
    config['dataset_name'] = args.dataset
    config['dataset_train'] = dataset_train
    config['dataset_test'] = dataset_test
    #config['data_displacement_augmentation'] = False
    config['train_data_crop_shape'] = [H,W]
    config['max_iter'] = args.max_iter
    config['snapshot'] = args.snapshot
    config['display'] = args.display
    config['lr'] = args.lr
    config['batch_size'] = args.batch_size
    config['step_size'] = args.step_size
    config['gamma'] = args.gamma
    config['checkpoints_dir'] = args.checkpoints_dir
    config['loss'] = args.loss
    config['checkpoint'] = args.checkpoint
    config['optim'] = args.optim
    config['Dual'] = False
    config['w1'] = args.w1
    config['w2'] = args.w2
    config['view_mode'] = args.view_mode
    config['specified_view'] = args.specified_view
    config['sift_extractor'] = SiftExtractor()    
    config['encoder_input'] = args.encoder_input

    print ('-------config-----------')
    for k,v in sorted(config.items()):
        print (k,v)
 
    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))
        

    if args.pretrained:

        MW_2stage_model = torch.load('./checkpoints_exp18/CP325000.pth')
        cur_model = net.state_dict()
        #print MW_2stage_model.keys()
        FlowNet_s1 = {'FlowNet_s1.'+k[21::]:v for k,v in MW_2stage_model.items() if 'FlowNet_s1.'+k[21::] in cur_model and k[0:21]=='MWNet_coarse.FlowNet.'}
        FlowNet_s2 = {'FlowNet_s2.'+k[19::]:v for k,v in MW_2stage_model.items() if 'FlowNet_s2.'+k[19::] in cur_model and k[0:19]=='MWNet_fine.FlowNet.'}

        encoder_decoder = {k[11::]:v for k,v in MW_2stage_model.items() if k[11::] in cur_model }        
        #print FlowNet_s1.keys(),FlowNet_s2.keys()
        print encoder_decoder
        cur_model.update(encoder_decoder)
        cur_model.update(FlowNet_s1)
        cur_model.update(FlowNet_s2)
        net.load_state_dict(cur_model)
        print 'finetuning...'

    if args.gpu:
        net.cuda()


    try:
        train_net(net=net,gpu=args.gpu,config = config)

    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
