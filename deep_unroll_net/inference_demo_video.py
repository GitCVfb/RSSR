import os
import torch
import random
import argparse
import numpy as np
from skimage import io
import shutil

import imageio
import cv2
import flow_viz
import softsplat

from package_core.generic_train_test import *
from forward_warp_package import *
from dataloader import *
from model_RSSR import *
from frame_utils import *
from Convert_m_t import *

##===================================================##
##********** Configure training settings ************##
##===================================================##
parser=argparse.ArgumentParser()
parser.add_argument('--batch_sz', type=int, default=1, help='batch size used for training')
parser.add_argument('--continue_train', type=bool, default=True, help='flags used to indicate if train model from previous trained weight')
parser.add_argument('--is_training', type=bool, default=False, help='flag used for selecting training mode or evaluation mode')
parser.add_argument('--n_chan', type=int, default=3, help='number of channels of input/output image')
parser.add_argument('--n_init_feat', type=int, default=32, help='number of channels of initial features')
parser.add_argument('--seq_len', type=int, default=2)
parser.add_argument('--shuffle_data', type=bool, default=False)
parser.add_argument('--crop_sz_H', type=int, default=448, help='cropped image size height')
parser.add_argument('--crop_sz_W', type=int, default=640, help='cropped image size width')

parser.add_argument('--model_label', type=str, required=True)
parser.add_argument('--log_dir', type=str, required=True)
parser.add_argument('--results_dir', type=str, required=True)
parser.add_argument('--data_dir', type=str, required=True)

parser.add_argument('--is_Fastec', type=int, default=0)

opts=parser.parse_args()

##===================================================##
##****************** Create model *******************##
##===================================================##
model=ModelRSSR(opts)

##===================================================##
##**************** Train the network ****************##
##===================================================##
class Demo(Generic_train_test):
    def test(self):
        with torch.no_grad():
            seq_lists = os.listdir(self.opts.data_dir)
            for seq in seq_lists:
                im_rs0_path = os.path.join(os.path.join(self.opts.data_dir, seq), 'rs_0.png')
                im_rs1_path = os.path.join(os.path.join(self.opts.data_dir, seq), 'rs_1.png')

                im_rs0 = torch.from_numpy(io.imread(im_rs0_path).transpose(2,0,1))[:3,:,:].unsqueeze(0).clone()
                im_rs1 = torch.from_numpy(io.imread(im_rs1_path).transpose(2,0,1))[:3,:,:].unsqueeze(0).clone()

                im_rs = torch.cat([im_rs0,im_rs1], dim=1).float()/255.
                
                if self.opts.is_Fastec==1:
                    im_rs0 = F.interpolate(im_rs[:,0:3,:,:],size=[448,640], mode='bilinear')#
                    im_rs1 = F.interpolate(im_rs[:,3:6,:,:],size=[448,640], mode='bilinear')#
                    im_rs  = torch.cat((im_rs0,im_rs1),1).clone()#
                    
                _input = [im_rs, None, None, None]
                
                self.model.set_input(_input)
                pred_im, pred_mask, pred_flow, pred_scale = self.model.forward()
                
                # save original RS images
                im_rs_0 = io.imread(im_rs0_path)
                im_rs_1 = io.imread(im_rs1_path)
                io.imsave(os.path.join(self.opts.results_dir, seq+'_rs_0.png'), im_rs_0)
                io.imsave(os.path.join(self.opts.results_dir, seq+'_rs_1.png'), im_rs_1)
                
                # generate GS images for any row
                preds_0=[]
                preds_1=[]
                preds_0_tensor=[]
                preds_1_tensor=[]
                im_rs_clone = im_rs.clone().cuda()
                copies = 11
                for t in range(0,copies):
                    #convert to GS of t-th row
                    tt=self.opts.crop_sz_H*t/(copies-1) #Generated Frame number: copies
                    pred_gs_t, pred_flow_t = self.convert_m2t(im_rs_clone, pred_flow, pred_scale, tt)
                    
                    if self.opts.is_Fastec==1:
                        for i in range(2):
                            pred_gs_t[i] = F.interpolate(pred_gs_t[i],size=[480,640], mode='bilinear')#
                    
                    # save results
                    io.imsave(os.path.join(self.opts.results_dir, seq+'_pred_0_'+str(t)+'.png'), (pred_gs_t[0].clamp(0,1).cpu().numpy().transpose(0,2,3,1)[0]*255).astype(np.uint8))
                    io.imsave(os.path.join(self.opts.results_dir, seq+'_pred_1_'+str(t)+'.png'), (pred_gs_t[1].clamp(0,1).cpu().numpy().transpose(0,2,3,1)[0]*255).astype(np.uint8))
                    
                    if t != copies-1:
                        preds_0.append((pred_gs_t[0].clamp(0,1).cpu().numpy().transpose(0,2,3,1)[0]*255).astype(np.uint8))
                        preds_0_tensor.append(pred_gs_t[0])
                    preds_1.append((pred_gs_t[1].clamp(0,1).cpu().numpy().transpose(0,2,3,1)[0]*255).astype(np.uint8))
                    preds_1_tensor.append(pred_gs_t[1])
                    
                    print('saved', self.opts.results_dir, seq+'_pred_'+str(t)+'.png')
                    
                
                pred_imgs_rec=preds_0_tensor
                for t in range(0,copies):
                    pred_imgs_rec.append(preds_1_tensor[t])
                
                img_rec = pred_imgs_rec[0].clone()
                for t in range(1,copies*2-1):
                    img_rec *= (pred_imgs_rec[t] * 255)
                    img_rec = img_rec.clamp(0,1)
                
                x_min, y_min, x_max, y_max = self.cut_img_without_margin(img_rec.squeeze(0))
                print(x_min, y_min, x_max, y_max)
                
                preds_0_crop=[]
                preds_1_crop=[]
                for t in range(0,copies-1):
                    preds_0_crop.append((preds_0_tensor[t][:,:,y_min:y_max,x_min:x_max].clamp(0,1).cpu().numpy().transpose(0,2,3,1)[0]*255).astype(np.uint8))
                    # save results
                    io.imsave(os.path.join(self.opts.results_dir, seq+'_pred_crop_0_'+str(t)+'.png'), (preds_0_tensor[t][:,:,y_min:y_max,x_min:x_max].clamp(0,1).cpu().numpy().transpose(0,2,3,1)[0]*255).astype(np.uint8))
                for t in range(0,copies):
                    preds_1_crop.append((preds_1_tensor[t][:,:,y_min:y_max,x_min:x_max].clamp(0,1).cpu().numpy().transpose(0,2,3,1)[0]*255).astype(np.uint8))
                    # save results
                    io.imsave(os.path.join(self.opts.results_dir, seq+'_pred_crop_1_'+str(t)+'.png'), (preds_1_tensor[t][:,:,y_min:y_max,x_min:x_max].clamp(0,1).cpu().numpy().transpose(0,2,3,1)[0]*255).astype(np.uint8))
                
                
                #make gif
                make_gif_flag = True
                if make_gif_flag:
                    pred_imgs_gif=preds_0
                    pred_imgs_gif_crop=preds_0_crop
                    for t in range(0,copies):
                        pred_imgs_gif.append(preds_1[t])
                        pred_imgs_gif_crop.append(preds_1_crop[t])
                    
                    #Save generated GS video images
                    #imageio.mimsave(os.path.join(self.opts.results_dir, seq+'_duration_'+str(0.5)+'.gif'), pred_imgs_gif, duration = 0.5) # modify the frame duration as needed
                    #imageio.mimsave(os.path.join(self.opts.results_dir, seq+'_duration_'+str(0.1)+'.gif'), pred_imgs_gif, duration = 0.1) # modify the frame duration as needed

                    #Save cropped GS video images
                    imageio.mimsave(os.path.join(self.opts.results_dir, seq+'_crop_duration_'+str(0.5)+'.gif'), pred_imgs_gif_crop, duration = 0.5)
                    imageio.mimsave(os.path.join(self.opts.results_dir, seq+'_crop_duration_'+str(0.1)+'.gif'), pred_imgs_gif_crop, duration = 0.1)



    def convert_m2t(self, im_rs, pred_flow, pred_scale, t):
        T_0_t = scale_from_m2t(pred_scale[0], self.opts.crop_sz_H, self.opts.crop_sz_W, t)
        T_1_t = scale_from_m2t(pred_scale[1], self.opts.crop_sz_H, self.opts.crop_sz_W, t)
        
        F_0_t = pred_flow[2] * T_0_t
        F_1_t = pred_flow[3] * T_1_t
        
        tenMetric_0_1 = torch.nn.functional.l1_loss(input=im_rs[:,0:3,:,:], target=backwarp(tenInput=im_rs[:,3:6,:,:], tenFlow=pred_flow[2]), reduction='none').mean(1, True)
        tenMetric_1_0 = torch.nn.functional.l1_loss(input=im_rs[:,3:6,:,:], target=backwarp(tenInput=im_rs[:,0:3,:,:], tenFlow=pred_flow[3]), reduction='none').mean(1, True)
        g_I0t = softsplat.FunctionSoftsplat(tenInput=im_rs[:,0:3,:,:], tenFlow=F_0_t, tenMetric=-20.0 * tenMetric_0_1, strType='softmax')
        g_I1t = softsplat.FunctionSoftsplat(tenInput=im_rs[:,3:6,:,:], tenFlow=F_1_t, tenMetric=-20.0 * tenMetric_1_0, strType='softmax')
        
        
        return [g_I0t, g_I1t], [F_0_t, F_1_t]
    
        
    def cut_img_without_margin(self, img_pre):
        """
        Extract the largest inscribed rectangle according to all corrected images with multiplied their corresponding masks 
        Input: tensor(3*H*W), image with black edges
        """
        
        img_bgr = img_pre.cpu().numpy().copy()
        
        img_bgr = (img_bgr * 255.0).astype(np.uint8)
        img_bgr_  = img_bgr.transpose(1,2,0).copy()
        img_bgr  = img_bgr.transpose(1,2,0)
        
        h, w, c = img_bgr.shape
        
        img_bgr  = (img_bgr * np.random.rand(h, w, c)).astype(np.uint8)
        #print(img_bgr[:,:,0])
        
        img_copy = img_bgr.copy()
        
        mask = np.zeros((h + 2, w + 2), np.uint8)  # Mask
        cv2.floodFill(img_copy, mask=mask, seedPoint=(0, 0), newVal=(0, 0, 255),
                      loDiff=(1, 1, 1), upDiff=(1, 1, 1))
        cv2.floodFill(img_copy, mask=mask, seedPoint=(w-1, 0), newVal=(0, 0, 255),
                      loDiff=(1, 1, 1), upDiff=(1, 1, 1))
        cv2.floodFill(img_copy, mask=mask, seedPoint=(0, h-1), newVal=(0, 0, 255),
                      loDiff=(1, 1, 1), upDiff=(1, 1, 1))
        cv2.floodFill(img_copy, mask=mask, seedPoint=(w-1, h-1), newVal=(0, 0, 255),
                      loDiff=(1, 1, 1), upDiff=(1, 1, 1))
                      
        for i in range(20,w-20,20):
            cv2.floodFill(img_copy, mask=mask, seedPoint=(i, 0), newVal=(0, 0, 255),
                      loDiff=(1, 1, 1), upDiff=(1, 1, 1))
            cv2.floodFill(img_copy, mask=mask, seedPoint=(i, h-1), newVal=(0, 0, 255),
                      loDiff=(1, 1, 1), upDiff=(1, 1, 1))
        for i in range(20,h-20,20):
            cv2.floodFill(img_copy, mask=mask, seedPoint=(0, i), newVal=(0, 0, 255),
                      loDiff=(1, 1, 1), upDiff=(1, 1, 1))
            cv2.floodFill(img_copy, mask=mask, seedPoint=(w-1, i), newVal=(0, 0, 255),
                      loDiff=(1, 1, 1), upDiff=(1, 1, 1))
                      
        mask_inv = np.where(mask > 0.5, 0, 1).astype(np.uint8) 
        
        kernel = np.ones((9, 9), np.uint8)
        mask_inv = cv2.morphologyEx(mask_inv, cv2.MORPH_OPEN, kernel)
        
        
        _, contours, hierarchy = cv2.findContours(mask_inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        index=0
        c_point=0
        for i in range(len(contours)):
            c_points = np.squeeze(contours[i])
            if len(c_points)>c_point:
                c_point=len(c_points)
                index=i

        contour = contours[index].reshape(len(contours[index]),2)

        rect = []
        for i in range(len(contour)):
            x1, y1 = contour[i]
            for j in range(len(contour)):
                x2, y2 = contour[j]
                area = abs(y2-y1)*abs(x2-x1)
                rect.append(((x1,y1), (x2,y2), area))

        all_rect = sorted(rect, key = lambda x : x[2], reverse = True)

        best_rect_found = False
        index_rect = 0
        nb_rect = len(all_rect)
    
        while not best_rect_found and index_rect < nb_rect:
            rect = all_rect[index_rect]
            (x1, y1) = rect[0]
            (x2, y2) = rect[1]

            valid_rect = True
        
            x = min(x1, x2)
            while x <max(x1,x2)+1 and valid_rect:
                if mask_inv[y1,x] == 0 or mask_inv[y2,x] == 0:
                    valid_rect = False
                x+=1

            y = min(y1, y2)
            while y <max(y1,y2)+1 and valid_rect:
                if mask_inv[y,x1] == 0 or mask_inv[y,x2] == 0:
                    valid_rect = False
                y+=1

            if valid_rect:
                best_rect_found = True

            index_rect+=1
            
        x_min = min(x1,x2)
        y_min = min(y1,y2)
        x_max = max(x1,x2)
        y_max = max(y1,y2)
        
        return x_min, y_min, x_max, y_max

                              
Demo(model, opts, None, None).test()


