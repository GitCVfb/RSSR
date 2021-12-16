import os
import torch
import random
import argparse
import numpy as np
from skimage import io
import shutil

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
                    im_rs0 = F.interpolate(im_rs[:,0:3,:,:],size=[448,640], mode='bilinear')
                    im_rs1 = F.interpolate(im_rs[:,3:6,:,:],size=[448,640], mode='bilinear')
                    im_rs  = torch.cat((im_rs0,im_rs1),1).clone()#
                    
                _input = [im_rs, None, None, None]
                B,C,H,W = im_rs.size()
                
                self.model.set_input(_input)
                pred_im, pred_mask, pred_flow, pred_scale = self.model.forward()
                
                #convert to 0th row
                esti_first_row = True
                if esti_first_row == True:
                    im_rs_clone = im_rs.clone().cuda()
                    pred_gs_f, pred_gs_f_mask, pred_gs_f_flow = self.convert_m2t(im_rs_clone, pred_flow, pred_scale, 0)
                
                if self.opts.is_Fastec==1:
                    for i in range(4):
                        pred_im[i] = F.interpolate(pred_im[i],size=[480,640], mode='bilinear')
                        pred_flow[i] = F.interpolate(pred_flow[i],size=[480,640], mode='bilinear')
                    if esti_first_row == True:
                        for i in range(2):
                            pred_gs_f[i] = F.interpolate(pred_gs_f[i],size=[480,640], mode='bilinear')
                            pred_gs_f_flow[i] = F.interpolate(pred_gs_f_flow[i],size=[480,640], mode='bilinear')
                
                # save results
                im_gs_0_f = io.imread(os.path.join(os.path.join(self.opts.data_dir, seq), 'gs_0_f.png'))
                im_gs_0_m = io.imread(os.path.join(os.path.join(self.opts.data_dir, seq), 'gs_0_m.png'))
                im_gs_1_f = io.imread(os.path.join(os.path.join(self.opts.data_dir, seq), 'gs_1_f.png'))
                im_gs_1_m = io.imread(os.path.join(os.path.join(self.opts.data_dir, seq), 'gs_1_m.png'))
                im_rs_0 = io.imread(im_rs0_path)
                im_rs_1 = io.imread(im_rs1_path)

                im_gs_0_f = im_gs_0_f[:].copy()
                im_gs_0_m = im_gs_0_m[:].copy()
                im_gs_1_f = im_gs_1_f[:].copy()
                im_gs_1_m = im_gs_1_m[:].copy()
                im_rs_0 = im_rs_0[:].copy()
                im_rs_1 = im_rs_1[:].copy()

                io.imsave(os.path.join(self.opts.results_dir, seq+'_rs_0.png'), im_rs_0)
                io.imsave(os.path.join(self.opts.results_dir, seq+'_rs_1.png'), im_rs_1)
                io.imsave(os.path.join(self.opts.results_dir, seq+'_rs_0_warped.png'), (pred_im[3].clamp(0,1).cpu().numpy().transpose(0,2,3,1)[0]*255).astype(np.uint8))
                io.imsave(os.path.join(self.opts.results_dir, seq+'_rs_1_warped.png'), (pred_im[2].clamp(0,1).cpu().numpy().transpose(0,2,3,1)[0]*255).astype(np.uint8))
                io.imsave(os.path.join(self.opts.results_dir, seq+'_pred_0_m.png'), (pred_im[0].clamp(0,1).cpu().numpy().transpose(0,2,3,1)[0]*255).astype(np.uint8))
                io.imsave(os.path.join(self.opts.results_dir, seq+'_pred_1_m.png'), (pred_im[1].clamp(0,1).cpu().numpy().transpose(0,2,3,1)[0]*255).astype(np.uint8))
                
                im_gs_0_m_new=torch.from_numpy(im_gs_0_m[:,:,:3].transpose(2,0,1)).unsqueeze(0).cuda().float()/255.
                io.imsave(os.path.join(self.opts.results_dir, seq+'_gs_0_m.png'), (im_gs_0_m_new.clamp(0,1).cpu().numpy().transpose(0,2,3,1)[0]*255).astype(np.uint8))
                im_gs_1_m_new=torch.from_numpy(im_gs_1_m[:,:,:3].transpose(2,0,1)).unsqueeze(0).cuda().float()/255.
                io.imsave(os.path.join(self.opts.results_dir, seq+'_gs_1_m.png'), (im_gs_1_m_new.clamp(0,1).cpu().numpy().transpose(0,2,3,1)[0]*255).astype(np.uint8))

                if esti_first_row == True:
                    io.imsave(os.path.join(self.opts.results_dir, seq+'_pred_0_f.png'), (pred_gs_f[0].clamp(0,1).cpu().numpy().transpose(0,2,3,1)[0]*255).astype(np.uint8))
                    io.imsave(os.path.join(self.opts.results_dir, seq+'_pred_1_f.png'), (pred_gs_f[1].clamp(0,1).cpu().numpy().transpose(0,2,3,1)[0]*255).astype(np.uint8))

                    im_gs_0_f_new=torch.from_numpy(im_gs_0_f[:,:,:3].transpose(2,0,1)).unsqueeze(0).cuda().float()/255.
                    io.imsave(os.path.join(self.opts.results_dir, seq+'_gs_0_f.png'), (im_gs_0_f_new.clamp(0,1).cpu().numpy().transpose(0,2,3,1)[0]*255).astype(np.uint8))
                    im_gs_1_f_new=torch.from_numpy(im_gs_1_f[:,:,:3].transpose(2,0,1)).unsqueeze(0).cuda().float()/255.
                    io.imsave(os.path.join(self.opts.results_dir, seq+'_gs_1_f.png'), (im_gs_1_f_new.clamp(0,1).cpu().numpy().transpose(0,2,3,1)[0]*255).astype(np.uint8))

                print('saved', self.opts.results_dir, seq+'_pred_m.png')
                

                save_flow = False
                if save_flow == True:
                    flow = pred_flow[2]
                    flow = flow.cpu().numpy().transpose(0,2,3,1)[0]
                    flow_image = flow_viz.flow_to_image(flow)
                    cv2.imwrite(os.path.join(self.opts.results_dir, seq+'_pred_flow_0_1.png'), flow_image)
                    
                    flow = pred_flow[3]
                    flow = flow.cpu().numpy().transpose(0,2,3,1)[0]
                    flow_image = flow_viz.flow_to_image(flow)
                    cv2.imwrite(os.path.join(self.opts.results_dir, seq+'_pred_flow_1_0.png'), flow_image)
                    
                    flow = pred_flow[0]
                    flow = flow.cpu().numpy().transpose(0,2,3,1)[0]
                    flow_image = flow_viz.flow_to_image(flow)
                    cv2.imwrite(os.path.join(self.opts.results_dir, seq+'_pred_flow_0_m.png'), flow_image)
                    
                    flow = pred_flow[1]
                    flow = flow.cpu().numpy().transpose(0,2,3,1)[0]
                    flow_image = flow_viz.flow_to_image(flow)
                    cv2.imwrite(os.path.join(self.opts.results_dir, seq+'_pred_flow_1_m.png'), flow_image)
                    
                    if esti_first_row == True:
                        flow = pred_gs_f_flow[0]
                        flow = flow.cpu().numpy().transpose(0,2,3,1)[0]
                        flow_image = flow_viz.flow_to_image(flow)
                        cv2.imwrite(os.path.join(self.opts.results_dir, seq+'_pred_flow_0_f.png'), flow_image)
                        
                        flow = pred_gs_f_flow[1]
                        flow = flow.cpu().numpy().transpose(0,2,3,1)[0]
                        flow_image = flow_viz.flow_to_image(flow)
                        cv2.imwrite(os.path.join(self.opts.results_dir, seq+'_pred_flow_1_f.png'), flow_image)
                
                
    def convert_m2t(self, im_rs, pred_flow, pred_scale, t):
        B,C,H,W = im_rs.size()

        warpers = ForwardWarp.create_with_implicit_mesh(B, 3, H, W, 2, 0.5)
        
        tt = H*t
        T_0_t = scale_from_m2t(pred_scale[0], H, W, tt)
        T_1_t = scale_from_m2t(pred_scale[1], H, W, tt)
        
        F_0_t = pred_flow[2] * T_0_t
        F_1_t = pred_flow[3] * T_1_t
        
        tenMetric_0_1 = torch.nn.functional.l1_loss(input=im_rs[:,0:3,:,:], target=backwarp(tenInput=im_rs[:,3:6,:,:], tenFlow=pred_flow[2]), reduction='none').mean(1, True)
        tenMetric_1_0 = torch.nn.functional.l1_loss(input=im_rs[:,3:6,:,:], target=backwarp(tenInput=im_rs[:,0:3,:,:], tenFlow=pred_flow[3]), reduction='none').mean(1, True)
        g_I0t = softsplat.FunctionSoftsplat(tenInput=im_rs[:,0:3,:,:], tenFlow=F_0_t, tenMetric=-20.0 * tenMetric_0_1, strType='softmax')
        g_I1t = softsplat.FunctionSoftsplat(tenInput=im_rs[:,3:6,:,:], tenFlow=F_1_t, tenMetric=-20.0 * tenMetric_1_0, strType='softmax')
        
        _, mask_I0t = warpers(im_rs[:,0:3,:,:], F_0_t)
        _, mask_I1t = warpers(im_rs[:,3:6,:,:], F_1_t)
        
        return [g_I0t, g_I1t], [mask_I0t, mask_I1t], [F_0_t, F_1_t]
        
        
Demo(model, opts, None, None).test()


