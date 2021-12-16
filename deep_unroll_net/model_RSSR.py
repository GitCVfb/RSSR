import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict

from package_core.model_base import *
from package_core.losses import *
from package_core.flow_utils import *
from package_core.image_proc import *
from net_scale import *

from correlation_package import Correlation

from package_core.net_basics import *
from forward_warp_package import *
from reblur_package import *
from net_pwc import *
from image_proc import *
from Convert_m_t import *

import softsplat

backwarp_tenGrid = {}
def backwarp(tenInput, tenFlow):
    if str(tenFlow.shape) not in backwarp_tenGrid:
        tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]), tenFlow.shape[3]).view(1, 1, 1, -1).expand(-1, -1, tenFlow.shape[2], -1)
        tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), tenFlow.shape[2]).view(1, 1, -1, 1).expand(-1, -1, -1, tenFlow.shape[3])

        backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([ tenHor, tenVer ], 1).cuda()
        # end

    tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)

    return torch.nn.functional.grid_sample(input=tenInput, grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=False)
# end

class ModelRSSR(ModelBase):
    def __init__(self, opts):
        super(ModelRSSR, self).__init__()
        
        self.opts = opts
        
        # create networks
        self.model_names=['flow', 'scale']
        self.net_flow = PWCDCNet().cuda()
        self.net_scale = UNet_Esti_Scale().cuda()
        
        # print network
        self.print_networks(self.net_flow)
        self.print_networks(self.net_scale)
        
        # load in initialized network parameters
        if not opts.is_training or opts.continue_train:
            self.load_checkpoint(opts.model_label)
        else:
            self.load_checkpoint_only_load_flow(opts.model_label_pwc)#load pretrained PWCNet model
        
        self.upsampleX4 = nn.Upsample(scale_factor=4, mode='bilinear')
        
        
        if self.opts.is_training:
            # initialize optimizers
            
            self.optimizer_G = torch.optim.Adam([
                {'params': self.net_flow.parameters()},
                {'params': self.net_scale.parameters()}], lr=opts.lr)
            
            self.optimizer_names = ['G']
            self.build_lr_scheduler()
            
            # create losses
            self.loss_fn_perceptual = PerceptualLoss(loss=nn.L1Loss())
            self.loss_fn_L1 = L1Loss()
            self.loss_fn_tv_2C = VariationLoss(nc=2)
            
            ###Initializing VGG16 model for perceptual loss
            self.MSE_LossFn = nn.MSELoss()
            vgg16 = torchvision.models.vgg16(pretrained=True)
            self.vgg16_conv_4_3 = nn.Sequential(*list(vgg16.children())[0][:22])
            self.vgg16_conv_4_3.to('cuda')
            for param in self.vgg16_conv_4_3.parameters():
		            param.requires_grad = False

    def set_input(self, _input):
        im_rs, im_gs, gt_flow, im_gs_f = _input
        self.im_rs = im_rs.cuda()
        self.im_gs = im_gs
        self.gt_flow = gt_flow
        self.im_gs_f = im_gs_f

        if self.im_gs is not None:
            self.im_gs = self.im_gs.cuda()
        if self.im_gs_f is not None:
            self.im_gs_f = self.im_gs_f.cuda()
        if self.gt_flow is not None:
            self.gt_flow = self.gt_flow.cuda()

    def forward(self):

        B,_,H,W=self.im_rs.size()
        self.mask_fn = FlowWarpMask.create_with_implicit_mesh(B, 3, H, W)
        self.warpers = ForwardWarp.create_with_implicit_mesh(B, 3, H, W, 2, 0.5)
        
        #impose offset constraint explicitly
        grid_rows = self.generate_2D_grid(H, W)[1]
        t_flow_offset = grid_rows.unsqueeze(0).unsqueeze(0)
        self.I0_t_flow_ref_to_m = -(t_flow_offset-H//2+0.001)/(H//2)
        self.I1_t_flow_ref_to_m = -self.I0_t_flow_ref_to_m
        
        im_rs0 = self.im_rs[:,0:3,:,:].clone()
        im_rs1 = self.im_rs[:,3:6,:,:].clone()
        B,C,H,W = im_rs0.size()

        Fs_0_1=self.net_flow(im_rs0, im_rs1)
        Fs_1_0=self.net_flow(im_rs1, im_rs0)
        
        F_0_1 = self.upsampleX4(Fs_0_1[0])*20.0
        F_1_0 = self.upsampleX4(Fs_1_0[0])*20.0
        
        mask_rs_1 = self.mask_fn(F_1_0)
        mask_rs_0 = self.mask_fn(F_0_1)

        g_I1_F_1_0, _ = warp_image_flow(im_rs0, F_1_0)
        g_I0_F_0_1, _ = warp_image_flow(im_rs1, F_0_1)
        
        interp_out_var = self.net_scale(im_rs0, F_0_1, im_rs1, F_1_0)
        interp_out_var_0 = interp_out_var[:,0,:,:]
        interp_out_var_1 = interp_out_var[:,1,:,:]
        
        F_0_1 = F_0_1 + interp_out_var[:,2:4,:,:]
        F_1_0 = F_1_0 + interp_out_var[:,4:6,:,:]
        
        #impose offset constraint explicitly (Proposition 1)
        T_0_m = self.I0_t_flow_ref_to_m * interp_out_var_0.unsqueeze(1)
        T_1_m = self.I1_t_flow_ref_to_m * interp_out_var_1.unsqueeze(1)
        
        F_0_m_final = F_0_1 * T_0_m
        F_1_m_final = F_1_0 * T_1_m
        
        tenMetric_0_1 = torch.nn.functional.l1_loss(input=im_rs0, target=backwarp(tenInput=im_rs1, tenFlow=F_0_1), reduction='none').mean(1, True)
        tenMetric_1_0 = torch.nn.functional.l1_loss(input=im_rs1, target=backwarp(tenInput=im_rs0, tenFlow=F_1_0), reduction='none').mean(1, True)
        g_I0m = softsplat.FunctionSoftsplat(tenInput=im_rs0, tenFlow=F_0_m_final, tenMetric=-20.0 * tenMetric_0_1, strType='softmax')
        g_I1m = softsplat.FunctionSoftsplat(tenInput=im_rs1, tenFlow=F_1_m_final, tenMetric=-20.0 * tenMetric_1_0, strType='softmax')
        
        _, mask_I0m = self.warpers(im_rs0, F_0_m_final)
        _, mask_I1m = self.warpers(im_rs1, F_1_m_final)
        
        out_mask=[mask_I0m, mask_I1m, mask_rs_1, mask_rs_0]
        out_image=[g_I0m, g_I1m, g_I1_F_1_0, g_I0_F_0_1]
        out_flow=[F_0_m_final, F_1_m_final, F_0_1, F_1_0]
        out_scale=[T_0_m, T_1_m]
        
        return out_image, out_mask, out_flow, out_scale

    def optimize_parameters(self):
        self.pred_im, self.pred_mask, self.pred_flow, self.pred_scale = self.forward()
        
        if self.opts.load_1st_GS:
            self.pred_gs_f, self.pred_gs_f_mask, self.pred_gs_f_flow, _ = self.convert_m2t(self.im_rs, self.pred_flow, self.pred_scale, 0)#convert to 0th row
        
        img_gs_0 = self.im_gs[:,0:3,:,:].clone()
        img_gs_1 = self.im_gs[:,3:6,:,:].clone()
        img_rs_0 = self.im_rs[:,0:3,:,:].clone()
        img_rs_1 = self.im_rs[:,3:6,:,:].clone()
        
        #===========================================================#
        #                   Initialize losses                       #
        #===========================================================#
        self.loss_L1 = torch.tensor([0.], requires_grad=True).cuda().float()
        self.loss_perceptual = torch.tensor([0.], requires_grad=True).cuda().float()
        self.loss_flow_smoothness = torch.tensor([0.], requires_grad=True).cuda().float()
        self.loss_L1_rs = torch.tensor([0.], requires_grad=True).cuda().float()
        if self.opts.load_1st_GS:
            self.loss_L1_f = torch.tensor([0.], requires_grad=True).cuda().float()
            self.loss_perceptual_f = torch.tensor([0.], requires_grad=True).cuda().float()
        
        #===========================================================#
        #                       Compute losses                      #
        #===========================================================#                            
        self.loss_L1 += self.opts.lamda_L1 *\
                            self.loss_fn_L1(self.pred_im[0], img_gs_0.detach(), mean=True)
        self.loss_L1 += self.opts.lamda_L1 *\
                            self.loss_fn_L1(self.pred_im[1], img_gs_1.detach(), mean=True)
        
        self.loss_L1_rs += self.opts.lamda_L1_rs *\
                            self.loss_fn_L1(self.pred_im[2], img_rs_1, self.pred_mask[2].detach(), mean=True)
        self.loss_L1_rs += self.opts.lamda_L1_rs *\
                            self.loss_fn_L1(self.pred_im[3], img_rs_0, self.pred_mask[3].detach(), mean=True)
        
        self.loss_perceptual += self.opts.lamda_perceptual *\
                                    self.loss_fn_perceptual.get_loss(self.pred_im[0], img_gs_0.detach())
        self.loss_perceptual += self.opts.lamda_perceptual *\
                                    self.loss_fn_perceptual.get_loss(self.pred_im[1], img_gs_1.detach())

        if self.pred_flow is not None and self.opts.lamda_flow_smoothness>1e-6:
            for lv in range(4):
                self.loss_flow_smoothness += self.opts.lamda_flow_smoothness *\
                                             self.loss_fn_tv_2C(self.pred_flow[lv], mean=True)
        ###
        if self.opts.load_1st_GS:
            self.loss_L1_f += self.opts.lamda_L1/2.0 *\
                            self.loss_fn_L1(self.pred_gs_f[0], self.im_gs_f[:,0:3,:,:], self.pred_gs_f_mask[0].detach(), mean=True)
            self.loss_L1_f += self.opts.lamda_L1/2.0 *\
                            self.loss_fn_L1(self.pred_gs_f[1], self.im_gs_f[:,3:6,:,:], self.pred_gs_f_mask[1].detach(), mean=True)

            self.loss_perceptual_f += self.opts.lamda_perceptual *\
                                    self.loss_fn_perceptual.get_loss(self.pred_gs_f[0], self.im_gs_f[:,0:3,:,:]*self.pred_gs_f_mask[0].float().detach())
            self.loss_perceptual_f += self.opts.lamda_perceptual *\
                                    self.loss_fn_perceptual.get_loss(self.pred_gs_f[1], self.im_gs_f[:,3:6,:,:]*self.pred_gs_f_mask[1].float().detach())
                                    
            for lv in range(2):
                self.loss_flow_smoothness += self.opts.lamda_flow_smoothness *\
                                             self.loss_fn_tv_2C(self.pred_gs_f_flow[lv], mean=True)
        ###
        
        # sum them up
        if self.opts.load_1st_GS:
            self.loss_G = self.loss_L1 +\
                        self.loss_L1_rs +\
                        self.loss_perceptual +\
                        self.loss_flow_smoothness +\
                        self.loss_L1_f +\
                        self.loss_perceptual_f
        else:
            self.loss_G = self.loss_L1 +\
                        self.loss_L1_rs +\
                        self.loss_perceptual +\
                        self.loss_flow_smoothness                

        # Optimize 
        self.optimizer_G.zero_grad()
        self.loss_G.backward()
        self.optimizer_G.step() 

    # save networks to file 
    def save_checkpoint(self, label):
        self.save_network(self.net_flow, 'flow', label, self.opts.log_dir)
        self.save_network(self.net_scale, 'scale', label, self.opts.log_dir)
        
    def load_checkpoint_only_load_flow(self, label):
        self.load_network(self.net_flow, 'flow', label, self.opts.log_dir_pwc)
        
    def load_checkpoint(self, label):
        self.load_network(self.net_flow, 'flow', label, self.opts.log_dir)
        self.load_network(self.net_scale, 'scale', label, self.opts.log_dir)
            
    def get_current_scalars(self):
        losses = {}
        if self.opts.load_1st_GS:
            losses['loss_G'] = self.loss_G.item()
            losses['loss_L1_rs'] = self.loss_L1_rs.item()
            losses['loss_L1_m'] = self.loss_L1.item()
            losses['loss_perceptual_m'] = self.loss_perceptual.item()
            losses['loss_flow_smoothness'] = self.loss_flow_smoothness.item()
            losses['loss_L1_f'] = self.loss_L1_f.item()
            losses['loss_perceptual_f'] = self.loss_perceptual_f.item()
        else:
            losses['loss_G'] = self.loss_G.item()
            losses['loss_L1_rs'] = self.loss_L1_rs.item()
            losses['loss_L1_m'] = self.loss_L1.item()
            losses['loss_perceptual_m'] = self.loss_perceptual.item()
            losses['loss_flow_smoothness'] = self.loss_flow_smoothness.item()
        return losses

    def get_current_visuals(self):
        output_visuals = {}

        output_visuals['im_rs'] = self.im_rs[:,-3:,:,:].clone()
        '''
        for lv in range(self.nlvs):
            if self.pred_flow[lv] is not None:
                output_visuals['flow_pred_'+str(lv)] = torch.from_numpy(flow_to_numpy_rgb(self.pred_flow[lv]).transpose(0,3,1,2)).float()/255.
                output_visuals['mask_'+str(lv)] = self.pred_mask[lv].clone().repeat(1,3,1,1)
            
            #if self.pred_im[lv] is None:
                #continue
            #output_visuals['im_gs_'+str(lv)] = self.im_gs_clone[lv]
            #output_visuals['im_gs_pred_'+str(lv)] = self.pred_im[lv]
            #output_visuals['res_im_gs_'+str(lv)] = torch.abs(self.pred_im[lv] - self.im_gs_clone[lv])*5.
        '''    
        return output_visuals


    def generate_2D_grid(self, H, W):
        x = torch.arange(0, W, 1).float().cuda() 
        y = torch.arange(0, H, 1).float().cuda()

        xx = x.repeat(H, 1)
        yy = y.view(H, 1).repeat(1, W)
    
        grid = torch.stack([xx, yy], dim=0) 

        return grid
        
    def convert_m2t(self, im_rs, pred_flow, pred_scale, t):
        B,_,H,W=im_rs.size()
        T_0_t = scale_from_m2t(pred_scale[0], H, W, t)
        T_1_t = scale_from_m2t(pred_scale[1], H, W, t)
        
        F_0_t = pred_flow[2] * T_0_t
        F_1_t = pred_flow[3] * T_1_t
        
        g_I0t, mask_I0t = self.warpers(im_rs[:,0:3,:,:], F_0_t)
        g_I1t, mask_I1t = self.warpers(im_rs[:,3:6,:,:], F_1_t)
        
        return [g_I0t, g_I1t], [mask_I0t, mask_I1t], [F_0_t, F_1_t], [T_0_t, T_1_t]