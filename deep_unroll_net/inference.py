import os
import torch
import random
import argparse
import numpy as np

from package_core.generic_train_test import *
from forward_warp_package import *
from dataloader import *
from model_RSSR import *
from package_core.metrics import *
from package_core.flow_utils import *
from Convert_m_t import *
from lpips import lpips
import softsplat

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

parser.add_argument('--model_label', type=str, default='pretrained', help='label used to load pre-trained model')

parser.add_argument('--dataset_type', type=str, required=True)
parser.add_argument('--dataset_root_dir', type=str, required=True, help='absolute path for training dataset')
parser.add_argument('--log_dir', type=str, required=True, help='directory used to store trained networks')
parser.add_argument('--results_dir', type=str, required=True, help='directory used to store trained networks')

parser.add_argument('--load_gt_flow', type=bool, default=False)
parser.add_argument('--load_1st_GS', type=int, default=1)#Control whether testing two first scanline GS images

parser.add_argument('--visualize_results', action='store_true')
parser.add_argument('--compute_metrics', action='store_true')

opts=parser.parse_args()
    
##===================================================##
##*************** Create dataloader *****************##
##===================================================##
dataloader = Create_dataloader(opts)

##===================================================##
##****************** Create model *******************##
##===================================================##
model=ModelRSSR(opts)

##===================================================##
##**************** Train the network ****************##
##===================================================##
class Inference(Generic_train_test):
    def augment_data(self, _input):
        im_rs, im_gs, flow, im_gs_f = _input
        
        if im_gs_f is not None:
            im_gs_f = im_gs_f.clone() 
        if flow is not None:
            flow = flow[:,0:2,:,:].clone()    
        
        # extract ground truth I_gs
        im_gs = im_gs[:,:,:,:].clone()
        return [im_rs, im_gs, flow, im_gs_f]

    def decode_input(self, data):
        im_rs=data['I_rs']
        im_gs=data['I_gs']

        im_rs_FC=data['I_rs']
        im_gs_FC=data['I_gs']
        
        flow=None
        mask=None
        im_gs_f=None
        im_gs_f_FC=None
        if self.opts.load_1st_GS==1:
            im_gs_f=data['I_gs_f']
            im_gs_f_FC=data['I_gs_f']
        if self.opts.load_gt_flow:
            flow=data['flow']
        if self.opts.dataset_type=='Carla':
            mask=data['mask']
            mask=mask[:,-1:,:,:].clone()

        if self.opts.dataset_type=='Fastec':
            im_rs0 = F.interpolate(im_rs[:,0:3,:,:],size=[448,640], mode='bilinear')
            im_rs1 = F.interpolate(im_rs[:,3:6,:,:],size=[448,640], mode='bilinear')
            im_rs  = torch.cat((im_rs0,im_rs1),1).clone()
            
            im_gs0 = F.interpolate(im_gs[:,0:3,:,:],size=[448,640], mode='bilinear')
            im_gs1 = F.interpolate(im_gs[:,3:6,:,:],size=[448,640], mode='bilinear')
            im_gs  = torch.cat((im_gs0,im_gs1),1).clone()
            if self.opts.load_1st_GS==1:
                im_gs0_f = F.interpolate(im_gs_f[:,0:3,:,:],size=[448,640], mode='bilinear')
                im_gs1_f = F.interpolate(im_gs_f[:,3:6,:,:],size=[448,640], mode='bilinear')
                im_gs_f  = torch.cat((im_gs0_f,im_gs1_f),1).clone()
            
        _input = [im_rs, im_gs, flow, im_gs_f]
        return self.augment_data(_input), mask, im_gs_FC, im_gs_f_FC

    def test(self):
        sum_psnr=0.
        sum_psnr_mask=0.
        sum_psnr_mask_esti=0.
        sum_ssim=0.
        sum_lpips=0.
        sum_time=0.
        f_metric_all=None
        f_metric_avg=None
        n_frames=0
        dir_results=os.path.join(self.opts.results_dir)
        
        if self.opts.compute_metrics and not os.path.exists(dir_results):
            os.makedirs(dir_results)

        if self.opts.compute_metrics:
            f_metric_all=open(os.path.join(dir_results, 'metric_all'), 'w')
            f_metric_avg=open(os.path.join(dir_results, 'metric_avg'), 'w')

            f_metric_all.write('# frame_id, PSNR_pred, PSNR_pred_mask, SSIM_pred, LPIPS_pred, time (milliseconds)\n')
            f_metric_avg.write('# avg_PSNR_pred, avg_PSNR_pred_mask, avg_SSIM_pred, avg_LPIPS_pred, time (milliseconds)\n')
            
            loss_fn_alex = lpips.LPIPS(net='alex')
            
            self.MSE_LossFn = nn.MSELoss()
            vgg16 = torchvision.models.vgg16(pretrained=True)
            self.vgg16_conv_4_3 = nn.Sequential(*list(vgg16.children())[0][:22])
            self.vgg16_conv_4_3.to('cuda')
            for param in self.vgg16_conv_4_3.parameters():
		            param.requires_grad = False
            
        for i, data in enumerate(self.dataloader):
            _input, mask, im_gs_FC, im_gs_f_FC=self.decode_input(data)
            im_rs, im_gs, flow, im_gs_f = _input
            self.model.set_input(_input)
            
            #compute time
            torch.cuda.synchronize()
            time_start=time.time()
            
            with torch.no_grad():
                pred_im, pred_mask, pred_flow, pred_scale = self.model.forward()
            
            torch.cuda.synchronize()
            time_end=time.time()
            
            #convert to 0th row
            if self.opts.load_1st_GS==1:
                im_rs_clone = im_rs.clone().cuda()
                pred_gs_f, pred_gs_f_mask, pred_gs_f_flow = self.convert_m2t(im_rs_clone, pred_flow, pred_scale, 0)
            
            # compute metrics 
            if self.opts.compute_metrics:
                
                if self.opts.load_1st_GS==1:
                    predict_GS=pred_gs_f[1]
                    GT_GS=self.model.im_gs_f[:,-self.opts.n_chan:,:,:]
                else:
                    predict_GS=pred_im[1]
                    GT_GS=self.model.im_gs[:,-self.opts.n_chan:,:,:]
                '''
                #if self.opts.dataset_type=='Fastec':
                    #predict_GS = F.interpolate(predict_GS, size=[480,640], mode='bilinear')
                    #if self.opts.load_1st_GS==1:
                        #GT_GS = im_gs_f_FC[:,-self.opts.n_chan:,:,:].clone().cuda()
                    #else:
                        #GT_GS = im_gs_FC[:,-self.opts.n_chan:,:,:].clone().cuda()
                '''
                psnr_pred=PSNR(predict_GS, GT_GS)
                psnr_pred_mask=PSNR(predict_GS, GT_GS, mask)
                ssim_pred=SSIM(predict_GS, GT_GS)

                lpips_pred=0.
                #lpips_pred=loss_fn_alex(predict_GS, GT_GS)                    #### Notes: Using this command to compute LPIPS
                
                diff_time = time_end - time_start

                sum_psnr += psnr_pred
                sum_psnr_mask += psnr_pred_mask
                sum_ssim += ssim_pred
                sum_lpips += lpips_pred
                sum_time += diff_time
                n_frames += 1

                print('PSNR(%.2f dB) PSNR_mask(%.2f dB) SSIM(%.2f) LPIPS(%.4f) time(%.2f ms)\n' % (psnr_pred, psnr_pred_mask, ssim_pred, lpips_pred, diff_time*1000))
                f_metric_all.write('%d %.2f %.2f %.2f %.4f %.2f\n' % (i, psnr_pred, psnr_pred_mask, ssim_pred, lpips_pred, diff_time*1000))

        if self.opts.compute_metrics:
            psnr_avg = sum_psnr / n_frames
            psnr_avg_mask = sum_psnr_mask / n_frames
            ssim_avg = sum_ssim / n_frames
            lpips_avg = sum_lpips / n_frames
            time_avg = sum_time / n_frames

            print('PSNR_avg (%.2f dB) PSNR_avg_mask (%.2f dB) SSIM_avg (%.3f) LPIPS_avg (%.4f) time_avg(%.2f ms)' % (psnr_avg, psnr_avg_mask, ssim_avg, lpips_avg, time_avg*1000))
            f_metric_avg.write('%.2f %.2f %.2f %.4f %.2f\n' % (psnr_avg, psnr_avg_mask, ssim_avg, lpips_avg, time_avg*1000))

            f_metric_all.close()
            f_metric_avg.close()


    def convert_m2t(self, im_rs, pred_flow, pred_scale, t):
        B,C,H,W = im_rs.size()
        
        warpers = ForwardWarp.create_with_implicit_mesh(B, 3, H, W, 2, 0.5)
        
        tt = H * t
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

Inference(model, opts, dataloader, None).test()
