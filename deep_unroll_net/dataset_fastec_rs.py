from torch.utils.data import Dataset
import os
import torch
import numpy as np
from skimage import io
from frame_utils import *

class Dataset_fastec_rs(Dataset):
    def __init__(self, root_dir, seq_len=2, load_optiflow=True, load_1st_GS=False, load_middle_gs=False):
        self.I_rs = []
        self.I_gs = []
        self.I_gs_f = []
        self.optiflow = []
        self.flow_m = []
        self.seq_len = seq_len
        self.load_1st_GS=load_1st_GS
        self.load_optiflow=load_optiflow
        
        for seq_path, _, fnames in sorted(os.walk(root_dir)):
            for fname in fnames:
                if fname != 'meta.log':
                    continue   

                # read in seq of images            
                for i in range(33):
                    if not os.path.exists(os.path.join(seq_path, str(i).zfill(3)+'_rolling.png')):
                        continue

                    seq_Irs=[]
                    seq_Igs=[]
                    seq_Igs_f=[]
                    seq_optiflow=[]
                    seq_flow_m=[]

                    seq_Irs.append(os.path.join(seq_path, str(i).zfill(3)+'_rolling.png'))
                    if load_middle_gs:
                        seq_Igs.append(os.path.join(seq_path, str(i).zfill(3)+'_global_middle.png'))
                    else:
                        seq_Igs.append(os.path.join(seq_path, str(i).zfill(3)+'_global_first.png'))
                    seq_flow_m.append(os.path.join(seq_path, str(i).zfill(3)+'_flow_raft_m.flo'))
                    if load_1st_GS:
                        seq_Igs_f.append(os.path.join(seq_path, str(i).zfill(3)+'_global_first.png'))
                        
                    seq_optiflow.append(os.path.join(seq_path, str(i).zfill(3)+'_flow_raft.flo'))
                            
                    for j in range(1,seq_len):
                        seq_Irs.append(os.path.join(seq_path, str(i+j).zfill(3)+'_rolling.png'))
                        if load_middle_gs:
                            seq_Igs.append(os.path.join(seq_path, str(i+j).zfill(3)+'_global_middle.png'))
                        else:
                            seq_Igs.append(os.path.join(seq_path, str(i+j).zfill(3)+'_global_first.png'))
                        seq_flow_m.append(os.path.join(seq_path, str(i+j).zfill(3)+'_flow_raft_m.flo'))
                        if load_1st_GS:
                            seq_Igs_f.append(os.path.join(seq_path, str(i+j).zfill(3)+'_global_first.png'))
                            
                        seq_optiflow.append(os.path.join(seq_path, str(i+j).zfill(3)+'_flow_raft.flo'))
                    
                    if os.path.exists(seq_optiflow[0]) or not os.path.exists(seq_optiflow[1]):
                        seq_optiflow[1] = seq_optiflow[0]
                    
                    if not os.path.exists(seq_Irs[-1]):
                        break

                    self.I_rs.append(seq_Irs.copy())
                    self.I_gs.append(seq_Igs.copy())
                    self.optiflow.append(seq_optiflow.copy())
                    self.flow_m.append(seq_flow_m.copy())
                    if load_1st_GS:
                        self.I_gs_f.append(seq_Igs_f.copy())
                    
    def __len__(self):
        return len(self.I_gs)

    def __getitem__(self, idx):
        path_rs = self.I_rs[idx]
        path_gs = self.I_gs[idx]
        path_optiflow = self.optiflow[idx]
        path_flow_m = self.flow_m[idx]
        if self.load_1st_GS:
            path_gs_f = self.I_gs_f[idx]

        temp = io.imread(path_rs[0])
        H,W,C=temp.shape
        if C>3:
            C=3

        I_rs=torch.empty([self.seq_len*C, H, W], dtype=torch.float32)
        I_gs=torch.empty([self.seq_len*C, H, W], dtype=torch.float32)
        optiflow=torch.empty([self.seq_len*2, H, W], dtype=torch.float32)
        I_gs_f=torch.empty([self.seq_len*C, H, W], dtype=torch.float32)
        flow_m=torch.empty([self.seq_len*2, H, W], dtype=torch.float32)
        
        for i in range(self.seq_len):
            I_rs[i*C:(i+1)*C,:,:] = torch.from_numpy(io.imread(path_rs[i]).transpose(2,0,1)).float()[:3]/255.
            I_gs[i*C:(i+1)*C,:,:] = torch.from_numpy(io.imread(path_gs[i]).transpose(2,0,1)).float()[:3]/255.
            if self.load_1st_GS:
                I_gs_f[i*C:(i+1)*C,:,:] = torch.from_numpy(io.imread(path_gs_f[i]).transpose(2,0,1)).float()[:3]/255.
            if self.load_optiflow:
                optiflow[i*2:(i+1)*2,:,:] = torch.from_numpy(readFlow(path_optiflow[i]).transpose(2,0,1)).float()
                flow_m[i*2:(i+1)*2,:,:] = torch.from_numpy(readFlow(path_flow_m[i]).transpose(2,0,1)).float()
        '''
        if optiflow is None:
            print(path_optiflow[0])
            print('\n')
            print(path_optiflow[1])
        '''
        #print(optiflow.shape)    
        return {'I_rs': I_rs,
                'I_gs': I_gs,
                'I_gs_f': I_gs_f,
                'flow': optiflow,
                'flow_m': flow_m,
                'path': path_rs}

