import torch


def generate_2D_grid(H, W):
    x = torch.arange(0, W, 1).float().cuda() 
    y = torch.arange(0, H, 1).float().cuda()

    xx = x.repeat(H, 1)
    yy = y.view(H, 1).repeat(1, W)
    
    grid = torch.stack([xx, yy], dim=0) 

    return grid
    

#Convert the scale map (T_0_m or T_1_m) of the middle row to row t (t=0,...,H)
#Note: Constant velocity assumption
def scale_from_m2t(T_i_m, H, W, t):
    grid_rows = generate_2D_grid(H, W)[1]
    t_flow_offset = grid_rows.unsqueeze(0).unsqueeze(0)
        
    s1 = -(t_flow_offset-H//2+0.001)
    s2 = t_flow_offset-H//2
    
    #T_i_t = T_i_m/s1*(t-H//2-s2)
    T_i_t = T_i_m/s1*(t-t_flow_offset+0.001)
    
    return T_i_t

