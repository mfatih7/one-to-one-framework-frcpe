import torch
import torch.nn as nn


class GridPosition(nn.Module):
    def __init__(self, grid_num, use_gpu, use_tpu):
        nn.Module.__init__(self)
        self.grid_num = grid_num
        self.use_gpu = use_gpu
        self.use_tpu = use_tpu
        
        if(self.use_tpu==True):
            import torch_xla.core.xla_model as xm
            self.device = xm.xla_device()  

    def forward(self, batch_size):
        
        if(self.use_gpu == True):
            grid_center_x = torch.linspace(-1.+2./self.grid_num/2,1.-2./self.grid_num/2,steps=self.grid_num).cuda()
            grid_center_y = torch.linspace(1.-2./self.grid_num/2,-1.+2./self.grid_num/2,steps=self.grid_num).cuda()
            
        if(self.use_tpu == True):
            grid_center_x = torch.linspace(-1.+1./self.grid_num/2,1.-1./self.grid_num/2,steps=self.grid_num).to(self.device)
            grid_center_y = torch.linspace(1.-1./self.grid_num/2,-1.+1./self.grid_num/2,steps=self.grid_num).to(self.device)
            
        if(self.use_gpu == False and self.use_tpu == False ):
            grid_center_x = torch.linspace(-1.+1./self.grid_num/2,1.-1./self.grid_num/2,steps=self.grid_num)
            grid_center_y = torch.linspace(1.-1./self.grid_num/2,-1.+1./self.grid_num/2,steps=self.grid_num)            
        
        # grid_center_x = torch.linspace(-1.+2./self.grid_num/2,1.-2./self.grid_num/2,steps=self.grid_num).cuda() if self.use_gpu else torch.linspace(-1.+1./self.grid_num/2,1.-1./self.grid_num/2,steps=self.grid_num)
        # grid_center_y = torch.linspace(1.-2./self.grid_num/2,-1.+2./self.grid_num/2,steps=self.grid_num).cuda() if self.use_gpu else torch.linspace(1.-1./self.grid_num/2,-1.+1./self.grid_num/2,steps=self.grid_num)
        
        # BCHW, (b,:,h,w)->(x,y)
        grid_center_position_mat = torch.reshape(
            torch.cartesian_prod(grid_center_x, grid_center_y),
            (1, self.grid_num, self.grid_num, 2)
        ).permute(0,3,2,1)
        # BCN, (b,:,n)->(x,y), left to right then up to down
        grid_center_position_seq = grid_center_position_mat.reshape(1, 2, self.grid_num*self.grid_num)
        return grid_center_position_seq.repeat(batch_size, 1, 1)


class AttentionPropagation(nn.Module):
    def __init__(self, channels, head):
        nn.Module.__init__(self)
        self.head = head
        self.head_dim = channels // head
        self.query_filter, self.key_filter, self.value_filter = nn.Conv1d(channels, channels, kernel_size=1),\
                                                              nn.Conv1d(channels, channels, kernel_size=1),\
                                                              nn.Conv1d(channels, channels, kernel_size=1)
        self.mh_filter = nn.Conv1d(channels, channels, kernel_size=1)
        self.cat_filter = nn.Sequential(
            nn.Conv1d(2*channels, 2*channels, kernel_size=1),
            nn.BatchNorm1d(2*channels), nn.ReLU(),
            nn.Conv1d(2*channels, channels, kernel_size=1),
        )

    def forward(self, motion1, motion2):
        # motion1(q) attend to motion(k,v)
        batch_size = motion1.shape[0]
        query, key, value = self.query_filter(motion1).view(batch_size, self.head, self.head_dim, -1),\
                            self.key_filter(motion2).view(batch_size, self.head, self.head_dim, -1),\
                            self.value_filter(motion2).view(batch_size, self.head, self.head_dim, -1)
        score = torch.softmax(torch.einsum('bhdn,bhdm->bhnm', query, key) / self.head_dim ** 0.5, dim = -1)
        add_value = torch.einsum('bhnm,bhdm->bhdn', score, value).reshape(batch_size, self.head_dim * self.head, -1)
        add_value = self.mh_filter(add_value)
        motion1_new = motion1 + self.cat_filter(torch.cat([motion1, add_value], dim=1))
        return motion1_new


class ResBlock(nn.Module):
    def __init__(self, channels):
        nn.Module.__init__(self)
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.net(x) + x
        return x


class Filter(nn.Module):
    def __init__(self, channels):
        nn.Module.__init__(self)
        self.resnet = nn.Sequential(*[ResBlock(channels) for _ in range(3)])
        self.scale =  nn.Sequential(
            nn.Conv2d(channels, channels, 1, padding=0),
        )

    def forward(self, x):
        x = self.scale(self.resnet(x))
        return x


class FilterNet(nn.Module):
    def __init__(self, grid_num, channels):
        nn.Module.__init__(self)
        self.grid_num = grid_num
        self.filter = Filter(channels)

    def forward(self, x):
        # BCN -> BCHW
        x = x.reshape(x.shape[0], x.shape[1], self.grid_num, self.grid_num)
        x = self.filter(x)
        # BCHW -> BCN
        x = x.reshape(x.shape[0], x.shape[1], self.grid_num*self.grid_num)
        return x


class PositionEncoder(nn.Module):
    def __init__(self, channels):
        nn.Module.__init__(self)
        self.position_encoder = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, channels, kernel_size=1)
        )
        
    def forward(self, x):
        return self.position_encoder(x)


class InitProject(nn.Module):
    def __init__(self, channels):
        nn.Module.__init__(self)
        self.init_project = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=1), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, channels, kernel_size=1)
        )
        
    def forward(self, x):
        return self.init_project(x)


class InlinerPredictor(nn.Module):
    def __init__(self, channels):
        nn.Module.__init__(self)
        self.inlier_pre = nn.Sequential(
            nn.Conv1d(channels, 64, kernel_size=1), nn.InstanceNorm1d(64, eps=1e-3), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 16, kernel_size=1), nn.InstanceNorm1d(16, eps=1e-3), nn.BatchNorm1d(16), nn.ReLU(),
            nn.Conv1d(16, 4, kernel_size=1), nn.InstanceNorm1d(4, eps=1e-3), nn.BatchNorm1d(4), nn.ReLU(),
            nn.Conv1d(4, 1, kernel_size=1)
        )

    def forward(self, d):
        # BCN -> B1N
        return self.inlier_pre(d)


class LayerBlock(nn.Module):
    def __init__(self, channels, head, grid_num):
        nn.Module.__init__(self)
        self.align = AttentionPropagation(channels, head)
        self.filter = FilterNet(grid_num, channels)
        self.dealign = AttentionPropagation(channels, head)
        self.inlier_pre = InlinerPredictor(channels)

    def forward(self, xs, d, grid_pos_embed):
        # xs: B1N4
        grid_d = self.align(grid_pos_embed, d)
        grid_d = self.filter(grid_d)
        d_new = self.dealign(d, grid_d)
        # BCN -> B1N -> BN
        logits = torch.squeeze(self.inlier_pre(d_new - d), 1)
        
        return d_new, logits
        
        # e_hat = weighted_8points(xs, logits)
        # return d_new, logits, e_hat


class ConvMatch(nn.Module):
    def __init__(self, use_gpu, use_tpu ):
        nn.Module.__init__(self)
        
        self.layer_num = 6        
        self.grid_num = 16        
        self.net_channels =128        
        self.head = 4        
        
        self.layer_num = self.layer_num

        self.grid_center = GridPosition(self.grid_num, use_gpu=use_gpu, use_tpu=use_tpu )
        self.pos_embed = PositionEncoder(self.net_channels)
        self.grid_pos_embed = PositionEncoder(self.net_channels)
        self.init_project = InitProject(self.net_channels)
        self.layer_blocks = nn.Sequential(
            *[LayerBlock(self.net_channels, self.head, self.grid_num) for _ in range(self.layer_num)]
        )
        
    def forward(self, data):
        
        batch_size, num_pts = data.shape[0], data.shape[2]
        
        input = data.transpose(1,3).squeeze(3)
        x1, x2 = input[:,:2,:], input[:,2:4,:]
        motion = x2 - x1
        
        pos = x1 # B2N
        grid_pos = self.grid_center(batch_size) # B2N

        pos_embed = self.pos_embed(pos) # BCN
        grid_pos_embed = self.grid_pos_embed(grid_pos)

        d = self.init_project(motion) + pos_embed # BCN

        res_logits= []        
        for i in range(self.layer_num):
            d, logits= self.layer_blocks[i](data, d, grid_pos_embed) # BCN
            res_logits.append(logits)
        return res_logits
        
        
    # def forward(self, data):
    #     assert data['xs'].dim() == 4 and data['xs'].shape[1] == 1
    #     batch_size, num_pts = data['xs'].shape[0], data['xs'].shape[2]
    #     # B1NC -> BCN
    #     input = data['xs'].transpose(1,3).squeeze(3)
    #     x1, x2 = input[:,:2,:], input[:,2:,:]
    #     motion = x2 - x1

    #     pos = x1 # B2N
    #     grid_pos = self.grid_center(batch_size) # B2N

    #     pos_embed = self.pos_embed(pos) # BCN
    #     grid_pos_embed = self.grid_pos_embed(grid_pos)

    #     d = self.init_project(motion) + pos_embed # BCN

    #     res_logits, res_e_hat = [], []
    #     for i in range(self.layer_num):
    #         d, logits, e_hat = self.layer_blocks[i](data['xs'], d, grid_pos_embed) # BCN
    #         res_logits.append(logits), res_e_hat.append(e_hat)
    #     return res_logits, res_e_hat 
    
def get_model_convmatch( config, N, model_width, en_checkpointing ):    
    
    if(config.device == 'cuda'):
        return ConvMatch( use_gpu=True, use_tpu=False )
    elif(config.device == 'tpu'):
        return ConvMatch( use_gpu=False, use_tpu=True )
    else:
        return ConvMatch( use_gpu=False, use_tpu=False )

if __name__ == '__main__':
    
    convMatch = ConvMatch( use_gpu=False, use_tpu=False )
    
    B = 32
    N = 512
    C = 4
    
    inp = torch.rand(B, 1, N, C)
    
    out = convMatch( inp )
    
