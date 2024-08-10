import torch
import torch.nn as nn

from loss_module import loss_functions_n_to_n

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = - ( xx + inner + xx.transpose(2, 1) )
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)topk是自带函数
    return idx



def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_pts = x.size(2)
    x = x.view(batch_size, -1, num_pts) #change
    if idx is None:
        idx_out = knn(x, k=k)  # (batch_size, num_points, k)
    else:
        idx_out = idx
        
    device = x.device

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_pts #change

    idx = idx_out + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_pts, -1)[idx, :]
    feature = feature.view(batch_size, num_pts, k, num_dims) #change

    x = x.view(batch_size, num_pts, 1, num_dims).repeat(1, 1, k, 1) #change
    feature = torch.cat((x, x - feature), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature



class transformer(nn.Module):
    def __init__(self,in_channel,out_channels=None):
        nn.Module.__init__(self)
        self.att1 = nn.Sequential(
            nn.InstanceNorm2d(in_channel, eps=1e-3),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, out_channels, kernel_size=1),
        )
        self.attq1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.attk1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.attv1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.gamma1 = nn.Parameter(torch.ones(1))

    def forward(self, x_row, x_local):
        # 局部attention
        x_local = self.att1(x_local)
        q = self.attq1(x_local)
        k = self.attk1(x_local)
        v = self.attv1(x_local)

        att = torch.mul(q, k)
        att = torch.softmax(att, dim=3)
        qv = torch.mul(att, v)
        out_local = torch.sum(qv, dim=3).unsqueeze(3)
        out = x_row + self.gamma1 * out_local #x_row只有在这里有用

        return out

class MLPs(nn.Module):
    def __init__(self, channels, out_channels=None):
        nn.Module.__init__(self)
        self.conv = nn.Sequential(
                nn.InstanceNorm2d(channels, eps=1e-3),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.Conv2d(channels, out_channels, kernel_size=1),
                )
    def forward(self,x):
        out = self.conv(x)
        return out

class PointCN_Layer_1(nn.Module):
    def __init__(self, w, d):
        super(PointCN_Layer_1, self).__init__()

        self.w = w
        self.d = d
        
        self.Conv2d = nn.Conv2d(in_channels=1, out_channels=self.d, kernel_size=(1,self.w), stride=(1,1) ) 
        
    def forward(self, x):
        out = self.Conv2d(x)
        return out

class PointCN_CN(nn.Module):
    def __init__(self, d):
        super(PointCN_CN, self).__init__()

        self.d = d        
        self.InstanceNorm2d = nn.InstanceNorm2d(self.d, eps=1e-3)
        
    def forward(self, x):
        out = self.InstanceNorm2d(x)        
        return out

class PointCN_ResNet_Block(nn.Module):
    def __init__(self, d_in, d_out, CN_active_or_CN_inactive, residual_connections_active_or_residual_connections_inactive):
        super(PointCN_ResNet_Block, self).__init__()

        self.d_in = d_in
        self.d_out = d_out
        self.CN_active_or_CN_inactive = CN_active_or_CN_inactive
        self.residual_connections_active_or_residual_connections_inactive = residual_connections_active_or_residual_connections_inactive        
        
        self.contextnorm_1 = PointCN_CN( self.d_in )
        self.batchnorm_1 = nn.BatchNorm2d(self.d_in )
        self.Conv2d_1 = nn.Conv2d(in_channels=self.d_in, out_channels=self.d_out, kernel_size=(1,1), stride=(1,1) )
        
        self.contextnorm_2 = PointCN_CN( self.d_out )
        self.batchnorm_2 = nn.BatchNorm2d(self.d_out )
        self.Conv2d_2 = nn.Conv2d(in_channels=self.d_out, out_channels=self.d_out, kernel_size=(1,1), stride=(1,1) ) 

        if(self.d_in != self.d_out):
            self.short_cut = nn.Conv2d(self.d_in, self.d_out, kernel_size=(1,1) )
        
        self.relu = nn.ReLU()
        
    def forward(self, x):        
        
        if(self.CN_active_or_CN_inactive == 'CN_inactive'):
            out = x
        elif(self.CN_active_or_CN_inactive == 'CN_active'):
            out = self.contextnorm_1(x)
        
        out = self.relu(self.batchnorm_1(out))
        
        out = self.Conv2d_1(out)
        
        if(self.CN_active_or_CN_inactive == 'CN_inactive'):
            out = out
        elif(self.CN_active_or_CN_inactive == 'CN_active'):
            out = self.contextnorm_2(out)
            
        out = self.relu(self.batchnorm_2(out))
        
        out = self.Conv2d_2(out)
        
# shortcut
        if(self.residual_connections_active_or_residual_connections_inactive == 'residual_connections_active'):
            if(self.d_in != self.d_out):                
                x = self.short_cut(x)
            out = out + x                
        elif(self.residual_connections_active_or_residual_connections_inactive == 'residual_connections_inactive'):
            out = out
        
        return out
    
class OANET_pool(nn.Module):
    def __init__(self, d, m):
        super(OANET_pool, self).__init__()

        self.d = d
        self.m = m
        
        self.contextnorm = PointCN_CN( self.d )        
        self.batchnorm = nn.BatchNorm2d( self.d )
        self.relu = nn.ReLU() 
        self.Conv2d = nn.Conv2d( in_channels=self.d, out_channels=self.m, kernel_size=(1,1), stride=(1,1) ) 
        self.softmax = nn.Softmax(dim=2)    
        
    def forward(self, x_level_1):      
        
        out = self.relu(self.batchnorm(self.contextnorm(x_level_1)))
        out = self.Conv2d(out)
        Spool = self.softmax(out)      
        
        out = torch.matmul( x_level_1.squeeze(3), torch.transpose(Spool, 1, 2).squeeze(3) ).unsqueeze(3)
        
        return out
    
class OANET_unpool(nn.Module):
    def __init__(self, d, m ):
        super(OANET_unpool, self).__init__()

        self.d = d
        self.m = m
        
        self.contextnorm = PointCN_CN( self.d )
        self.batchnorm = nn.BatchNorm2d( self.d )
        self.relu = nn.ReLU()
        self.Conv2d = nn.Conv2d( in_channels=self.d, out_channels=self.m, kernel_size=(1,1), stride=(1,1) ) 
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x_level_1, x_level_2):      
        
        out = self.relu(self.batchnorm(self.contextnorm(x_level_1)))
        out = self.Conv2d(out)
        Sunpool = self.softmax(out)
        
        out = torch.matmul( x_level_2.squeeze(3), Sunpool.squeeze(3) ).unsqueeze(3)
        
        return out

class Order_Aware_Filter_Block(nn.Module):
    def __init__(self, m, d):
        super(Order_Aware_Filter_Block, self).__init__()
            
        self.m = m
        self.d = d
        
        self.contextnorm_1 = PointCN_CN( self.d ) 
        self.batchnorm_1 = nn.BatchNorm2d( self.d)
        self.Conv2d_1 = nn.Conv2d( in_channels=self.d, out_channels=self.d, kernel_size=(1,1), stride=(1,1) ) 
        
        self.batchnorm_2 = nn.BatchNorm2d( self.m )
        self.Conv2d_2 = nn.Conv2d( in_channels=self.m, out_channels=self.m, kernel_size=(1,1), stride=(1,1) ) 
        
        self.contextnorm_3 = PointCN_CN( self.d ) 
        self.batchnorm_3 = nn.BatchNorm2d( self.d )
        self.Conv2d_3 = nn.Conv2d( in_channels=self.d, out_channels=self.d, kernel_size=(1,1), stride=(1,1) ) 
        
        self.relu = nn.ReLU()        
        
    def forward(self, x):
        
        out = self.relu(self.batchnorm_1(self.contextnorm_1(x)))
        out_short_cut = self.Conv2d_1(out)
        
        out = torch.transpose(out_short_cut, 1, 2)
        
        out = self.relu(self.batchnorm_2(out))
        out = self.Conv2d_2(out)
        
        out = torch.transpose(out, 1, 2)
        
        out = out + out_short_cut
        
        out = self.relu(self.batchnorm_3(self.contextnorm_3(out)))
        out = self.Conv2d_3(out)

        out = out + x
        
        return out
    
class MS2DGNET(nn.Module):
    def __init__(self, n, model_width, inner_dimension, m, CN_active_or_CN_inactive, residual_connections_active_or_residual_connections_inactive):        
        super(MS2DGNET, self).__init__()
            
        self.n = n
        self.w = model_width
        self.d = inner_dimension
        self.m = m
        
        self.k = 20
        
        self.CN_active_or_CN_inactive = CN_active_or_CN_inactive        
        self.residual_connections_active_or_residual_connections_inactive = residual_connections_active_or_residual_connections_inactive
        
        self.layer_1_stage_1 = PointCN_Layer_1( self.w, self.d//2 )
        
        self.att1_1 = transformer(self.d, self.d // 2)
        self.s1_1 = MLPs(self.d, self.d)
        # self.dropout = nn.Dropout(p=0.1)
        
        self.ResNet_1_stage_1 = PointCN_ResNet_Block( ( 3 * self.d) // 2, self.d, self.CN_active_or_CN_inactive, self.residual_connections_active_or_residual_connections_inactive )
        self.ResNet_2_stage_1 = PointCN_ResNet_Block( self.d, self.d, self.CN_active_or_CN_inactive, self.residual_connections_active_or_residual_connections_inactive )
        self.ResNet_3_stage_1 = PointCN_ResNet_Block( self.d, self.d, self.CN_active_or_CN_inactive, self.residual_connections_active_or_residual_connections_inactive )
        
        self.OANET_pool_stage_1 = OANET_pool( self.d, self.m )
        
        self.Order_Aware_Filter_Block_1_stage_1 = Order_Aware_Filter_Block( self.m, self.d )
        self.Order_Aware_Filter_Block_2_stage_1 = Order_Aware_Filter_Block( self.m, self.d )
        self.Order_Aware_Filter_Block_3_stage_1 = Order_Aware_Filter_Block( self.m, self.d )
        
        self.OANET_unpool_stage_1 = OANET_unpool( self.d, self.m)
        
        self.ResNet_4_stage_1 = PointCN_ResNet_Block( 2*self.d, self.d, self.CN_active_or_CN_inactive, self.residual_connections_active_or_residual_connections_inactive )
        self.ResNet_5_stage_1 = PointCN_ResNet_Block( self.d, self.d, self.CN_active_or_CN_inactive, self.residual_connections_active_or_residual_connections_inactive )
        self.ResNet_6_stage_1 = PointCN_ResNet_Block( self.d, self.d, self.CN_active_or_CN_inactive, self.residual_connections_active_or_residual_connections_inactive )
        
        self.layer_last_stage_1 = nn.Conv2d(in_channels=self.d, out_channels=1, kernel_size=(1,1), stride=(1,1) ) 
        
        #########################################################################################################################################################################################

        self.layer_1_stage_2 = PointCN_Layer_1( self.w+2, self.d )
        
        self.ResNet_1_stage_2 = PointCN_ResNet_Block( self.d, self.d, self.CN_active_or_CN_inactive, self.residual_connections_active_or_residual_connections_inactive )
        self.ResNet_2_stage_2 = PointCN_ResNet_Block( self.d, self.d, self.CN_active_or_CN_inactive, self.residual_connections_active_or_residual_connections_inactive )
        self.ResNet_3_stage_2 = PointCN_ResNet_Block( self.d, self.d, self.CN_active_or_CN_inactive, self.residual_connections_active_or_residual_connections_inactive )
        
        self.OANET_pool_stage_2 = OANET_pool( self.d, self.m )
        
        self.Order_Aware_Filter_Block_1_stage_2 = Order_Aware_Filter_Block( self.m, self.d )
        self.Order_Aware_Filter_Block_2_stage_2 = Order_Aware_Filter_Block( self.m, self.d )
        self.Order_Aware_Filter_Block_3_stage_2 = Order_Aware_Filter_Block( self.m, self.d )
        
        self.OANET_unpool_stage_2 = OANET_unpool( self.d, self.m )
        
        self.ResNet_4_stage_2 = PointCN_ResNet_Block( 2*self.d, self.d, self.CN_active_or_CN_inactive, self.residual_connections_active_or_residual_connections_inactive )
        self.ResNet_5_stage_2 = PointCN_ResNet_Block( self.d, self.d, self.CN_active_or_CN_inactive, self.residual_connections_active_or_residual_connections_inactive )
        self.ResNet_6_stage_2 = PointCN_ResNet_Block( self.d, self.d, self.CN_active_or_CN_inactive, self.residual_connections_active_or_residual_connections_inactive )
        
        self.layer_last_stage_2 = nn.Conv2d(in_channels=self.d, out_channels=1, kernel_size=(1,1), stride=(1,1) ) 
        
        self.relu = nn.ReLU()        
        self.tanh = nn.Tanh()

    def forward(self, x):
        
### STAGE 1 ###
        
        out = self.layer_1_stage_1(x)
        
        x_att1 = get_graph_feature(out, k=self.k)
        x_SDG1 = self.att1_1(out, x_att1)

        x_f = get_graph_feature(x_SDG1, k=self.k)
        x_l11 = self.s1_1(x_f)
        x_SDG2 = x_l11.max(dim=-1, keepdim=False)[0]
        #x_SDG2 = self.dropout(x_l2) #sun3d
        x_SDG2 = x_SDG2.unsqueeze(3)
        out = torch.cat((x_SDG1,x_SDG2), dim=1)
        
        out = self.ResNet_1_stage_1(out)
        out = self.ResNet_2_stage_1(out)
        out_1st_ResNet_Group = self.ResNet_3_stage_1(out)
        
        out_pool = self.OANET_pool_stage_1(out_1st_ResNet_Group)
        
        out = self.Order_Aware_Filter_Block_1_stage_1(out_pool)
        out = self.Order_Aware_Filter_Block_2_stage_1(out)
        out = self.Order_Aware_Filter_Block_3_stage_1(out)
        
        out_unpool = self.OANET_unpool_stage_1(out_1st_ResNet_Group, out)
        
        out = torch.cat( [out_1st_ResNet_Group, out_unpool], dim=1)
        
        out = self.ResNet_4_stage_1(out)
        out = self.ResNet_5_stage_1(out)
        out = self.ResNet_6_stage_1(out)
        
        out = self.layer_last_stage_1(out)
        
        out_1 = torch.reshape(out, (-1, self.n) )

### STAGE 1 TO STAGE 2 ###
        
        e_hat = loss_functions_n_to_n.weighted_8points(x[:,:,:,:4], out)
        residual = loss_functions_n_to_n.batch_episym(x[:,0,:,:2], x[:,0,:,2:4], e_hat).unsqueeze(dim=1).unsqueeze(dim=-1).detach()
        
        logits = torch.relu( torch.tanh( out ) ).detach()
        
        x_stage_2 = torch.cat( ( x, residual, logits ), dim=-1)
        
### STAGE 2 ###
        
        out = self.layer_1_stage_2( x_stage_2 )
        
        out = self.ResNet_1_stage_2(out)
        out = self.ResNet_2_stage_2(out)
        out_1st_ResNet_Group = self.ResNet_3_stage_2(out)
        
        out_pool = self.OANET_pool_stage_2(out_1st_ResNet_Group)
        
        out = self.Order_Aware_Filter_Block_1_stage_2(out_pool)
        out = self.Order_Aware_Filter_Block_2_stage_2(out)
        out = self.Order_Aware_Filter_Block_3_stage_2(out)
        
        out_unpool = self.OANET_unpool_stage_2(out_1st_ResNet_Group, out)
        
        out = torch.cat( [out_1st_ResNet_Group, out_unpool], dim=1)
        
        out = self.ResNet_4_stage_2(out)
        out = self.ResNet_5_stage_2(out)
        out = self.ResNet_6_stage_2(out)
        
        out = self.layer_last_stage_2(out)
        
        out = torch.reshape(out, (-1, self.n) )

        return [ out_1, out ]     

def get_model_MS2DGNET( N, model_width, en_checkpointing ):   
    
    inner_dimension = 128
    m = 500
    
    CN_active_or_CN_inactive = 'CN_active'
    # CN_active_or_CN_inactive = 'CN_inactive'    
    
    residual_connections_active_or_residual_connections_inactive = 'residual_connections_active'
    # residual_connections_active_or_residual_connections_inactive = 'residual_connections_inactive'
    
    return MS2DGNET( N, model_width, inner_dimension, m, CN_active_or_CN_inactive, residual_connections_active_or_residual_connections_inactive )

if __name__ == '__main__':
    
    import os    
    os.chdir( os.path.dirname( os.getcwd( ) ) )
    
    from loss_module import loss_functions_n_to_n
    
    N = 512
    model_width = 4
    inner_dimension = 128
    m = 500    
    
    MS2DGNET_00 = MS2DGNET( N, model_width, inner_dimension, m, CN_active_or_CN_inactive = 'CN_active',
                            residual_connections_active_or_residual_connections_inactive = 'residual_connections_active' )
    MS2DGNET_01 = MS2DGNET( N, model_width, inner_dimension, m, CN_active_or_CN_inactive = 'CN_active',
                            residual_connections_active_or_residual_connections_inactive = 'residual_connections_inactive' )
    MS2DGNET_10 = MS2DGNET( N, model_width, inner_dimension, m, CN_active_or_CN_inactive = 'CN_inactive',
                            residual_connections_active_or_residual_connections_inactive = 'residual_connections_active' )
    MS2DGNET_11 = MS2DGNET( N, model_width, inner_dimension, m, CN_active_or_CN_inactive = 'CN_inactive',
                            residual_connections_active_or_residual_connections_inactive = 'residual_connections_inactive' )
        
    