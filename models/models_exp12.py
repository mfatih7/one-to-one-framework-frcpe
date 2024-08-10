import torch
import torch.nn as nn

class Non_Lin(nn.Module):
    def __init__(self, non_lin):
        super(Non_Lin, self).__init__()
        
        if( non_lin == 'ReLU' ):
            self.non_lin = nn.ReLU()
        elif( non_lin == 'LeakyReLU' ):
            self.non_lin = nn.LeakyReLU()
        elif non_lin == 'tanh':
            self.non_lin = nn.Tanh()

    def forward(self, x):        
        out = self.non_lin( x )        
        return out

class Conv2d_N(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride ):
        super(Conv2d_N, self).__init__()
        
        self.conv2d = nn.Conv2d( in_channels = in_channels, out_channels = out_channels, kernel_size = kernel_size, stride = stride, bias = False, )
        
        self.norm = nn.BatchNorm2d( out_channels, track_running_stats=False, )
            
    def forward(self, x):
        
        out = self.norm( self.conv2d(x) )
            
        return out
    
class Width_Reduction(nn.Module):
    def __init__(self, in_width, out_channels, non_lin):
        super(Width_Reduction, self).__init__()
        
        self.width_reduction = Conv2d_N( in_channels = 2, out_channels = out_channels, kernel_size = (1, in_width), stride = (1,1), )
        
        self.non_lin = Non_Lin( non_lin )
            
    def forward(self, x):
        
        out = self.non_lin( self.width_reduction(x) )
        
        return out

class Block_Height_Reducing_Filtering(nn.Module):
    def __init__(self, height_in, height_out, channel_count_in, channel_count_out, non_lin):
        super(Block_Height_Reducing_Filtering, self).__init__()
                
        self.height_reductions = Conv2d_N( in_channels = channel_count_in, out_channels = channel_count_in, kernel_size = (2,1), stride = (2,1) )
        
        self.channel_expansion = Conv2d_N( in_channels = channel_count_in, out_channels = channel_count_out, kernel_size = (1,1), stride = (1,1) )
                
        self.non_lin = Non_Lin( non_lin )
        
    def forward(self, x):
        
        out = self.non_lin( self.height_reductions( x ) ) 
        
        out = self.non_lin( self.channel_expansion( out ) )
        
        return out

class model_exp_00(nn.Module):
    def __init__(self,   N,
                         in_width,
                         heights_in,
                         heights_out,
                         channel_counts_in,
                         channel_counts_out,
                         non_lin, ):
        super(model_exp_00, self).__init__()
        
        self.N = N
        self.in_width = in_width
        self.non_lin = non_lin
        
        self.heights_in = heights_in 
        self.heights_out = heights_out
        self.channel_counts_in = channel_counts_in
        self.channel_counts_out = channel_counts_out
        
        self.n_blocks = int( len(self.channel_counts_in) )         
        
        layers = []
        
        for block_no in range(self.n_blocks):
        
            if(block_no==0):                
                layers.append( Width_Reduction( in_width = self.in_width, out_channels = self.channel_counts_out[block_no], non_lin = self.non_lin, ) )
            else:
                layers.append( Block_Height_Reducing_Filtering( height_in = self.heights_in[block_no],
                                                                height_out = self.heights_out[block_no],
                                                                channel_count_in = self.channel_counts_in[block_no],
                                                                channel_count_out = self.channel_counts_out[block_no],
                                                                non_lin = non_lin, ) )
        self.net = nn.Sequential(*layers)
        
        self.initial_fully_connected_size = self.channel_counts_out[-1]
        
        self.fc1 = nn.Linear(self.initial_fully_connected_size * 1 * 1, 128 )
        
        self.fc2 = nn.Linear( 128 * 1 * 1, 1)
        
        self.non_lin = Non_Lin( non_lin )
        
    def forward(self, x):
        
        out = self.net( x )
        
        out = out.view( -1, self.initial_fully_connected_size * 1 * 1 )
        out = self.non_lin( self.fc1(out) )
        out = self.fc2(out)
        
        return out
        
def get_model( config, N, model_width, en_checkpointing, model_adjust_params = None ):   

    if( N == 512 or N == 1024 or N == 2048 ):
        
        in_width = model_width
        
        non_lin = 'ReLU'
        # non_lin = 'LeakyReLU'  
        # non_lin = 'tanh'
        
        if(model_adjust_params != None):            
            raise NotImplementedError(f"The feature '{model_adjust_params}' is not implemented yet.")
        else:
            
            if( config.model_exp_no in [  0, ] ):
                exp_no_list_index = [  0, ].index(config.model_exp_no)
                
                heights_in = [       2048,
                               1024+512*0,    512+256*0,   
                                256+128*0,     128+64*0,     
                                  64+32*0,      32+16*0,      
                                   16+8*0,        8+4*0,
                                    4+2*0,        2+1*0, ]
                
                heights_out = [ 1024+512*0,    512+256*0,   
                                 256+128*0,     128+64*0,     
                                   64+32*0,      32+16*0,      
                                    16+8*0,        8+4*0,
                                     4+2*0,        2+1*0,    1, ]
                
                ch_in = 128
                cn_inc = 256
                
                channel_counts_in = [               2,
                                       ch_in+cn_inc*0,    ch_in+cn_inc*1,
                                       ch_in+cn_inc*2,    ch_in+cn_inc*3,
                                       ch_in+cn_inc*4,    ch_in+cn_inc*5,
                                       ch_in+cn_inc*6,    ch_in+cn_inc*7,
                                       ch_in+cn_inc*8,    ch_in+cn_inc*9,  ch_in+cn_inc*10, ]
                
                channel_counts_out = [  ch_in+cn_inc*0,    ch_in+cn_inc*1,
                                        ch_in+cn_inc*2,    ch_in+cn_inc*3,
                                        ch_in+cn_inc*4,    ch_in+cn_inc*5,
                                        ch_in+cn_inc*6,    ch_in+cn_inc*7,
                                        ch_in+cn_inc*8,    ch_in+cn_inc*9,  ch_in+cn_inc*10,  ch_in+cn_inc*11, ]
            
            elif( config.model_exp_no in [  100, ] ):
                exp_no_list_index = [  100, ].index(config.model_exp_no)
                                
                heights_in = [       2048,
                               1024+512*0,    512+256*0,   
                                256+128*0,     128+64*0,     
                                  64+32*0,      32+16*0,      
                                   16+8*0,        8+4*0,
                                    4+2*0,        2+1*0, ]
                
                heights_out = [ 1024+512*0,    512+256*0,   
                                 256+128*0,     128+64*0,     
                                   64+32*0,      32+16*0,      
                                    16+8*0,        8+4*0,
                                     4+2*0,        2+1*0,      1, ]
                
                channel_counts_in = [    2,     
                                       128,   256,
                                       384,   512,
                                       768,   1024,
                                      1280,   1536,
                                      2048,   2560,  3072, ]
                
                channel_counts_out = [ 128,   256,
                                       384,   512,
                                       768,   1024,
                                      1280,   1536,
                                      2048,   2560,  3072,  3584, ]
            
            else:
                raise ValueError(f"The provided argument is not valid: {config.model_exp_no}")
                
            if(N==512):
                heights_in = [512] + heights_in[2:]            
                heights_out = [512] + heights_out[2:]
                channel_counts_in = channel_counts_in[0:-2]
                channel_counts_out = channel_counts_out[0:-2]
                
            elif(N==1024):
                heights_in = [1024] + heights_in[1:]            
                heights_out = [1024] + heights_out[1:]
                channel_counts_in = channel_counts_in[0:-1]
                channel_counts_out = channel_counts_out[0:-1]
            elif(N==2048):
                heights_in = [2048] + heights_in[0:]            
                heights_out = [2048] + heights_out[0:]
                channel_counts_in = channel_counts_in[0:]
                channel_counts_out = channel_counts_out[0:]
                
        return model_exp_00( N,
                             in_width,
                             heights_in,
                             heights_out,
                             channel_counts_in,
                             channel_counts_out,
                             non_lin, )
    else:
        raise ValueError(f"The provided argument is not valid: {N}")

if __name__ == '__main__':
    
    import os    
    os.chdir( os.path.dirname( os.getcwd( ) ) )       
    from config import get_config
    
    from models.models import get_model_structure
    from models.models import get_model_params_and_FLOPS
    
    config = get_config()
    
    device = 'cpu'
    
    # N = 512
    # N = 1024
    N = 2048
    model_width = 4
    en_checkpointing = False
    
####################################################################################

    first_model_no = 0
    last_model_no = 1
    
    # first_model_no = 100
    # last_model_no = 101
        
####################################################################################
    
    for i in range( first_model_no, last_model_no, 1 ):
        config.model_exp_no = i 
        print(f'config.model_exp_no: {config.model_exp_no}')
        model = get_model( config, N, model_width, en_checkpointing ).to(device)
        get_model_structure( config, device, model, N, model_width, en_grad_checkpointing=0)
        # get_model_params_and_FLOPS( config, device, model, N, model_width, en_grad_checkpointing=0)
        print('-'*80)
        
        for name, layer in model.named_modules():
            if ( 'conv2d' in name ):
                print(name)
               
else:
    
    pass
    