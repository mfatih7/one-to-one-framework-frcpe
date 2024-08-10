from models.models_LTFGC import get_model_LTFGC as LTFGC_get_model
from models.models_OANET import get_model_OANET as OANET_get_model
from models.models_OANET import get_model_OANET_Iter as OANET_Iter_get_model

from models.models_convmatch import get_model_convmatch as convmatch_get_model
from models.models_convmatch_plus import get_model_convmatch_plus as convmatch_get_model_plus
from models.models_CLNet import get_model_CLNet as CLNet_get_model
from models.models_MS2DGNET import get_model_MS2DGNET as MS2DGNET_get_model

from models.models_exp4 import get_model as model_exp4_get_model
from models.models_exp12 import get_model as model_exp12_get_model

import torch
from torchsummary import summary
from thop import profile
from models.torchSummaryWrapper import get_torchSummaryWrapper
        
def get_model( config, model_type, N, model_width, en_checkpointing ):   
    
    if( N == 512 or N == 1024 or N == 2048 or N == 4096 ):
        
        print( f'Getting the model {model_type}')
        
        if( model_type == 'LTFGC' ):
            return LTFGC_get_model( N, model_width, en_checkpointing )
        elif( model_type == 'OANET' ):
            return OANET_get_model( N, model_width, en_checkpointing )
        elif( model_type == 'OANET_Iter' ):
            return OANET_Iter_get_model( N, model_width, en_checkpointing )
        
        elif( model_type == 'convmatch' ):
            return convmatch_get_model( config, N, model_width, en_checkpointing )
        elif( model_type == 'convmatch_plus' ):
            return convmatch_get_model_plus( config, N, model_width, en_checkpointing )
        elif( model_type == 'CLNet' ):
            return CLNet_get_model( config, N, model_width, en_checkpointing )
        elif( model_type == 'MS2DGNET' ):
            return MS2DGNET_get_model( N, model_width, en_checkpointing )
        
        elif( model_type == 'model_EOT'):
            return model_exp4_get_model( config, N, model_width, en_checkpointing )
        elif( model_type == 'model_CNN_cont_0' or model_type == 'model_CNN_cont_1' or model_type == 'model_CNN_cont_2' or model_type == 'model_CNN_cont_3' ):
            return model_exp12_get_model( config, N, model_width, en_checkpointing )
        
        else:
            raise ValueError(f"The provided argument is not valid: {model_type}")
    else:        
        raise ValueError(f"The provided argument is not valid: {N}")

def get_model_structure( config, device, model, N, model_width, en_grad_checkpointing):   
    
    if(en_grad_checkpointing==False):
        summary(model, (config.input_channel_count, N, model_width), batch_size=2, device=device ) # batch_size must be at least 2 to prevent batch norm errors
    else:                    
        summary(get_torchSummaryWrapper( model ), (config.input_channel_count, N, model_width), batch_size=2, device=device ) # batch_size must be at least 2 to prevent batch norm errors
    input_thop = torch.randn(2, config.input_channel_count, N, model_width, device=device)  # Example input tensor
    flops, params = profile(model, inputs=(input_thop, ))
    flops = int(flops)
    params = int(params)
    print(f"Model FLOPs: {flops:,}")
    print(f"Model Parameters: {params:,}")
    
def get_model_params_and_FLOPS( config, device, model, N, model_width, en_grad_checkpointing):   
    input_thop = torch.randn(2, config.input_channel_count, N, model_width, device=device)  # Example input tensor
    flops, params = profile(model, inputs=(input_thop, ))
    flops = int(flops)
    params = int(params)
    print(f"Model FLOPs: {flops:,}")
    print(f"Model Parameters: {params:,}")
    
    return params, flops
    