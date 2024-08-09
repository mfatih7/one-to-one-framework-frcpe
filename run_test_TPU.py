import sys

from config import get_config

from process.test_1_1_TPU_single import test as test_1_1_TPU_single_test
from process.test_n_n_TPU_single import test as test_n_n_TPU_single_test

from process.test_1_1_TPU_multi import test as test_1_1_TPU_multi_test
from process.test_n_n_TPU_multi import test as test_n_n_TPU_multi_test
from process.test_n_n_CLNET_TPU_multi import test as test_n_n_CLNET_TPU_multi_test

from process.test_1_1_TPU_spmd import test as test_1_1_TPU_spmd_test

import torch_xla.distributed.xla_multiprocessing as xmp    # for multi core processing

import tpu_related.set_env_variables_for_TPU as set_env_variables_for_TPU

import checkpoint

# Wrap most of you main script’s code within if __name__ == '__main__': block, to make sure it doesn’t run again
# (most likely generating error) when each worker process is launched. You can place your dataset and DataLoader
# instance creation logic here, as it doesn’t need to be re-executed in workers.

if __name__ == '__main__':
    
    if(len(sys.argv)>=4):
        arg_first_experiment = int(sys.argv[1])
        arg_model_type       = sys.argv[2]
        arg_model_exp_no     = int(sys.argv[3])
        if(len(sys.argv)>=6):
            arg_optimizer_type = sys.argv[4]
            arg_learning_rate  = float(sys.argv[5])
            if(len(sys.argv)==7):
                arg_momentum   = float(sys.argv[6])

    set_env_variables_for_TPU.set_env_variables_for_TPU_PJRT( )

    config = get_config()
    
    if( config.tpu_cores == 'spmd' ):
        set_env_variables_for_TPU.set_env_variables_for_TPU_SPMD( )
    
    set_env_variables_for_TPU.set_env_debug_variables_for_TPU_PJRT( config )
    
    if(len(sys.argv)>=4):
        config.first_experiment = arg_first_experiment
        config.model_type = arg_model_type
        config.model_exp_no = arg_model_exp_no
        
        if(len(sys.argv)>=6):
            config.optimizer_type = arg_optimizer_type
            config.learning_rate = arg_learning_rate
            if(len(sys.argv)==7):
                config.momentum = arg_momentum
        if(config.optimizer_type == 'SGD'):
            print(f'For TPUv4 Preempt run first_experiment is {config.first_experiment}, model_type is {config.model_type}, '
                  f'model_exp_no is {config.model_exp_no}, optimizer is {config.optimizer_type}, learning_rate is {config.learning_rate}, momentum is {config.momentum}')
        else:
            print(f'For TPUv4 Preempt run first_experiment is {config.first_experiment}, model_type is {config.model_type}, '
                  f'model_exp_no is {config.model_exp_no}, optimizer is {config.optimizer_type}, learning_rate is {config.learning_rate}')
    elif(len(sys.argv)==1):
        if(config.output_data_storage_local_or_bucket == 'bucket'):
            raise ValueError(f"For preemtible TPUs use call run_train_TPU.py from command line with {3} arguments.")
    else:
        raise ValueError(f"The provided arguments are not valid: {len(sys.argv)}")
    
    experiment_no = config.first_experiment
    
    config.update_output_folder(experiment_no)
    
    if(config.output_data_storage_local_or_bucket == 'bucket'):  # For Preemtible devices
        config.copy_output_folder_from_bucket_to_local()
        
    if( config.tpu_cores == 'spmd' ):
        chkpt_mgr = checkpoint.get_chkpt_mgr(config)    
    
    if(config.input_type=='1_to_1'):
        config.update_training_params_for_test()
    
    N_images_in_batch = config.training_params[0][0]
    N = config.training_params[0][1]
    batch_size = config.training_params[0][2]
    
    if(config.operation == 'test'):
    
        if(config.input_type == '1_to_1'):
            
            if( N_images_in_batch >= 1 and N == batch_size ):
        
                if( config.tpu_cores == 'single' ):
                    
                    learning_rate = config.learning_rate
                    n_epochs = config.n_epochs
                    num_workers = config.num_workers
                    model_type = config.model_type
                    en_grad_checkpointing = config.en_grad_checkpointing
                    
                    print('Testing starts for ' + 'test_1_1_TPU_single_test')
                    
                    test_results = test_1_1_TPU_single_test(   
                                                            config,
                                                            experiment_no,
                                                            learning_rate,
                                                            n_epochs,
                                                            num_workers,
                                                            model_type,
                                                            en_grad_checkpointing,
                                                            N_images_in_batch,
                                                            N,
                                                            batch_size, )
                    
                elif( config.tpu_cores == 'multi' ):
                    
                    print('Testing starts for ' + 'test_1_1_TPU_multi_test')
                    
                    FLAGS = {}
                    FLAGS['config']                     = config
                    FLAGS['experiment_no']              = experiment_no
                    FLAGS['learning_rate']              = config.learning_rate * config.n_tpu_cores  # Learning Rate is increased for multi cores operation
                    FLAGS['n_epochs']                   = config.n_epochs
                    FLAGS['num_workers']                = config.num_workers
                    FLAGS['model_type']                 = config.model_type
                    FLAGS['en_grad_checkpointing']      = config.en_grad_checkpointing
                    FLAGS['N_images_in_batch']          = config.training_params[0][0]
                    FLAGS['N']                          = config.training_params[0][1]
                    FLAGS['batch_size']                 = config.training_params[0][2]
                    
                    xmp.spawn(test_1_1_TPU_multi_test, args=(FLAGS,) )
                    
                elif( config.tpu_cores == 'spmd' ):
                
                    learning_rate = config.learning_rate
                    n_epochs = config.n_epochs
                    num_workers = config.num_workers
                    model_type = config.model_type
                    optimizer_type = config.optimizer_type
                    en_grad_checkpointing = config.en_grad_checkpointing

                    print('Testing starts for ' + 'test_1_1_TPU_spmd_test')

                    test_results = test_1_1_TPU_spmd_test(   
                                                            config,
                                                            experiment_no,
                                                            learning_rate,
                                                            n_epochs,
                                                            num_workers,
                                                            model_type,
                                                            en_grad_checkpointing,
                                                            N_images_in_batch,
                                                            N,
                                                            batch_size,
                                                            optimizer_type,
                                                            chkpt_mgr, )                    
                else:
                    raise ValueError(f"The provided arguments are not valid: {config.tpu_cores}")                    
            else:
                raise ValueError(f"The provided arguments are not valid: {N_images_in_batch} {N} {batch_size}")
            
        elif(config.input_type == 'n_to_n'):
            
            if( N_images_in_batch == batch_size ):
    
                if( config.tpu_cores == 'single' ):
                    
                    learning_rate = config.learning_rate
                    n_epochs = config.n_epochs
                    num_workers = config.num_workers
                    model_type = config.model_type
                    en_grad_checkpointing = config.en_grad_checkpointing
                    
                    print('Testing starts for ' + 'test_n_n_TPU_single_test')
                    
                    test_results = test_n_n_TPU_single_test(   
                                                            config,
                                                            experiment_no,
                                                            learning_rate,
                                                            n_epochs,
                                                            num_workers,
                                                            model_type,
                                                            en_grad_checkpointing,
                                                            N_images_in_batch,
                                                            N,
                                                            batch_size, )
                elif( config.tpu_cores == 'multi' ):
                    
                    print('Testing starts for ' + 'test_n_n_TPU_multi_test')
                    
                    FLAGS = {}
                    FLAGS['config']                     = config
                    FLAGS['experiment_no']              = experiment_no
                    FLAGS['learning_rate']              = config.learning_rate * config.n_tpu_cores  # Learning Rate is increased for multi cores operation
                    FLAGS['n_epochs']                   = config.n_epochs
                    FLAGS['num_workers']                = config.num_workers
                    FLAGS['model_type']                 = config.model_type
                    FLAGS['en_grad_checkpointing']      = config.en_grad_checkpointing
                    FLAGS['N_images_in_batch']          = config.training_params[0][0]
                    FLAGS['N']                          = config.training_params[0][1]
                    FLAGS['batch_size']                 = config.training_params[0][2]
                    
                    if( config.model_type != 'CLNet' ):                    
                        xmp.spawn(test_n_n_TPU_multi_test, args=(FLAGS,) )
                    else:
                        xmp.spawn(test_n_n_CLNET_TPU_multi_test, args=(FLAGS,) )
                    
                else:
                    raise ValueError(f"The provided arguments are not valid: {config.tpu_cores}")                    
            else:
                raise ValueError(f"The provided arguments are not valid: {N_images_in_batch} {batch_size}")            
        else:            
            raise ValueError(f"The provided arguments are not valid: {config.input_type}")
    else:
        raise ValueError(f"The provided arguments are not valid: {config.operation}")