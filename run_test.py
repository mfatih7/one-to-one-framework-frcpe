from config import get_config

from process.test_1_1 import test as test_1_1_test
from process.test_n_n import test as test_n_n_test
from process.test_n_n_CLNET import test as test_n_n_test_CLNET

# Wrap most of you main script’s code within if __name__ == '__main__': block, to make sure it doesn’t run again
# (most likely generating error) when each worker process is launched. You can place your dataset and DataLoader
# instance creation logic here, as it doesn’t need to be re-executed in workers.

if __name__ == '__main__':
    
    config = get_config()
    
    experiment_no = config.first_experiment
    
    config.update_output_folder(experiment_no)
    if(config.input_type=='1_to_1'):
        config.update_training_params_for_test()
    
    learning_rate = config.learning_rate
    n_epochs = config.n_epochs
    num_workers = config.num_workers
    model_type = config.model_type
    en_grad_checkpointing = config.en_grad_checkpointing
    N_images_in_batch = config.training_params[0][0]
    N = config.training_params[0][1]
    batch_size = config.training_params[0][2]
    
    if(config.operation == 'test'):
    
        if(config.input_type=='1_to_1'):
            
            print('Testing starts for ' + 'test_1_1_test')
        
            test_results = test_1_1_test(   
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
            
        elif(config.input_type=='n_to_n'):
            
            if(config.model_type != 'CLNet' ):
                
                print('Testing starts for ' + 'test_n_n_test')
            
                test_results = test_n_n_test(   
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
            else:
                
                print('Testing starts for ' + 'test_n_n_test_CLNET')
                
                test_results = test_n_n_test_CLNET(   
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
            
        else:
            
            raise ValueError(f"The provided arguments are not valid: {config.input_type}")
    else:
        raise ValueError(f"The provided arguments are not valid: {config.operation}")