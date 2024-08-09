import torch
from torch.utils.data import DataLoader
import numpy as np
import time

from datasets.datasets import get_dataset
from datasets.datasets import collate_fn2

from models.models import get_model
from models.models import get_model_structure

from samplers.CustomBatchSampler import get_sampler

from optimizer.optimizer import get_optimizer

from loss_module import loss_functions
import checkpoint
import plots

def train_and_val( 
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
                    optimizer_type, ):
    
    device = config.device
   
    model_width = config.model_width
    model = get_model( config, model_type, N, model_width, en_grad_checkpointing ).to(device)
    
    get_model_structure( config, device, model, N, model_width, en_grad_checkpointing)

    optimizer = get_optimizer( config, optimizer_type, model, learning_rate, )

    checkpoint.save_initial_checkpoint( config, model, optimizer )
    epoch, chunk, model, optimizer, success_checkpoint, \
    loss_checkpoint, proc_time_checkpoint = checkpoint.load_checkpoint( config, device, model, optimizer)
    
    batch_count_in_N = int(N/batch_size) 
    stop_debug = 1
    
    accumulator_0 = torch.zeros( N, 1, dtype=torch.float32, requires_grad=False, device=device )
    accumulator_1 = torch.zeros( N, 1, dtype=torch.float32, requires_grad=True, device=device )
    
    if(config.n_chunks==1):

        dataset_train = get_dataset( config, N_images_in_batch, N, batch_size, train_val_test = 'train', chunk=0 )
            
        sampler_train = get_sampler( config, dataset_train, N_images_in_batch, N, batch_size )
        
        dataloader_train = DataLoader(  dataset = dataset_train,
                                        sampler = sampler_train,
                                        pin_memory = True,
                                        num_workers = num_workers,
                                        collate_fn = collate_fn2, )
    
        if(config.validation == 'enable'):
        
            dataset_val = get_dataset( config, N_images_in_batch, N, batch_size, train_val_test = 'val', chunk=0 )
                
            sampler_val = get_sampler( config, dataset_val, N_images_in_batch, N, batch_size )
                        
            dataloader_val = DataLoader(    dataset = dataset_val,
                                            sampler = sampler_val,
                                            pin_memory = True,
                                            num_workers = num_workers,
                                            collate_fn = collate_fn2, )
      
    while epoch < n_epochs[0]:
        while chunk < config.n_chunks:
            
            loss_cls_train = 0
            loss_geo_train = 0
            loss_ess_train = 0
            loss_count_train = 0       
        
            loss_cls_val = 0
            loss_geo_val = 0
            loss_ess_val = 0
            loss_count_val = 0
                    
            confusion_matrix_at_epoch_train_device  = torch.zeros( (2,2), device = device, requires_grad = False )
            confusion_matrix_at_epoch_val_device    = torch.zeros( (2,2), device = device, requires_grad = False )
            
### Generating dataset, sampler and dataloader for the current train chunk
            
            if(config.n_chunks>1):
            
                dataset_train = get_dataset( config, N_images_in_batch, N, batch_size, train_val_test = 'train', chunk=chunk )
                
                sampler_train = get_sampler( config, dataset_train, N_images_in_batch, N, batch_size )
                
                dataloader_train = DataLoader(  dataset = dataset_train,
                                                sampler = sampler_train,
                                                pin_memory = True,
                                                num_workers = num_workers,
                                                collate_fn = collate_fn2, )
            
            start_time_train = time.perf_counter()
        
            model.train()
                
            for i, data in enumerate(dataloader_train):
                
                ii = i % (2 * batch_count_in_N + 1 )
                
                if(stop_debug == 1 and i>5000 and ii == 0):
                    stop_debug = 0
                
                if( ii < batch_count_in_N ):
                    
                    # To calculate in eval mode batch norm layers must have track_running_stats=False set 
                    
                    # if(ii == 0):
                    #     model.eval()
                    #     inputs_model_B = torch.zeros( N, 1, dtype=torch.float32, device=device )
                    # xs_device = data['xs'].to(device)
                    
                    # with torch.no_grad():
                    #     outputs = model(xs_device)                                    
                    #     inputs_model_B[ batch_size * ii : batch_size * ii + batch_size, : ] = outputs.detach()
                        
                    # if( ii == 0 ):                    
                    #     inputs_model_B = torch.zeros( N, 1, dtype=torch.float32, device=device )
                        
                    xs_device = data['xs'].to(device)
                    outputs = model(xs_device)
                    accumulator_0[ batch_size * ii : batch_size * ii + batch_size, : ] = outputs.detach()
                        
                elif( ii == batch_count_in_N ):
                    
                    xs_ess =  data['xs_ess'].to(device)
                    R_device =  data['R'].to(device)
                    t_device =  data['t'].to(device)
                    virtPt_device =  data['virtPt'].to(device)
                    
                    # accumulator_1.data = accumulator_0.data.clone() 
                    accumulator_1.data = accumulator_0.data
                    
                    # inputs_model_B.requires_grad = True
                    
                    geo_loss, ess_loss, _ = loss_functions.calculate_ess_loss_and_L2loss( config, accumulator_1, xs_ess, R_device, t_device, virtPt_device )
                    
                    if(config.ess_loss == "geo_loss"):
                        geo_loss.backward()
                    else:
                        ess_loss.backward()                  
                        
                else:
                        
                    images_device, labels_device = data['xs'].to(device), data['ys'].to(device)
                    
                    if(en_grad_checkpointing==True):
                        images_device.requires_grad = True
                    
                    logits = model(images_device)
                    
                    classif_loss = loss_functions.get_losses( config, device, labels_device, logits)
                                    
                    feedback_from_network_2 = accumulator_1.grad[ batch_size*(ii-batch_count_in_N-1) : batch_size*(ii-batch_count_in_N-1)+batch_size ]
                    
                    logits_2 = loss_functions.tune_logits_with_ess_loss( config, logits, feedback_from_network_2)
                    
                    ess_loss_feedback = loss_functions.get_losses( config, device, labels_device, logits_2)
                    
                    if( epoch < config.n_epochs[1] or ( epoch == config.n_epochs[1] and chunk < config.n_epochs[2] ) ):
                        loss = classif_loss                
                    else:
                        if(config.ess_loss == 'geo_loss'):
                            loss = 1 * classif_loss + config.geo_loss_ratio * ess_loss_feedback
                        elif(config.ess_loss == 'ess_loss'):
                            loss = 1 * classif_loss + config.ess_loss_ratio * ess_loss_feedback
                    
                    loss.backward()
    
                    if( ii == 2 * batch_count_in_N):
                        optimizer.step()
                        optimizer.zero_grad()
                    
                    confusion_matrix_at_epoch_train_device[0,0] += torch.sum( torch.logical_and( logits<0, labels_device>config.obj_geod_th ) )
                    confusion_matrix_at_epoch_train_device[0,1] += torch.sum( torch.logical_and( logits>0, labels_device>config.obj_geod_th ) )
                    confusion_matrix_at_epoch_train_device[1,0] += torch.sum( torch.logical_and( logits<0, labels_device<config.obj_geod_th ) )
                    confusion_matrix_at_epoch_train_device[1,1] += torch.sum( torch.logical_and( logits>0, labels_device<config.obj_geod_th ) )
                        
                    loss_cls_train = loss_cls_train * loss_count_train + classif_loss.detach().cpu().numpy() * batch_size
                    loss_ess_train = loss_ess_train * loss_count_train + ess_loss.detach().cpu().numpy() * batch_size
                    loss_geo_train = loss_geo_train * loss_count_train + geo_loss.detach().cpu().numpy() * batch_size
                    loss_count_train = loss_count_train + batch_size
                    loss_cls_train = loss_cls_train / loss_count_train
                    loss_ess_train = loss_ess_train / loss_count_train  
                    loss_geo_train = loss_geo_train / loss_count_train
                    
                if( ( (i*batch_size) % 100000 ) > ( ((i+1)*batch_size) % 100000 ) or (i+1) == len(dataloader_train) ):
                    
                    tot_it_train = torch.sum(confusion_matrix_at_epoch_train_device)
                    acc_train = torch.sum(confusion_matrix_at_epoch_train_device[0,0]+confusion_matrix_at_epoch_train_device[1,1]) / tot_it_train * 100
                    pre_train = confusion_matrix_at_epoch_train_device[1,1] / torch.sum(confusion_matrix_at_epoch_train_device[:,1]) * 100
                    rec_train = confusion_matrix_at_epoch_train_device[1,1] / torch.sum(confusion_matrix_at_epoch_train_device[1,:]) * 100
                    f1_train = 2 * pre_train * rec_train / ( pre_train + rec_train )
                            
                    print("Exp {} Train Epoch {}/{} Chunk {}/{} Batch {}/{} LR {:.6f} lCls {:.6f} lGeo {:.6f} LEss {:.6f} CorPred {}/{} Acc {:.6f} Pre {:.6f} Rec {:.6f} F1 {:.6f}"
                            .format(    experiment_no,
                                        epoch,
                                        n_epochs[0]-1,
                                        chunk,
                                        config.n_chunks-1,
                                        i,
                                        len(dataloader_train)-1,
                                        optimizer.param_groups[0]['lr'],
                                        loss_cls_train,
                                        loss_geo_train,
                                        loss_ess_train,
                                        int(torch.sum(confusion_matrix_at_epoch_train_device[0,0]+confusion_matrix_at_epoch_train_device[1,1])),
                                        int(tot_it_train),
                                        acc_train,
                                        pre_train,
                                        rec_train,
                                        f1_train,
                                        ) )          
                    
                    # print(confusion_matrix_at_epoch_train_device)
            
            success_checkpoint[0, epoch, chunk, :] = np.array([acc_train.detach().cpu().numpy(), pre_train.detach().cpu().numpy(), rec_train.detach().cpu().numpy(), f1_train.detach().cpu().numpy()])
            loss_checkpoint[0, epoch, chunk, :] = np.array([loss_cls_train, loss_geo_train, loss_ess_train]) 
                  
            proc_time_checkpoint[0,epoch, chunk] = time.perf_counter() - start_time_train

### Generating dataset, sampler and dataloader for the current validation chunk
            if(config.validation == 'enable'):
            
                if(config.n_chunks>1):

                    dataset_val = get_dataset( config, N_images_in_batch, N, batch_size, train_val_test = 'val', chunk=chunk )
                    
                    sampler_val = get_sampler( config, dataset_val, N_images_in_batch, N, batch_size )
                                
                    dataloader_val = DataLoader(    dataset = dataset_val,
                                                    sampler = sampler_val,
                                                    pin_memory = True,
                                                    num_workers = num_workers,
                                                    collate_fn = collate_fn2, )
                                   
            
                start_time_val = time.perf_counter()
                
                model.eval()  # Sets the model to evaluation mode
                with torch.no_grad():        
                    for i, data in enumerate(dataloader_val):
                        
                        ii = i % (1 * batch_count_in_N + 1 )
                        
                        if( ii < batch_count_in_N ):
                            
                            xs_device = data['xs'].to(device)
                            ys_device = data['ys'].to(device)
                            
                            logits = model(xs_device)
                            
                            classif_loss = loss_functions.get_losses( config, device, ys_device, logits)
                                     
                            confusion_matrix_at_epoch_val_device[0,0] += torch.sum( torch.logical_and( logits<0, ys_device>config.obj_geod_th ) )
                            confusion_matrix_at_epoch_val_device[0,1] += torch.sum( torch.logical_and( logits>0, ys_device>config.obj_geod_th ) )
                            confusion_matrix_at_epoch_val_device[1,0] += torch.sum( torch.logical_and( logits<0, ys_device<config.obj_geod_th ) )
                            confusion_matrix_at_epoch_val_device[1,1] += torch.sum( torch.logical_and( logits>0, ys_device<config.obj_geod_th ) )
                                             
                            loss_cls_val = loss_cls_val * loss_count_val + classif_loss.detach().cpu().numpy() * batch_size
                            loss_count_val = loss_count_val + batch_size
                            loss_cls_val = loss_cls_val / loss_count_val
                            
                            # if( ii == 0 ):                    
                            #     inputs_model_B = torch.zeros( N, 1, dtype=torch.float32, device=device )
                            # inputs_model_B[ batch_size * ii : batch_size * ii + batch_size, : ] = logits.detach()
                            accumulator_0[ batch_size * ii : batch_size * ii + batch_size, : ] = logits.detach()
                            
                        elif( ii == batch_count_in_N ):
                            
                            xs_ess =  data['xs_ess'].to(device)
                            R_device =  data['R'].to(device)
                            t_device =  data['t'].to(device)
                            virtPt_device =  data['virtPt'].to(device)
                            
                            # accumulator_1.data = accumulator_0.data.clone() 
                            accumulator_1.data = accumulator_0.data
                            
                            # inputs_model_B.requires_grad = True
                            
                            geo_loss, ess_loss, _ = loss_functions.calculate_ess_loss_and_L2loss( config, accumulator_1, xs_ess, R_device, t_device, virtPt_device )
                            
                            loss_ess_val = loss_ess_val * (loss_count_val-N) + ess_loss.detach().cpu().numpy() * N
                            loss_geo_val = loss_geo_val * (loss_count_val-N) + geo_loss.detach().cpu().numpy() * N
                            loss_ess_val = loss_ess_val / loss_count_val  
                            loss_geo_val = loss_geo_val / loss_count_val
                                
                        if( ( (i*batch_size) % 100000 ) > ( ((i+1)*batch_size) % 100000 ) or (i+1) == len(dataloader_val) ):
                            
                            tot_it_val = torch.sum(confusion_matrix_at_epoch_val_device)
                            acc_val = torch.sum(confusion_matrix_at_epoch_val_device[0,0]+confusion_matrix_at_epoch_val_device[1,1]) / tot_it_val * 100
                            pre_val = confusion_matrix_at_epoch_val_device[1,1] / torch.sum(confusion_matrix_at_epoch_val_device[:,1]) * 100
                            rec_val = confusion_matrix_at_epoch_val_device[1,1] / torch.sum(confusion_matrix_at_epoch_val_device[1,:]) * 100
                            f1_val = 2 * pre_val * rec_val / ( pre_val + rec_val )
                            
                            print("Exp {} Val Epoch {}/{} Chunk {}/{} Batch {}/{} LR {:.6f} LossCls {:.6f} lGeo {:.6f} LEss {:.6f} CorPred {}/{} Acc {:.6f} Pre {:.6f} Rec {:.6f} F1 {:.6f}"
                                    .format(    experiment_no,
                                                epoch,
                                                n_epochs[0]-1,
                                                chunk,
                                                config.n_chunks-1,
                                                i,
                                                len(dataloader_val)-1,
                                                optimizer.param_groups[0]['lr'],
                                                loss_cls_val,
                                                loss_geo_val,
                                                loss_ess_val,
                                                int(torch.sum(confusion_matrix_at_epoch_val_device[0,0]+confusion_matrix_at_epoch_val_device[1,1])),
                                                int(tot_it_val),
                                                acc_val,
                                                pre_val,
                                                rec_val,
                                                f1_val,
                                                ) )
                success_checkpoint[1, epoch, chunk, :] = np.array([acc_val.detach().cpu().numpy(), pre_val.detach().cpu().numpy(), rec_val.detach().cpu().numpy(), f1_val.detach().cpu().numpy()])
                loss_checkpoint[1, epoch, chunk, :] = np.array([loss_cls_val, loss_geo_val, loss_ess_val])
                proc_time_checkpoint[1,epoch, chunk] = time.perf_counter() - start_time_val
            
            plots.plot_success_and_loss( config, epoch, chunk, success_checkpoint, loss_checkpoint)
            
            plots.plot_proc_time( config, epoch, chunk, proc_time_checkpoint)
            
            checkpoint.save_checkpoint( config, epoch, chunk, model, optimizer, success_checkpoint, loss_checkpoint, proc_time_checkpoint)
            
            if(chunk==config.n_chunks-1):
                chunk = 0
                epoch = epoch + 1
                break
            else:
                chunk = chunk + 1
            
            print("-" * 40)
            
        if(epoch == config.early_finish_epoch):
            break
            
    return 0
