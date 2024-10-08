import torch
from torch.utils.data import DataLoader
import numpy as np
import time

from datasets.datasets import get_dataset
from datasets.datasets import collate_fn2

from models.models import get_model
from models.models import get_model_structure

from samplers.CustomBatchSampler import get_sampler

from loss_module import loss_functions_n_to_n
from mAP import mAP
import checkpoint
import plots

def test( 
            config,
            experiment_no,
            learning_rate,
            n_epochs,
            num_workers,
            model_type,
            en_grad_checkpointing,
            N_images_in_batch,
            N,
            batch_size, ):
    
    device = config.device
    
    model_width = config.model_width
    model = get_model( config, model_type, N, model_width, en_grad_checkpointing ).to(device)
    
    get_model_structure( config, device, model, N, model_width, en_grad_checkpointing)
    
    checkpoint_files = checkpoint.get_all_checkpoint_files_for_test(config)
    
    success_checkpoint = np.zeros( (1, len(checkpoint_files), config.n_chunks, 4) )
    loss_checkpoint = np.zeros( (1, len(checkpoint_files), config.n_chunks, 3) )
    proc_time_checkpoint = np.zeros( (1, len(checkpoint_files), config.n_chunks) )
    mAP_checkpoint = np.zeros( (1, len(checkpoint_files), config.n_chunks, 3, 360) )
    
    mAP_exact_checkpoint = np.zeros( (1, len(checkpoint_files), config.n_chunks, 3, 3) )
    
    if(config.n_chunks == 1):
        dataset_test = get_dataset( config, N_images_in_batch, N, batch_size, train_val_test = 'test', chunk=0 )
        
        sampler_test = get_sampler( config, dataset_test, N_images_in_batch, N, batch_size )
            
        dataloader_test = DataLoader(   dataset = dataset_test,
                                        sampler = sampler_test,
                                        pin_memory = True,
                                        num_workers = num_workers,
                                        collate_fn=collate_fn2,)
    
    for epoch in range( len(checkpoint_files) ):
        
        model = checkpoint.load_test_checkpoint( config, device, model, checkpoint_file_with_path=checkpoint_files[epoch] )
    
        for chunk in range(0, config.n_chunks):
        
            loss_cls_test = 0
            loss_geo_test = 0
            loss_ess_test = 0
            loss_count_test = 0
            
            err_q_list = []
            err_t_list = []
            err_qt_list = []
            
            confusion_matrix_at_epoch_test_device  = torch.zeros( (2,2), device = device, requires_grad = False )
            
### Generating dataset, sampler and dataloader for the current test chunk
            
            if(config.n_chunks > 1):
                dataset_test = get_dataset( config, N_images_in_batch, N, batch_size, train_val_test = 'test', chunk=chunk )
                
                sampler_test = get_sampler( config, dataset_test, N_images_in_batch, N, batch_size )
                    
                dataloader_test = DataLoader(   dataset = dataset_test,
                                                sampler = sampler_test,
                                                pin_memory = True,
                                                num_workers = num_workers,
                                                collate_fn=collate_fn2,)
                
            start_time_test = time.perf_counter()       
                
            model.eval()  # Sets the model to evaluation mode
            with torch.no_grad():        
                for i, data in enumerate(dataloader_test):
                    
                    xs_device = data['xs'].to(device)
                    labels_device = data['ys'].to(device)
                    
                    xs_ess =  data['xs_ess'].to(device)
                    R_device =  data['R'].to(device)
                    t_device =  data['t'].to(device)
                    virtPt_device =  data['virtPt'].to(device) 
                    
                    logits = model(xs_device)
                    
                    classif_loss = loss_functions_n_to_n.get_losses( config, device, labels_device, logits[-1])
                    
                    geo_loss, ess_loss, e_hat = loss_functions_n_to_n.calculate_ess_loss_and_L2loss( config, logits[-1], xs_ess, R_device, t_device, virtPt_device )
                    
                    confusion_matrix_at_epoch_test_device[0,0] += torch.sum( torch.logical_and( logits[-1]<0, labels_device.squeeze()>config.obj_geod_th ) )
                    confusion_matrix_at_epoch_test_device[0,1] += torch.sum( torch.logical_and( logits[-1]>0, labels_device.squeeze()>config.obj_geod_th ) )
                    confusion_matrix_at_epoch_test_device[1,0] += torch.sum( torch.logical_and( logits[-1]<0, labels_device.squeeze()<config.obj_geod_th ) )
                    confusion_matrix_at_epoch_test_device[1,1] += torch.sum( torch.logical_and( logits[-1]>0, labels_device.squeeze()<config.obj_geod_th ) )
                                         
                    loss_cls_test = loss_cls_test * loss_count_test + classif_loss.detach().cpu().numpy() * batch_size
                    loss_ess_test = loss_ess_test * loss_count_test + ess_loss.detach().cpu().numpy() * batch_size
                    loss_geo_test = loss_geo_test * loss_count_test + geo_loss.detach().cpu().numpy() * batch_size
                    loss_count_test = loss_count_test + batch_size
                    loss_cls_test = loss_cls_test / loss_count_test
                    loss_ess_test = loss_ess_test / loss_count_test  
                    loss_geo_test = loss_geo_test / loss_count_test
                    
                    for b in range( xs_ess.shape[0]):
                        
                        xs_ess_b = data['xs_ess'][b]
                        R_b = data['R'][b]
                        t_b = data['t'][b]
                        E_hat_b = e_hat[b].unsqueeze(dim=0)
                        y_hat_b = logits[-1].detach()[b].unsqueeze(dim=-1)
                    
                        err_q, err_t, err_qt = mAP.calculate_err_q_err_t( config=config, xs_ess=xs_ess_b, R=R_b, t=t_b, E_hat=E_hat_b, y_hat=y_hat_b )
                        mAP_checkpoint = checkpoint.update_mAP_checkpoint( mAP_checkpoint, err_q, err_t, err_qt, epoch=epoch, chunk=chunk )
                        
                        err_q_list.append(err_q)
                        err_t_list.append(err_t)
                        err_qt_list.append(err_qt)
                            
                    if( ( (i*batch_size) % 1000 ) > ( ((i+1)*batch_size) % 1000 ) or (i+1) == len(dataloader_test) ):
                        
                        tot_it_test = torch.sum(confusion_matrix_at_epoch_test_device)
                        acc_test = torch.sum(confusion_matrix_at_epoch_test_device[0,0]+confusion_matrix_at_epoch_test_device[1,1]) / tot_it_test * 100
                        pre_test = confusion_matrix_at_epoch_test_device[1,1] / torch.sum(confusion_matrix_at_epoch_test_device[:,1]) * 100
                        rec_test = confusion_matrix_at_epoch_test_device[1,1] / torch.sum(confusion_matrix_at_epoch_test_device[1,:]) * 100
                        f1_test = 2 * pre_test * rec_test / ( pre_test + rec_test )
                            
                        print("Exp {} Test Epoch {}/{} Chunk {}/{} Batch {}/{} LR {:.6f} LossCls {:.6f} lGeo {:.6f} LEss {:.6f} CorPred {}/{} Acc {:.6f} Pre {:.6f} Rec {:.6f} F1 {:.6f}"
                                .format(    experiment_no,
                                            epoch,
                                            len(checkpoint_files)-1,
                                            chunk,
                                            config.n_chunks-1,
                                            i,
                                            len(dataloader_test)-1,
                                            learning_rate,
                                            loss_cls_test,
                                            loss_geo_test,
                                            loss_ess_test,
                                            int(torch.sum(confusion_matrix_at_epoch_test_device[0,0]+confusion_matrix_at_epoch_test_device[1,1])),
                                            int(tot_it_test),
                                            acc_test,
                                            pre_test,
                                            rec_test,
                                            f1_test,
                                            ) )

            success_checkpoint[0, epoch, chunk, :] = np.array([acc_test.detach().cpu().numpy(), pre_test.detach().cpu().numpy(), rec_test.detach().cpu().numpy(), f1_test.detach().cpu().numpy()])
            loss_checkpoint[0, epoch, chunk, :] = np.array([loss_cls_test, loss_geo_test, loss_ess_test])
            proc_time_checkpoint[0, epoch, chunk] = time.perf_counter() - start_time_test
            
            mAP_exact_checkpoint = checkpoint.update_mAP_exact_checkpoint( mAP_exact_checkpoint, err_q_list, err_t_list, err_qt_list, epoch=epoch, chunk=chunk ) 
            
            print("-" * 40)
        
    plots.plot_success_and_loss( config, epoch, config.n_chunks-1, success_checkpoint, loss_checkpoint)
    
    plots.plot_mAP( config, epoch, config.n_chunks-1, mAP_checkpoint, ref_angles = [5, 10, 20])
    
    plots.plot_mAP_exact( config, epoch, config.n_chunks-1, mAP_exact_checkpoint, ref_angles = [5, 10, 20])
    
    plots.plot_proc_time( config, epoch, config.n_chunks-1, proc_time_checkpoint)       
    
    checkpoint.save_test_checkpoint( config, success_checkpoint, loss_checkpoint, proc_time_checkpoint, mAP_checkpoint, mAP_exact_checkpoint)
            
    return 0
