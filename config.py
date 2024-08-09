import os
import shutil
import h5py

class Config:
    def __init__(self):
        
        self.operation = 'train'
        self.operation = 'test'
        
        self.device = 'cpu'
        self.device = 'cuda'
        # self.device = 'tpu'
        
        self.tpu_cores = 'single'
        self.tpu_cores = 'multi'
        # self.tpu_cores = 'spmd'
        
        self.validation = 'disable'
        self.validation = 'enable'
        
        self.tpu_profiler = 'disable'
        # self.tpu_profiler = 'enable'
        
        if(self.device=='tpu'):
            
            self.XLA_USE_BF16 = 0
            # self.XLA_USE_BF16 = 1
            
            self.home_dir = os.path.expanduser('~')
            
            self.input_data_storage_local_or_bucket = 'bucket'
            
            self.tpu_version = 'v2'
            self.tpu_version = 'v3'
            self.tpu_version = 'v4'
            self.tpu_version = 'v4pre'
            
            if( self.tpu_version == 'v2' ):
                self.bucket_name = 'bucket-us-central1-relativeposeestimation'  # v2
                self.n_tpu_cores = 8
                self.output_data_storage_local_or_bucket = 'local'
            elif( self.tpu_version == 'v3' ):
                self.bucket_name = 'bucket-europe-west4-relativeposeestimation' # v3
                self.n_tpu_cores = 8
                self.output_data_storage_local_or_bucket = 'local'
            elif( self.tpu_version == 'v4' or self.tpu_version == 'v4pre' ):
                self.bucket_name = 'bucket-us-central2-relativeposeestimation'  # v4
                self.n_tpu_cores = 4
                if( self.tpu_version == 'v4' ):
                    self.output_data_storage_local_or_bucket = 'local'
                elif( self.tpu_version == 'v4pre' ):
                    self.output_data_storage_local_or_bucket = 'bucket'
            
            self.TPU_DEBUG = 0
            # self.TPU_DEBUG = 1
            self.tpu_debug_path = os.path.join(self.home_dir, 'tpu_debug')                
        else:
            self.input_data_storage_local_or_bucket = 'local'
            self.output_data_storage_local_or_bucket = 'local'
        
        self.first_experiment = 100
        
        # self.model_type = 'CNN_Plain'
        # self.model_type = 'CNN_Residual'
        # self.model_type = 'CNN_Residual_Context'        
        # self.model_type = 'MobileNetV1'
        # self.model_type = 'MobileNetV2'
        # self.model_type = 'MobileNetV3'
        
        # self.model_type = 'model_exp'
        # self.model_exp_no = 0
        
        # self.model_type = 'model_exp2'
        # self.model_exp_no = 0
        
        # self.model_type = 'model_exp3'
        # self.model_exp_no = 1
        
        # self.model_type = 'model_exp4'
        # self.model_exp_no = 21
        
        # self.model_type = 'model_exp5'
        # self.model_exp_no = 1
        
        # self.model_type = 'model_exp6'
        # self.model_exp_no = 1
        
        # self.model_type = 'model_exp7'
        # self.model_exp_no = 1

        # self.model_type = 'model_exp8'
        # self.model_exp_no = 1
        
        # self.model_type = 'model_exp9'
        # self.model_exp_no = 1
        
        # self.model_type = 'model_exp10'
        # self.model_exp_no = 1
        
        # self.model_type = 'model_exp11'
        # self.model_exp_no = 1
        
        # self.model_type = 'model_exp12'
        # self.model_exp_no = 0
        
        # self.model_type = 'model_exp13'
        # self.model_exp_no = 0
        
        # self.model_type = 'model_exp14'
        # self.model_exp_no = 0
        
        # self.model_type = 'model_exp15'
        # self.model_exp_no = 0
        
        # self.model_type = 'model_exp20'
        # self.model_exp_no = 4  #0,1,3,4,10,11,13,14,20,21,23,24   
        
        # self.model_type = 'model_exp21'
        # self.model_exp_no = 0  # 0, 10,
        
        # self.model_type = 'model_exp22'
        # self.model_exp_no = 4  #0,1,3,4,10,11,13,14,20,21,23,24
        
        # self.model_type = 'model_exp23'
        # self.model_exp_no = 324  #0,1,3,4,10,11,13,14,20,21,23,24
        
        self.model_type = 'LTFGC'
        # self.model_type = 'OANET'
        # self.model_type = 'OANET_Iter'
        
        # self.model_type = 'convmatch'
        # self.model_type = 'convmatch_plus'
        # self.model_type = 'CLNet'
        # self.model_type = 'MS2DGNET'
        
        if( self.model_type == 'LTFGC' or self.model_type == 'OANET' or self.model_type == 'OANET_Iter' or
            self.model_type == 'convmatch' or self.model_type == 'convmatch_plus' or self.model_type == 'CLNet' or self.model_type == 'MS2DGNET' ):
            self.input_type = 'n_to_n'
            
            # self.data_ordering_type = 0 # no random selection, no ordering
            self.data_ordering_type = 1 # random selection of repeated elements, no ordering
        else:
            self.input_type = '1_to_1'
            if(self.model_type != 'model_exp20' and self.model_type != 'model_exp21' and self.model_type != 'model_exp22' ):
                self.en_batch_build_with_context = 1    # No context information is needed from dataset when LTFGC type models are used as preprocessing
            else:
                self.en_batch_build_with_context = 0    # No context information is needed from dataset when LTFGC type models are used as preprocessing

            if( self.model_type == 'model_exp23' ):
                self.en_tl_on_cpu = 1
            else:
                self.en_tl_on_cpu = 0
                
            if( self.model_type == 'model_exp15' ):
                self.context_type_for_all_candidates_of_sample = 1 # Closer context together  wrt euclidean distance 
            else:
                self.context_type_for_all_candidates_of_sample = 0
            
            if( self.model_type == 'model_exp14' ):
                self.data_ordering_type = 2 # random selection of repeated elements, ordering wrt respect to magnitude of disparity
            elif( self.model_type == 'model_exp13' or self.model_type == 'model_exp15' ):
                self.data_ordering_type = 1 # random selection of repeated elements, no ordering
            else:
                self.data_ordering_type = 0 # no random selection, no ordering
        
        self.use_ratio = 0  # 0-> don't use, 1-> mask xs and ys, 2-> use as side
        self.use_mutual = 0  # 0-> don't use, 1-> mask xs and ys, 2-> use as side
        if(self.use_ratio==0 and self.use_mutual == 0):
            self.model_width = 4
        else:
            self.model_width = 6
        
        self.ess_loss = 'geo_loss'
        # self.ess_loss = 'ess_loss'
        
        # training_params ->   [model_type(n_to_n, 1_to_1), N_images_in_batch, N, batch_size]
        
        if( self.input_type == '1_to_1'):
            # self.training_params = [ [ 1, 512, 64, ],  ]        
            # self.training_params = [ [ 1, 512, 512, ],  ]
            # self.training_params = [ [ 1, 1024, 1024, ],  ]
            self.training_params = [ [ 1, 2048, 2048, ],  ]
            
            # self.n_epochs = [500, 500, 0] # always cls loss
            self.n_epochs = [500, 0, 1] # only first chunk cls loss
            
            self.early_finish_epoch = 100
        else:
            if(self.operation == 'train'):
                # self.training_params = [ [ 32, 512, 32, ],  ]
                # self.training_params = [ [ 32, 1024, 32, ],  ]
                self.training_params = [ [ 32, 2048, 32, ],  ]
            elif(self.operation == 'test'):
                # self.training_params = [ [ 25, 512, 25, ],  ]
                # self.training_params = [ [ 25, 1024, 25, ],  ]
                self.training_params = [ [ 25, 2048, 25, ],  ]
            
            # self.n_epochs = [5000, 5000, 0] # always cls loss
            self.n_epochs = [5000, 0, 1] # only first chunk cls loss
            
            self.early_finish_epoch = 100
            
        if( self.input_type == 'n_to_n'):
            self.input_channel_count = 1
        else:
            self.input_channel_count = 2
            
        if(self.tpu_cores == 'spmd'):
            if(self.model_type == 'model_exp20' or self.model_type == 'model_exp21' or self.model_type == 'model_exp22'):
                self.spmd_type = 'model'
            else:
                self.spmd_type = 'batch'
        
        self.use_hdf5_or_picle = 'hdf5'
        self.use_hdf5_or_picle = 'pickle'
        
        self.file_name_train = 'yfcc-sift-2000-train.hdf5'
        self.file_name_val = 'yfcc-sift-2000-val.hdf5'
        self.file_name_test = 'yfcc-sift-2000-test.hdf5'
        
        if( self.device == 'tpu' ):
            os.chdir( os.path.join(self.home_dir, 'one-to-one-framework-frcpe') )
        
        self.input_path_bucket = '01_datasets' 
        self.input_path_local = os.path.join('..', self.input_path_bucket)                
        
        self.output_folder_name = '08_outputs'
        if(self.output_data_storage_local_or_bucket == 'bucket'):
            self.output_path_bucket = os.path.join( 'gs://' + self.bucket_name, self.output_folder_name ) 
        self.output_path_local = os.path.join('..', self.output_folder_name)  
        
        if(self.use_hdf5_or_picle == 'hdf5'):
                        
            if( self.device != 'tpu' ):
            
                self.num_workers = 3
                
                self.n_image_pairs_train = 541172
                self.n_image_pairs_val = 6694
                self.n_image_pairs_test = 4000
                
                system_ram_mb = 800_000
                # system_ram_mb = 10
                # system_ram_mb = 50
                # system_ram_mb = 100
                s_each_image_pair_apprx_mb = 0.05  
                s_t_image_pair_apprx_mb = self.n_image_pairs_train * s_each_image_pair_apprx_mb * ( self.num_workers + 1 )
                self.n_chunks = int( s_t_image_pair_apprx_mb / system_ram_mb ) + 1
                self.n_chunk_files = 1
            else:
                raise NotImplementedError(f"The feature '{self.device}' is not implemented yet.")
            
        elif(self.use_hdf5_or_picle == 'pickle'):
            
            if( self.device != 'tpu' ):
                if(self.operation == 'train'):
                    self.num_workers = 3                
                    self.pickle_set_no = 1
                    self.input_path_pickle_local = os.path.join( self.input_path_local, str(self.pickle_set_no) )
                    self.input_path_pickle_bucket = os.path.join( self.input_path_bucket, str(self.pickle_set_no) )
                    self.n_chunks = self.get_n_chunks_from_files()
                    self.n_chunk_files = self.n_chunks
                elif(self.operation == 'test'):
                    self.num_workers = 3                
                    self.pickle_set_no = 0
                    self.input_path_pickle_local = os.path.join( self.input_path_local, str(self.pickle_set_no) )
                    self.input_path_pickle_bucket = os.path.join( self.input_path_bucket, str(self.pickle_set_no) )
                    self.n_chunks = self.get_n_chunks_from_files() # Must be 1 for tests
                    self.n_chunk_files = self.n_chunks # Must be 1 for tests            
            else:                
                if( self.tpu_cores == 'single' or self.tpu_cores == 'spmd' ):
                    if(self.operation == 'train'):
                        if( self.input_type == 'n_to_n'):
                            self.pickle_set_no = 0
                            self.num_workers = 2
                        elif( self.input_type == '1_to_1'):
                            if( self.tpu_cores == 'single' ):
                                self.pickle_set_no = 2
                                self.num_workers = 1
                            elif( self.tpu_cores == 'spmd' ):
                                if(self.tpu_version == 'v4pre'):
                                    self.pickle_set_no = 3
                                else:
                                    self.pickle_set_no = 2
                                if( self.en_tl_on_cpu == 0):
                                    self.num_workers = 1
                                else:
                                    self.num_workers = 8          ###########################################################                      
                        self.input_path_pickle_local = os.path.join( self.input_path_local, str(self.pickle_set_no) )
                        self.input_path_pickle_bucket = os.path.join( self.input_path_bucket, str(self.pickle_set_no) )
                        self.n_chunk_files = self.get_n_chunks_from_files()
                        self.n_chunks = self.n_chunk_files
                    elif(self.operation == 'test'):
                        if( self.input_type == 'n_to_n'):
                            self.pickle_set_no = 0
                            self.num_workers = 7
                        elif( self.input_type == '1_to_1'):
                            self.pickle_set_no = 0
                            self.num_workers = 7
                        self.input_path_pickle_local = os.path.join( self.input_path_local, str(self.pickle_set_no) )
                        self.input_path_pickle_bucket = os.path.join( self.input_path_bucket, str(self.pickle_set_no) )
                        self.n_chunk_files = self.get_n_chunks_from_files()
                        self.n_chunks = self.n_chunk_files
                elif( self.tpu_cores == 'multi' ):
                    if( self.input_type == 'n_to_n'):
                        if( self.tpu_version == 'v4' or self.tpu_version == 'v4pre' ):                        
                            self.pickle_set_no = 1
                        else:
                            self.pickle_set_no = 2
                        self.num_workers = 0
                    elif( self.input_type == '1_to_1'):
                        if( self.tpu_version == 'v4' or self.tpu_version == 'v4pre' ):                        
                            self.pickle_set_no = 1
                        else:
                            self.pickle_set_no = 2
                        
                        if(self.context_type_for_all_candidates_of_sample == 0):
                            self.num_workers = 0
                        else:
                            self.num_workers = 3
                    self.input_path_pickle_local = os.path.join( self.input_path_local, str(self.pickle_set_no) )
                    self.input_path_pickle_bucket = os.path.join( self.input_path_bucket, str(self.pickle_set_no) )
                    self.n_chunk_files = self.get_n_chunks_from_files()
                    if( int(self.n_chunk_files) % int(self.n_tpu_cores) == 0 ):                        
                        self.n_chunks = int( self.n_chunk_files / self.n_tpu_cores )
                    else:
                        raise ValueError(f"The provided argument is not valid: {self.n_chunk_files}")
            
        print( f'Number of chunks {self.n_chunks}, Number of chunk files {self.n_chunk_files}' )
        
        if( self.input_type == 'n_to_n'):
            self.learning_rate = 0.0001 # presented in LTFGC paper
            # self.learning_rate = 0.001 # CLNET
            # self.learning_rate = 0.001 # MS2DGNET
            self.optimizer_type = 'ADAM'
        else:
            self.learning_rate = 0.01
            self.momentum = 0.9
            self.optimizer_type = 'SGD'
            
            # self.learning_rate = 0.001
            #### self.learning_rate = 0.001
            # self.optimizer_type = 'ADAM'
            
            # self.learning_rate = 0.0001
            # #### self.learning_rate = 0.001
            # self.optimizer_type = 'ADAMW'
        
        self.geo_loss_ratio = 0.5
        self.ess_loss_ratio = 0.1

        self.ratio_test_th = 0.8
        
        self.obj_geod_th = 1e-4
        
        self.geo_loss_margin = 0.1
        
        if(self.device == 'tpu'):
            
            self.validation_chunk_or_all = 'all'
            
            # if(self.tpu_version == 'v4pre'):
            #     self.validation_chunk_or_all = 'chunk'
            # else:
            #     self.validation_chunk_or_all = 'all'
        else:
            self.validation_chunk_or_all = 'all'
        
        self.save_checkpoint_last_or_all = 'last'
        self.save_checkpoint_last_or_all = 'all'
        
        self.en_grad_checkpointing = False
        # self.en_grad_checkpointing = True
    
    def get_n_chunks_from_files(self):
        
        chunk = 0
        if(self.input_data_storage_local_or_bucket == 'local'):
            while True:            
                file_name_with_path = os.path.join(self.input_path_pickle_local, 'train' + f'_{chunk:04d}' + '.pkl')            
                if os.path.isfile(file_name_with_path):
                    chunk += 1
                else:
                    return chunk

        elif(self.input_data_storage_local_or_bucket == 'bucket'):                
            from google.cloud import storage
            
            storage_client = storage.Client()    
            bucket = storage_client.get_bucket(self.bucket_name)
            blobs = bucket.list_blobs(prefix=self.input_path_pickle_bucket)

            blob_names = []
            for blob in blobs:
                blob_names.append(blob.name)

            while True:
                file_name = os.path.join(self.input_path_pickle_bucket, 'train' + f'_{chunk:04d}' + '.pkl')
                if(file_name in blob_names):
                    chunk += 1
                else:
                    return chunk
                
    def copy_config_file_to_output_folder(self, experiment):    
        
        folder_name = os.path.join(self.output_path_local, f'{experiment:04d}')        
        if(not os.path.isdir(folder_name)):
            os.makedirs(folder_name)        
        destination = os.path.join(folder_name, 'config.py')
        shutil.copyfile('config.py', destination)
        
        self.update_output_folder(experiment)
    
    def update_output_folder(self, experiment):
    
        self.output_path_local = os.path.join(self.output_path_local, f'{experiment:04d}')
        print(self.output_path_local)
        
        if(self.output_data_storage_local_or_bucket == 'bucket'):
            self.output_path_bucket = os.path.join(self.output_path_bucket, f'{experiment:04d}')
            print(self.output_path_bucket)
            
    def start_multi_loss_training_from_start(self):        
        if( ( (self.model_type == 'model_exp20' or self.model_type == 'model_exp22' ) and self.model_exp_no >=200) or
            ( self.model_type == 'model_exp23' ) ):
            self.n_epochs[-1] = 0 # if tl is active do not train only cls
        
    def copy_output_folder_from_bucket_to_local(self):
        import subprocess
        command = 'gsutil -m rsync -r ' + self.output_path_bucket + ' ' + self.output_path_local
        
        subprocess.run(command, shell=True, capture_output=True, text=True)        
        print( f'{self.output_path_local} is being synced with data from {self.output_path_bucket}')

    def sync_local_output_folder_with_bucket_output_folder(self):
        import subprocess
        command = 'gsutil -m rsync -r ' + self.output_path_local + ' ' + self.output_path_bucket
        
        subprocess.run(command, shell=True, capture_output=True, text=True)        
        print( f'{self.output_path_bucket} is being synced with data from {self.output_path_local}')
    
    def update_training_params_for_test(self):
        
        for training_params in self.training_params:
            training_params[2] = training_params[1]

def get_config():
    return Config()
        
if __name__ == '__main__':
    
    config = Config()
