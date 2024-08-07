import os
import numpy as np
import h5py
import pickle

class ConvertHDF5toPickle:
    def __init__(self):
        
        self.folder_name = 'D:\\01_featureMatchingDatasets'
        
        self.file_name_train = os.path.join(self.folder_name, 'yfcc-sift-2000-train.hdf5')
        self.file_name_val = os.path.join(self.folder_name, 'yfcc-sift-2000-val.hdf5')
        self.file_name_test = os.path.join(self.folder_name, 'yfcc-sift-2000-test.hdf5')
        
        self.parent_folder_name_pickle = 'D:\\01_featureMatchingDatasets'
        self.pickle_set_no = 1
        self.folder_name_pickle = os.path.join(self.parent_folder_name_pickle, str(self.pickle_set_no))
        
        self.num_workers = 3
        # self.num_workers = 0
        
        self.system_ram_mb = 2_000
        self.n_image_pairs_train = self.get_total_data_length( 'train' )
        self.s_each_image_pair_apprx_mb = 0.05  
        self.s_t_image_pair_apprx_mb = self.n_image_pairs_train * self.s_each_image_pair_apprx_mb * ( self.num_workers + 1 )
        self.n_chunks = int( self.s_t_image_pair_apprx_mb / self.system_ram_mb ) + 1
        
        self.n_image_pairs_val = self.get_total_data_length( 'val' )
        self.n_image_pairs_test = self.get_total_data_length( 'test' )
        
        print( 'Number of chunks ' + str( self.n_chunks ) )
        
    def get_total_data_length(self, train_val_test):
        
        if(train_val_test=='train'):
            self.data = h5py.File( self.file_name_train, 'r', libver='latest', swmr=True )
        elif(train_val_test=='val'):
            self.data = h5py.File( self.file_name_val, 'r', libver='latest', swmr=True )
        elif(train_val_test=='test'):
            self.data = h5py.File( self.file_name_test, 'r', libver='latest', swmr=True )
                
        self.size_dataset = len( self.data['xs'] )
        print( 'Number of image pairs in ' + train_val_test + ' dataset ' + str( self.size_dataset ) )
        self.data.close()
        
        return self.size_dataset
    
    def convert_hdf5_to_pickle(self):        
        
        train_val_test_vec = ['train', 'val', 'test']        
        for train_val_test in train_val_test_vec:
            
            if(train_val_test=='train'):
                n_image_pairs = self.n_image_pairs_train
                data = h5py.File( self.file_name_train, 'r', libver='latest', swmr=True )
            elif(train_val_test=='val'):
                n_image_pairs = self.n_image_pairs_val
                data = h5py.File( self.file_name_val, 'r', libver='latest', swmr=True )
            elif(train_val_test=='test'):
                n_image_pairs = self.n_image_pairs_test
                data = h5py.File( self.file_name_test, 'r', libver='latest', swmr=True )
            
            for chunk in range(self.n_chunks):
                
                start = chunk
                end = n_image_pairs
                indices = list(range(start, end, self.n_chunks))
                    
                xs_chunk = [ np.asarray( data['xs'][str(start)] ) ]
                ys_chunk = [ np.asarray( data['ys'][str(start)] ) ]
                ratios_chunk = [ np.asarray( data['ratios'][str(start)] ) ]
                mutuals_chunk = [ np.asarray( data['mutuals'][str(start)] ) ]
                R_chunk = [ np.asarray( data['Rs'][str(start)] ) ]
                t_chunk = [ np.asarray( data['ts'][str(start)] ) ]
                
                xs_chunk = [ xs_chunk[0].copy() for _ in range(len(indices)) ]
                ys_chunk = [ ys_chunk[0].copy() for _ in range(len(indices)) ]
                ratios_chunk = [ ratios_chunk[0].copy() for _ in range(len(indices)) ]
                mutuals_chunk = [ mutuals_chunk[0].copy() for _ in range(len(indices)) ]
                R_chunk = [ R_chunk[0].copy() for _ in range(len(indices)) ]
                t_chunk = [ t_chunk[0].copy() for _ in range(len(indices)) ]
            
                i = 0
                for ind in indices:
                    
                    xs_chunk[i] = np.asarray( data['xs'][str(ind)] )
                    ys_chunk[i] = np.asarray( data['ys'][str(ind)] )
                    ratios_chunk[i] = np.asarray( data['ratios'][str(ind)] )
                    mutuals_chunk[i] = np.asarray( data['mutuals'][str(ind)] )
                    R_chunk[i] = np.asarray( data['Rs'][str(ind)] )
                    t_chunk[i] = np.asarray( data['ts'][str(ind)] )
                    
                    i = i + 1
                    
                    print( str(i) + ' ' + str(ind) )
                    
                data_pickle = [xs_chunk, ys_chunk, ratios_chunk, mutuals_chunk, R_chunk, t_chunk]
                
                if(not os.path.isdir(self.folder_name_pickle)):
                    os.makedirs(self.folder_name_pickle)
                file_name_with_path = os.path.join(self.folder_name_pickle, train_val_test + f'_{chunk:04d}' + '.pkl')
                with open(file_name_with_path, 'wb') as file:
                    pickle.dump(data_pickle, file)        
    
        
if __name__ == '__main__':
    
    convertHDF5toPickle = ConvertHDF5toPickle()
    convertHDF5toPickle.convert_hdf5_to_pickle()
        
