import torch
import numpy as np
import h5py
from torch.utils.data import DataLoader, TensorDataset
from pdb import set_trace
import pickle
def load_data_post(data_dir, batch_size):
    """Return data loader

    Args:
        data_dir: directory to hdf5 file, e.g. `dir/to/kle4225_lhs256.hdf5`
        batch_size (int): mini-batch size for loading data

    Returns:
        (data_loader (torch.utils.data.DataLoader), stats)
    """

    # with h5py.File(data_dir, 'r') as f:
    #     x_data = f['input'][()]
    #     y_data = f['output'][()]
        
    # # set_trace()    
    # data_dir = r"C:\Users\Gloria\Documents\Masterarbeit\Solving_PDE_NeuralNetwork\Data_rfields_new_model\Data_for_Training_trainingsdata_800_samples_01.pickle"
    # set_trace()
    with open(data_dir,'rb') as g:
        data_for_nn = pickle.load(g)
       
    # set_trace()
    
    x_data = np.float32(np.vstack(data_for_nn['rfields']).reshape(-1,1,20,20))
    y_data = np.float32(np.vstack(np.array(data_for_nn['sigma_33']))).reshape(-1,20,20)
   
    sig_11_data = np.float32(np.vstack(np.array(data_for_nn['sigma_11']))).reshape(-1,20,20)
    sig_12_data = np.float32(np.vstack(np.array(data_for_nn['sigma_12']))).reshape(-1,20,20)
    sig_22_data = np.float32(np.vstack(np.array(data_for_nn['sigma_22']))).reshape(-1,20,20)
    sig_13_data = np.float32(np.vstack(np.array(data_for_nn['sigma_13']))).reshape(-1,20,20)
    sig_23_data = np.float32(np.vstack(np.array(data_for_nn['sigma_23']))).reshape(-1,20,20)
    # set_trace()
    y_data_6_channels = np.stack((sig_11_data, sig_12_data, sig_13_data, sig_22_data, sig_23_data, y_data), axis=1)
    
    print("input data shape: {}".format(x_data.shape))
    print("output data shape: {}".format(y_data_6_channels.shape))

    kwargs = {'num_workers': 20,
              'pin_memory': True} if torch.cuda.is_available() else {}

    dataset = TensorDataset(torch.tensor(x_data), torch.tensor(y_data_6_channels))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, **kwargs)

    
    # simple statistics of output data
    y_data_mean = np.mean( y_data_6_channels, 0)
    y_data_var =   np.var( y_data_6_channels, 0)
    # y_data_var = 1/len(y_data) * np.sum((y_data - y_data_mean) ** 2,0)
    stats = {}
    stats['y_mean'] = y_data_mean
    stats['y_var'] = y_data_var

    return data_loader, stats
