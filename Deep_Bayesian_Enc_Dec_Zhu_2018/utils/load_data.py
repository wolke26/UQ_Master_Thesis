import torch
import numpy as np
import h5py
from torch.utils.data import DataLoader, TensorDataset
from pdb import set_trace
import pickle


def normalize_between(values, min_values, max_values):
    a = 0.00
    b = 1.00
    
    # min_values = 100.00
    # max_values = 300.00
    normalized_values = (b - a) * (values - min_values) / (max_values - min_values) + a
    return(normalized_values)
def load_data(data_dir, batch_size):
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
    # data_dir = r"C:\Users\Gloria\Documents\Masterarbeit\Output_nH10000\Output\Trainings_Data\Training_data_4200.pickle"
        # data_dir = r"C:\Users\Gloria\Documents\Masterarbeit\Output_nH10000\Output\Trainings_Data\Unseen_data_5000.pickle"
        # data_dir = r"C:\Users\Gloria\Documents\Masterarbeit\Output_nH10000\Output\Trainings_Data\Test_data_800.pickle"

    # set_trace()
    with open(data_dir,'rb') as g:
        data_for_nn = pickle.load(g)
       
    # set_trace()
    
    x_data = np.float32(np.vstack(data_for_nn['rfields'])).reshape(-1,1,20,20)
    y_data = np.float32(normalize_between(np.vstack(np.array(data_for_nn['sigma_33'])),data_for_nn['min_sigma_33'][0][0], \
                                          data_for_nn['min_sigma_33'][0][1])).reshape(-1,1,20,20)
   
    
    print("input data shape: {}".format(x_data.shape))
    print("output data shape: {}".format(y_data.shape))

    kwargs = {'num_workers': 20,
              'pin_memory': True} if torch.cuda.is_available() else {}

    dataset = TensorDataset(torch.tensor(x_data), torch.tensor(y_data))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, **kwargs)

    # simple statistics of output data
    y_data_mean = np.mean(y_data, 0)
    y_data_var =  np.var(y_data)
    # y_data_var = 1/len(y_data) * np.sum((y_data - y_data_mean) ** 2,0)
    stats = {}
    stats['y_mean'] = y_data_mean
    stats['y_var'] = y_data_var

    return data_loader, stats
