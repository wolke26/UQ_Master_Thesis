# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 17:05:19 2020

@author: Gloria
"""

import torch
import numpy as np
import h5py
from torch.utils.data import DataLoader, TensorDataset
from pdb import set_trace
import pickle
import matplotlib.pyplot as plt


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
    # data_dir = r"C:\Users\Gloria\Documents\Masterarbeit\Output_nH10000\Output\Trainings_Data\Final_Trainings_data_4200.pickle"
        # data_dir = r"C:\Users\Gloria\Documents\Masterarbeit\Output_nH10000\Output\Trainings_Data\Unseen_data_5000.pickle"
        # data_dir = r"C:\Users\Gloria\Documents\Masterarbeit\Output_nH10000\Output\Trainings_Data\Test_data_800.pickle"

    # set_trace()
    with open(data_dir,'rb') as g:
        data_for_nn = pickle.load(g)
       
    # set_trace()
    # set_trace()
    x_data = np.float32(np.vstack(data_for_nn['rfields']).reshape(-1,1,20,20))
  
    
  
    # y_data = np.float32(normalize_between(np.vstack(np.array(data_for_nn['sigma_33'])),data_for_nn['min_sigma_33'][0][0], \
    #                                       data_for_nn['min_sigma_33'][0][1])).reshape(-1,400)
   
    
    y_data = np.float32(np.vstack(np.array(data_for_nn['sigma_33']))).reshape(-1,400)
    sig_11_data = np.float32(np.vstack(np.array(data_for_nn['sigma_11']))).reshape(-1,400)
    sig_12_data = np.float32(np.vstack(np.array(data_for_nn['sigma_12']))).reshape(-1,400)
    sig_22_data = np.float32(np.vstack(np.array(data_for_nn['sigma_22']))).reshape(-1,400)
    sig_13_data = np.float32(np.vstack(np.array(data_for_nn['sigma_13']))).reshape(-1,400)
    sig_23_data = np.float32(np.vstack(np.array(data_for_nn['sigma_23']))).reshape(-1,400)
    # set_trace()
    
    def PAV(sig_11_data,sig_12_data,sig_13_data,sig_23_data,sig_22_data,y_data):
        sig11 = list(map(lambda x:x*x, sig_11_data))
        sig12 = list(map(lambda x:x*x, sig_12_data))
        sig23 = list(map(lambda x:x*x, sig_23_data))
        sig13 = list(map(lambda x:x*x, sig_13_data))
        sig22 = list(map(lambda x:x*x, sig_22_data))
        sig33 = list(map(lambda x:x*x, y_data))
        
        return(np.array([sum(x) for x in zip(sig11,sig12,sig13,sig22,sig33,sig23)]))
    
    # pav_values = PAV(sig_11_data,sig_12_data,sig_13_data,sig_23_data,sig_22_data,y_data).reshape(-1,20,20)
    # set_trace()
    pav_values22 = PAV(sig_11_data,sig_12_data,sig_13_data,sig_23_data,sig_22_data,y_data).reshape(-1,20,20)

    pav_values = normalize_between(pav_values22, np.amin(pav_values22), np.amax(pav_values22))
    for i in range(5):
        
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(y_data[i].reshape(20,20))
        plt.title('$\sigma_{33}$')
        plt.subplot(1,2,2)
        plt.imshow(pav_values[i])
        plt.title('PAV -  squared sum \n at each location')
        plt.savefig('PAV_comparison_sigma_33_'+str(i)+'.pdf')
    
    pav_values = pav_values.reshape(-1,1,20,20)
    print("input data shape: {}".format(x_data.shape))
    print("output data shape: {}".format(pav_values.shape))

    kwargs = {'num_workers': 20,
              'pin_memory': True} if torch.cuda.is_available() else {}

    dataset = TensorDataset(torch.tensor(x_data), torch.tensor(pav_values))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, **kwargs)
    # set_trace()
    # simple statistics of output data
    y_data_mean = np.mean(pav_values, 0)
    y_data_var =  np.var(pav_values)
    # y_data_var = 1/len(y_data) * np.sum((y_data - y_data_mean) ** 2,0)
    stats = {}
    stats['y_mean'] = y_data_mean
    stats['y_var'] = y_data_var

    return data_loader, stats
