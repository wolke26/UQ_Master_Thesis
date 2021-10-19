# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 20:17:15 2020

@author: Gloria
"""


# import torch
# from args import args
# from models.dense_ed import DenseED
# from models.bayes_nn import BayesNN
# from uq import UQ
# from utils.load_data import load_data
# import torch.nn as nn
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pickle 

# n_pred = 4990
n_pred = 5000
bin_size = 20
with open("./UQ_dictionary_predict_"+str(n_pred)+"_fields_const_model_PAV.pickle",'rb') as f:
    my_dict = pickle.load(f)    
    
my_dict['pred_mean_all'] = my_dict['pred_mean_all'].reshape(n_pred,20,20)
# pix_of_interest_1 = [:,1,10]

arr_pix_of_interest_1 = my_dict['pred_mean_all'][:,1,1]
arr_pix_of_interest_2 = my_dict['pred_mean_all'][:,1,10]
arr_pix_of_interest_3 = my_dict['pred_mean_all'][:,1,19]


arr_pix_of_interest_4 = my_dict['pred_mean_all'][:,10,1]
arr_pix_of_interest_5 = my_dict['pred_mean_all'][:,10,10]
arr_pix_of_interest_6 = my_dict['pred_mean_all'][:,10,19]


arr_pix_of_interest_7 = my_dict['pred_mean_all'][:,19,1]
arr_pix_of_interest_8 = my_dict['pred_mean_all'][:,19,10]
arr_pix_of_interest_9 = my_dict['pred_mean_all'][:,19,19]

fig, axs = plt.subplots(3, 3, figsize = (7,7))
plt.suptitle('Histogram of '+str(n_pred)+' predictions', size = 18 )
# plt.subplots_adjust(top=0.85)

im1 = axs[0, 0].hist(arr_pix_of_interest_1, bins=bin_size)
axs[0, 0].set_title('$loc 1$')

im2 = axs[0, 1].hist(arr_pix_of_interest_2, bins=bin_size)
axs[0, 1].set_title('loc 2')

im3 = axs[0, 2].hist(arr_pix_of_interest_3, bins=bin_size)
axs[0, 2].set_title('loc 3')

im4 = axs[1, 0].hist(arr_pix_of_interest_4, bins=bin_size)
axs[1, 0].set_title('loc 4')


im5 = axs[1, 1].hist(arr_pix_of_interest_5, bins=bin_size)
axs[1, 1].set_title('loc 5')


im6 = axs[1, 2].hist(arr_pix_of_interest_6, bins=bin_size)
axs[1, 2].set_title('loc 6')


im7 = axs[2, 0].hist(arr_pix_of_interest_7, bins=bin_size)
axs[2, 0].set_title('loc 7')


im7 = axs[2,1].hist(arr_pix_of_interest_8, bins=bin_size)
axs[2, 1].set_title('loc 8')


im9 = axs[2, 2].hist(arr_pix_of_interest_9, bins=bin_size)
axs[2, 2].set_title('loc 9')



plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.2, hspace=0.5)
plt.savefig('Histogram_'+str(n_pred)+'_'+str(bin_size)+'_const_model_PAV.pdf')

