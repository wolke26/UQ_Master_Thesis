"""
Post processing, mainly for uncertainty quantification tasks using pre-trained
Bayesian NNs.
"""

import torch
from args import args
from models.dense_ed import DenseED
from models.bayes_nn import BayesNN
from uq import UQ
from utils.load_data import load_data
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# assert args.post, 'Add --post flag in command line for post-proc UQ tasks'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model
# deterministic NN
dense_ed = DenseED(in_channels=args.nic, 
                    out_channels=args.noc, 
                    blocks=args.blocks,
                    growth_rate=args.growth_rate, 
                    init_features=args.init_features,
                    drop_rate=args.drop_rate,
                    bn_size=args.bn_size,
                    bottleneck=args.bottleneck,
                    out_activation=None)
# print(dense_ed)
# Bayesian NN
bayes_nn = nn.DataParallel(BayesNN(dense_ed, n_samples=args.n_samples)).to(device)
# # load the pre-trained model
# if args.ckpt_epoch is not None:
#     checkpoint = args.ckpt_dir + '/model_epoch{}.pth'.format(args.ckpt_epoch)
# else:
checkpoint = args.ckpt_dir + '/model_epoch{}.pth'.format(args.ckpt_epoch)
print(args.ckpt_epoch)
# bayes_nn.load_state_dict(torch.load(checkpoint))


# --> checkpoint = "./experiments/Bayesian/rfields1000/nsamples20_ntrain1000_batch100_lr0.03_noiselr0.0_epochs350/checkpoints" + '/model_epoch{}.pth'.format(args.epochs)

bayes_nn.load_state_dict(torch.load(checkpoint))


print('Loaded pre-trained model: {}'.format(checkpoint))

# load Monte Carlo data
# mc_data_dir = r"./Trainings_Data/Final_unseen_data_5000.pickle"    
mc_data_dir = r"./Trainings_Data/UQ_const_model_unseen_data_5000.pickle"    




print('Process used is ' + str(device))
# from utils.load_data_PAV import load_data
# Load and train either one more output channels
mc_loader, _ = load_data(mc_data_dir, args.mc_batch_size)
print('Loaded Monte Carlo data!')

# Now performs UQ tasks
uq = UQ(bayes_nn, mc_loader)


# Predict some and so Zbaras Uncertainty Quantification:
with torch.no_grad():
    uq.plot_prediction_at_x(n_pred=8)
    uq.propagate_uncertainty()
    uq.plot_dist(num_loc=20)
    uq.plot_reliability_diagram()











    
    
  
    
    
    