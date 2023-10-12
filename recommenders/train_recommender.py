import random
import re
from sys import get_coroutine_origin_tracking_depth
from sys import exit
random.seed(101)
import matplotlib.pyplot as plt
import math
import matplotlib.patches as mpatches
#from scipy.linalg import svd
import itertools
import torch
import time
import numpy as np
from tqdm import tqdm
from evaluator import ProxyEvaluator
import collections
import os
from data import Data
from parse import parse_args

from torch.utils.data import Dataset, DataLoader
# from collect_log import read_log
import torch.nn.functional as F
from models.base.utils import *
import wandb

# load model
if __name__ == '__main__':
    args, special_args = parse_args()
    fix_seeds(args.seed) # set random seed

    if(not args.no_wandb):
        wandb.init(
            # set the wandb project where this run will be logged
            project = "recommender_training",
            name = args.saveID,
            group = args.modeltype
        )
    # import sys
    # print(__file__)
    # print(sys.argv[0])

    # from models.LightGCN import LightGCN_RS
    # exec('from models.'+ args.modeltype + ' import ' + args.modeltype + '_RS')
    try:
        exec('from models.'+ args.modeltype + ' import ' + args.modeltype + '_RS') # load the model
    except:
        print('Model %s not implemented!' % (args.modeltype))
    #     exit(1)
        
    RS = eval(args.modeltype + '_RS(args, special_args)')

    # activate the recommender system
    # print(os.getcwd())
    RS.execute() # train and test
    if(not args.no_wandb):
        wandb.finish()
    
