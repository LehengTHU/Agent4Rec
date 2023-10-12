import os
import random
import numpy as np
import torch
from termcolor import colored, cprint
import matplotlib.pyplot as plt




def fix_seeds(seed=101):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # In order to disable hash randomization and make the experiment reproducible.
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

def get_accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

def get_recall(y_true, y_pred):
    return np.sum(y_true & y_pred) / np.sum(y_true)

def get_precision(y_true, y_pred):
    return np.sum(y_true & y_pred) / np.sum(y_pred)

def get_f1(y_true, y_pred):
	p = get_precision(y_true, y_pred)
	r = get_recall(y_true, y_pred)
	if p + r == 0:
		return 0
	else:
		return 2 * p * r / (p + r)

# def write_log(log_file, log, color=None, attrs=None):
# 	with open(log_file, 'w') as f:
# 		f.write(log + '\n')
# 		f.flush()
# 	cprint(log, color=color, attrs=attrs)
