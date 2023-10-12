import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base.abstract_model import AbstractModel
from models.base.abstract_RS import AbstractRS
from tqdm import tqdm


class Random_RS(AbstractRS):
    def __init__(self, args, special_args) -> None:
        super().__init__(args, special_args)

    def train_one_epoch(self, epoch):
        return None

class Random(AbstractModel):
    def __init__(self, args, data) -> None:
        super().__init__(args, data)
    
    def forward(self):
        return None
    
    def predict(self, users, items=None):
        if items is None:
            items = list(range(self.data.n_items))

        return np.random.rand(len(users), len(items))