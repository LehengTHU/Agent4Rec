import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base.abstract_model import AbstractModel
from models.base.abstract_RS import AbstractRS
from tqdm import tqdm

from data import Data


class Pop_RS(AbstractRS):
    def __init__(self, args, special_args) -> None:
        super().__init__(args, special_args)

    def train_one_epoch(self, epoch):
        return None

class Pop(AbstractModel):
    def __init__(self, args, data) -> None:
        super().__init__(args, data)
    
    def forward(self):
        return None
    
    def predict(self, users, items=None):
        if items is None:
            items = list(range(self.data.n_items))

        rating_matrix = np.zeros((len(users), len(items)))
        for i, user in enumerate(users):
            random_idx = np.random.choice(self.data.pop_candidates, 2*self.args.Ks, replace=False) # 每次从pop_candidates中选取20个item
            # print(sorted(random_idx))
            rating_matrix[i, random_idx] = 1
        # print(rating_matrix.sum())


        return rating_matrix
    
class Pop_Data(Data):
    def __init__(self, args):
        super().__init__(args)
    
    def add_special_model_attr(self, args):
        sorted_items = sorted(self.pop_item.items(), key=lambda x: x[1], reverse=True)
        self.pop_candidates = [x[0] for x in sorted_items[:30*args.Ks]]
        print("pop_candidates: ", sorted(self.pop_candidates))
        # pop_matrix = np.zeros((1, self.n_items))
        # # 随机从pop_candidates中选取20个item
        # rating_matrix = np.zeros((self.n_users, self.n_items))
        # for i, user in enumerate(range(self.n_users)):
        #     print(i, user)
        #     random_idx = np.random.choice(self.pop_candidates, 20, replace=False)
        #     rating_matrix[i, random_idx] = 1

        # # 取rating_matrix排名前20的item的idx
        # np.argsort(-rating_matrix, axis=1)
        # print("??")
        # print(pop_matrix)
        # print(self.pop_candidates)
        # # return None