from cmath import cos
import torch
import torch.nn as nn
import math
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from scipy.special import lambertw
import random
# from data import Data
import scipy.sparse as sp

# based on LightGCN
# n_layers = 0: MF
class AbstractModel(nn.Module):
    def __init__(self, args, data):
        super(AbstractModel, self).__init__()
        print("AbstractModel")

        # basic information
        self.args = args
        self.name = args.modeltype
        self.device = torch.device(args.cuda)
        # self.saveID = args.saveID
        self.data = data

        # graph
        self.Graph = data.getSparseGraph()

        # basic hyper-parameters
        self.emb_dim = args.embed_size
        self.decay = args.regs
        self.train_norm = args.train_norm
        self.pred_norm = args.pred_norm
        self.n_layers = args.n_layers
        self.modeltype = args.modeltype
        self.batch_size = args.batch_size

        self.init_embedding()

    def init_embedding(self):
        self.embed_user = nn.Embedding(self.data.n_users, self.emb_dim)
        self.embed_item = nn.Embedding(self.data.n_items, self.emb_dim)

        nn.init.xavier_normal_(self.embed_user.weight)
        nn.init.xavier_normal_(self.embed_item.weight)

    def compute(self):
        users_emb = self.embed_user.weight
        items_emb = self.embed_item.weight
        all_emb = torch.cat([users_emb, items_emb])

        embs = [all_emb]
        g_droped = self.Graph

        for layer in range(self.n_layers):
            # print(g_droped.device, all_emb.device)
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)

        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.data.n_users, self.data.n_items])

        return users, items

    #! must be implemented
    def forward(self):
        raise NotImplementedError

    # Prediction function used when evaluation
    def predict(self, users, items=None):
        if items is None:
            items = list(range(self.data.n_items))

        all_users, all_items = self.compute()

        users = all_users[torch.tensor(users).cuda(self.device)]
        items = all_items[torch.tensor(items).cuda(self.device)]
        
        if(self.pred_norm == True):
            users = F.normalize(users, dim = -1)
            items = F.normalize(items, dim = -1)
        items = torch.transpose(items, 0, 1)
        rate_batch = torch.matmul(users, items) # user * item

        return rate_batch.cpu().detach().numpy()