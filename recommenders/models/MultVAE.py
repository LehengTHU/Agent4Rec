import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base.abstract_model import AbstractModel
from models.base.abstract_RS import AbstractRS
from tqdm import tqdm

from data import Data, TrainDataset
from torch.utils.data import DataLoader

from scipy.sparse import csr_matrix

# import random
import random as rd
from reckit import randint_choice
import scipy.sparse as sp

def naive_sparse2tensor(data):
    return torch.FloatTensor(data.toarray())

class MultVAE_RS(AbstractRS):
    def __init__(self, args, special_args) -> None:
        super().__init__(args, special_args)
        self.total_anneal_steps = args.total_anneal_steps
        self.anneal_cap = args.anneal_cap
        self.update_count = 0
    
    def set_optimizer(self):
        self.optimizer = torch.optim.Adam([param for param in self.model.parameters() if param.requires_grad == True], lr=self.lr)
    
    def loss_function(self, recon_x, x, mu, logvar, anneal=1.0):
        # BCE = F.binary_cross_entropy(recon_x, x)
        BCE = -torch.mean(torch.sum(F.log_softmax(recon_x, 1) * x, -1))
        KLD = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

        return BCE + anneal * KLD

    def train_one_epoch(self, epoch):
        running_loss, num_batches = 0, 0

        n_users = self.data.n_users
        idxlist = np.arange(n_users)
        np.random.shuffle(idxlist)
        pbar = tqdm(enumerate(range(0, n_users, self.batch_size)))
        for batch_i, start_idx in pbar:          
            end_idx = min(start_idx + self.batch_size, n_users)
            batch = self.data.ui_mat[idxlist[start_idx:end_idx]]
            batch = naive_sparse2tensor(batch).cuda(self.device)

            if self.total_anneal_steps > 0:
                anneal = min(self.anneal_cap, 
                                1. * self.update_count / self.total_anneal_steps)
            else:
                anneal = self.anneal_cap

            self.optimizer.zero_grad()
            recon_batch, mu, logvar = self.model(batch)
            
            loss = self.loss_function(recon_batch, batch, mu, logvar, anneal)
            loss.backward()
            running_loss += loss.detach().item()
            num_batches += 1
            self.optimizer.step()

            self.update_count += 1
        return [running_loss/num_batches]


class MultVAE_Data(Data):
    def __init__(self, args):
        super().__init__(args)
    
    def add_special_model_attr(self, args):
        try:
            self.ui_mat = sp.load_npz(self.path + '/ui_mat.npz')
            print("successfully loaded ui_mat...")
        except:
            self.trainItem = np.array(self.trainItem)
            self.trainUser = np.array(self.trainUser)
            self.ui_mat = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                    shape=(self.n_users, self.n_items))
            sp.save_npz(self.path + '/ui_mat.npz', self.ui_mat)
            print("successfully saved ui_mat...")

class MultVAE(AbstractModel):
    def __init__(self, args, data) -> None:
        super().__init__(args, data)
        self.p_dims = [args.p_dim0, args.p_dim1, data.n_items]
        self.q_dims = self.p_dims[::-1]

        # Last dimension of q- network is for mean and variance
        temp_q_dims = self.q_dims[:-1] + [self.q_dims[-1] * 2]
        self.q_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])])
        self.p_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:])])
        
        self.drop = nn.Dropout(0.5)
        self.init_weights()
    
    def forward(self, input):
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def encode(self, input):
        h = F.normalize(input)
        h = self.drop(h)
        
        for i, layer in enumerate(self.q_layers):
            h = layer(h)
            if i != len(self.q_layers) - 1:
                h = F.tanh(h)
            else:
                mu = h[:, :self.q_dims[-1]]
                logvar = h[:, self.q_dims[-1]:]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
    
    def decode(self, z):
        h = z
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                h = F.tanh(h)
        return h

    def init_weights(self):
        for layer in self.q_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)
        
        for layer in self.p_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)
    
    def predict(self, users, items=None):
        if items is None:
            items = list(range(self.data.n_items))
        
        batch = naive_sparse2tensor(self.data.ui_mat[users])
        batch = batch.cuda(self.device)
        rate_batch, _, _ = self.forward(batch)

        return rate_batch.cpu().detach().numpy()
