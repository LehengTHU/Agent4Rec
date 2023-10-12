import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base.abstract_model import AbstractModel
from models.base.abstract_RS import AbstractRS
from tqdm import tqdm

class InfoNCE_RS(AbstractRS):
    def __init__(self, args, special_args) -> None:
        super().__init__(args, special_args)
        self.neg_sample =  args.neg_sample if args.neg_sample!=-1 else self.batch_size-1

    def train_one_epoch(self, epoch):
        running_loss, running_mf_loss, running_reg_loss, num_batches = 0, 0, 0, 0

        pbar = tqdm(enumerate(self.data.train_loader), mininterval=2, total = len(self.data.train_loader))
        for batch_i, batch in pbar:          

            batch = [x.cuda(self.device) for x in batch]
            users, pos_items, users_pop, pos_items_pop, pos_weights  = batch[0], batch[1], batch[2], batch[3], batch[4]

            self.model.train()
            if(self.inbatch):
                mf_loss, reg_loss = self.model(users, pos_items)
            else:
                neg_items = batch[5]
                neg_items_pop = batch[6]
                mf_loss, reg_loss = self.model(users, pos_items, neg_items)
            loss = mf_loss + reg_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.detach().item()
            running_reg_loss += reg_loss.detach().item()
            running_mf_loss += mf_loss.detach().item()
            num_batches += 1
        return [running_loss/num_batches, running_mf_loss/num_batches, running_reg_loss/num_batches]

class InfoNCE(AbstractModel):
    def __init__(self, args, data) -> None:
        super().__init__(args, data)
        self.tau = args.tau
    
    def forward(self, users, pos_items, neg_items):

        all_users, all_items = self.compute()

        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)
        negEmb0 = self.embed_item(neg_items)

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        if(self.train_norm):
            users_emb = F.normalize(users_emb, dim = -1)
            pos_emb = F.normalize(pos_emb, dim = -1)
            neg_emb = F.normalize(neg_emb, dim = -1)
        
        pos_ratings = torch.sum(users_emb*pos_emb, dim = -1)
        neg_ratings = torch.matmul(torch.unsqueeze(users_emb, 1), 
                                       neg_emb.permute(0, 2, 1)).squeeze(dim=1)

        numerator = torch.exp(pos_ratings / self.tau)

        denominator = numerator + torch.sum(torch.exp(neg_ratings / self.tau), dim = 1)
        
        ssm_loss = torch.mean(torch.negative(torch.log(numerator/denominator)))

        regularizer = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2 + 0.5 ** torch.norm(negEmb0)
        regularizer = regularizer / self.batch_size
        reg_loss = self.decay * regularizer

        return ssm_loss, reg_loss

class InfoNCE_batch(AbstractModel):
    def __init__(self, args, data) -> None:
        super().__init__(args, data)
        self.tau = args.tau

    def forward(self, users, pos_items):

        all_users, all_items = self.compute()

        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]

        users_emb = F.normalize(users_emb, dim=1)
        pos_emb = F.normalize(pos_emb, dim=1)
        
        ratings = torch.matmul(users_emb, torch.transpose(pos_emb, 0, 1))
        ratings_diag = torch.diag(ratings)
        
        #分子
        numerator = torch.exp(ratings_diag / self.tau)
        #分母
        denominator = torch.sum(torch.exp(ratings / self.tau), dim = 1)
        ssm_loss = torch.mean(torch.negative(torch.log(numerator/denominator)))


        regularizer = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2
        regularizer = regularizer
        reg_loss = self.decay * regularizer

        return ssm_loss, reg_loss