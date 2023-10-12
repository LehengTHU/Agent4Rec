import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base.abstract_model import AbstractModel
from models.base.abstract_RS import AbstractRS
from tqdm import tqdm

class MF_RS(AbstractRS):
    def __init__(self, args, special_args) -> None:
        super().__init__(args, special_args)

    def train_one_epoch(self, epoch):
        running_loss, running_mf_loss, running_reg_loss, num_batches = 0, 0, 0, 0

        pbar = tqdm(enumerate(self.data.train_loader), mininterval=2, total = len(self.data.train_loader))
        for batch_i, batch in pbar:          
            
            batch = [x.cuda(self.device) for x in batch]
            users, pos_items, users_pop, pos_items_pop, pos_weights  = batch[0], batch[1], batch[2], batch[3], batch[4]

            if self.args.infonce == 0 or self.args.neg_sample != -1:
                neg_items = batch[5]
                neg_items_pop = batch[6]

            self.model.train()
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


class MF(AbstractModel):
    def __init__(self, args, data) -> None:
        super().__init__(args, data)
    
    def forward(self, users, pos_items, neg_items):
        all_users, all_items = self.embed_user.weight, self.embed_item.weight

        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]

        userEmb0 = self.embed_user(users)
        posEmb0 = self.embed_item(pos_items)
        negEmb0 = self.embed_item(neg_items)


        pos_scores = torch.sum(torch.mul(users_emb, pos_emb), dim=1)  # users, pos_items, neg_items have the same shape
        neg_scores = torch.sum(torch.mul(users_emb, neg_emb), dim=1)

        regularizer = 0.5 * torch.norm(userEmb0) ** 2 + 0.5 * torch.norm(posEmb0) ** 2 + 0.5 * torch.norm(negEmb0) ** 2
        regularizer = regularizer / self.batch_size

        maxi = torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-10)
        mf_loss = torch.negative(torch.mean(maxi))
        reg_loss = self.decay * regularizer

        return mf_loss, reg_loss
    
    def predict(self, users, items=None):
        if items is None:
            items = list(range(self.data.n_items))

        all_users, all_items = self.embed_user.weight, self.embed_item.weight

        users = all_users[torch.tensor(users).cuda(self.device)]
        items = all_items[torch.tensor(items).cuda(self.device)]
        
        items = torch.transpose(items, 0, 1)
        rate_batch = torch.matmul(users, items) # user * item

        return rate_batch.cpu().detach().numpy()