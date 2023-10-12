import random as rd
import collections
from types import new_class
import numpy as np
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from parse import parse_args
import time
import torch
from copy import deepcopy
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from reckit import randint_choice
import os


# Helper function used when loading data from files
def helper_load(filename):
    user_dict_list = {}
    item_dict = set()

    with open(filename) as f:
        for line in f.readlines():
            line = line.strip('\n').split(' ')
            if len(line) == 0:
                continue
            line = [int(i) for i in line]
            user = line[0]
            items = line[1:]
            item_dict.update(items)
            if len(items) == 0:
                continue
            user_dict_list[user] = items

    return user_dict_list, item_dict,

def helper_load_train(filename):
    user_dict_list = {}
    item_dict = set()
    item_dict_list = {}
    trainUser, trainItem = [], []

    with open(filename) as f:
        for line in f.readlines():
            line = line.strip('\n').split(' ')
            # print(line)
            if len(line) == 0:
                continue
            line = [int(i) for i in line]
            user = line[0]
            items = line[1:]
            item_dict.update(items)
            # LightGCN
            trainUser.extend([user] * len(items))
            trainItem.extend(items)
            if len(items) == 0:
                continue
            user_dict_list[user] = items

            for item in items:
                if item in item_dict_list.keys():
                    item_dict_list[item].append(user)
                else:
                    item_dict_list[item] = [user]

    return user_dict_list, item_dict, item_dict_list, trainUser, trainItem
# It loads the data and creates a train_loader

class Data:
    
    def __init__(self, args):
        self.path = args.data_path + args.dataset + '/cf_data/'
        self.small_path=args.data_path + args.dataset+".mid"+"/"
        self.train_file = self.path + 'train.txt'
        self.valid_file = self.path + 'valid.txt'
        self.test_file = self.path + 'test.txt'

        if(args.nodrop):
            self.train_nodrop_file = self.path + 'train_nodrop.txt'
        self.nodrop = args.nodrop

        self.candidate = args.candidate
        if(args.candidate):
            self.test_neg_file = self.path + 'test_neg.txt'
        self.batch_size = args.batch_size
        self.neg_sample = args.neg_sample
        self.IPStype = args.IPStype
        self.device = torch.device(args.cuda)
        self.modeltype = args.modeltype

        self.user_pop_max = 0
        self.item_pop_max = 0
        self.infonce = args.infonce
        self.num_workers = args.num_workers
        self.dataset = args.dataset
        self.candidate = args.candidate

        # Number of total users and items
        self.n_users, self.n_items, self.n_observations = 0, 0, 0
        self.users = []
        self.items = []
        self.population_list = []
        self.weights = []

        # List of dictionaries of users and its observed items in corresponding dataset
        # {user1: [item1, item2, item3...], user2: [item1, item3, item4],...}
        # {item1: [user1, user2], item2: [user1, user3], ...}
        self.train_user_list = collections.defaultdict(list)
        self.valid_user_list = collections.defaultdict(list)
        if(self.dataset == "tencent_synthetic" or self.dataset == "kuairec_ood"):
            self.test_ood_user_list_1 = collections.defaultdict(list)
            self.test_ood_user_list_2 = collections.defaultdict(list)
            self.test_ood_user_list_3 = collections.defaultdict(list)
        else:
            self.test_user_list = collections.defaultdict(list)

        # Used to track early stopping point
        self.best_valid_recall = -np.inf
        self.best_valid_epoch, self.patience = 0, 0

        self.train_item_list = collections.defaultdict(list)
        self.Graph = None
        self.trainUser, self.trainItem, self.UserItemNet = [], [], []
        self.n_interactions = 0
        if(self.dataset == "tencent_synthetic" or self.dataset == "kuairec_ood"):
            self.test_ood_item_list_1 = []
            self.test_ood_item_list_2 = []
            self.test_ood_item_list_3 = []
        else:
            self.test_item_list = []

        #Dataloader 
        self.train_data = None
        self.train_loader = None

        self.load_data()
        # model-specific attributes
        self.add_special_model_attr(args)

        self.get_dataloader()

    def add_special_model_attr(self, args):
        pass

    # self.trainUser and self.trainItem are respectively the users and items in the training set, in the form of an interaction list.
    def load_data(self):
        self.train_user_list, train_item, self.train_item_list, self.trainUser, self.trainItem = helper_load_train(
            self.train_file)
        self.valid_user_list, valid_item = helper_load(self.valid_file)

        self.test_user_list, self.test_item_list = helper_load(self.test_file)

        if(self.nodrop):
            self.train_nodrop_user_list, self.train_nodrop_item_list = helper_load(self.train_nodrop_file)

        if(self.candidate):
            self.test_neg_user_list, self.test_neg_item_list = helper_load(self.test_neg_file)
        else:
            self.test_neg_user_list, self.test_neg_item_list = None, None
        self.pop_dict_list = []


        temp_lst = [train_item, valid_item, self.test_item_list]

        self.users = list(set(self.train_user_list.keys()))
        self.items = list(set().union(*temp_lst))
        self.items.sort()
        # print(self.items)
        self.n_users = len(self.users)
        self.n_items = len(self.items)

        
        print("n_users: ", self.n_users)
        print("n_items: ", self.n_items)
        
        for i in range(self.n_users):
            self.n_observations += len(self.train_user_list[i])
            self.n_interactions += len(self.train_user_list[i])
            if i in self.valid_user_list.keys():
                self.n_interactions += len(self.valid_user_list[i])
            if(self.dataset == "tencent_synthetic" or self.dataset == "kuairec_ood"):
                if i in self.test_ood_user_list_1.keys():
                    self.n_interactions += len(self.test_ood_user_list_1[i])
                if i in self.test_ood_user_list_2.keys():
                    self.n_interactions += len(self.test_ood_user_list_2[i])
                if i in self.test_ood_user_list_3.keys():
                    self.n_interactions += len(self.test_ood_user_list_3[i])
            else:
                if i in self.test_user_list.keys():
                    self.n_interactions += len(self.test_user_list[i])



        # Population matrix
        pop_dict = {}
        for item, users in self.train_item_list.items():
            pop_dict[item] = len(users) + 1
        for item in range(0, self.n_items):
            if item not in pop_dict.keys():
                pop_dict[item] = 1

            self.population_list.append(pop_dict[item])

        pop_user = {key: len(value) for key, value in self.train_user_list.items()}
        pop_item = {key: len(value) for key, value in self.train_item_list.items()}
        self.pop_item = pop_item
        self.pop_user = pop_user
        # Convert to a unique value.
        sorted_pop_user = list(set(list(pop_user.values())))
        sorted_pop_item = list(set(list(pop_item.values())))
        sorted_pop_user.sort()
        sorted_pop_item.sort()
        self.n_user_pop = len(sorted_pop_user)
        self.n_item_pop = len(sorted_pop_item)

        user_idx = {}
        item_idx = {}
        for i, item in enumerate(sorted_pop_user):
            user_idx[item] = i
        for i, item in enumerate(sorted_pop_item):
            item_idx[item] = i

        self.user_pop_idx = np.zeros(self.n_users, dtype=int)
        self.item_pop_idx = np.zeros(self.n_items, dtype=int)
        # Convert the originally sparse popularity into dense popularity.
        for key, value in pop_user.items():
            self.user_pop_idx[key] = user_idx[value]
        for key, value in pop_item.items():
            # print(key, value)
            self.item_pop_idx[key] = item_idx[value]

        user_pop_max = max(self.user_pop_idx)
        item_pop_max = max(self.item_pop_idx)

        self.user_pop_max = user_pop_max
        self.item_pop_max = item_pop_max        

        self.weights = self.get_weight()
        self.weight_dict={i:self.weights[i] for i in range(len(self.weights))}
        self.sorted_weight=sorted(self.weight_dict.items(),key=lambda x: x[1])

        self.sample_items = np.array(self.items, dtype=int)

        
    def get_dataloader(self):
        self.train_data = TrainDataset(self.modeltype, self.users, self.train_user_list, self.user_pop_idx, self.item_pop_idx, \
                                        self.neg_sample, self.n_observations, self.n_items, self.sample_items, self.weights, self.infonce, self.items)

        self.train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True)

    def get_weight(self):

        if 's' in self.IPStype:
            pop = self.population_list
            pop = np.clip(pop, 1, max(pop))
            pop = pop / max(pop)
            return pop


        pop = self.population_list
        pop = np.clip(pop, 1, max(pop))
        pop = pop / np.linalg.norm(pop, ord=np.inf)
        pop = 1 / pop

        if 'c' in self.IPStype:
            pop = np.clip(pop, 1, np.median(pop))
        if 'n' in self.IPStype:
            pop = pop / np.linalg.norm(pop, ord=np.inf)

        return pop

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def getSparseGraph(self):

        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + '/s_pre_adj_mat.npz')
                print("finish loading adjacency matrix")
                norm_adj = pre_adj_mat
            # If there is no preprocessed adjacency matrix, generate one.
            except:
                print("generating adjacency matrix")
                s = time.time()
                adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
                adj_mat = adj_mat.tolil()
                self.trainItem = np.array(self.trainItem)
                self.trainUser = np.array(self.trainUser)
                self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                                shape=(self.n_users, self.n_items))
                R = self.UserItemNet.tolil()
                adj_mat[:self.n_users, self.n_users:] = R
                adj_mat[self.n_users:, :self.n_users] = R.T
                adj_mat = adj_mat.tocsr()
                sp.save_npz(self.path + '/adj_mat.npz', adj_mat)
                print("successfully saved adj_mat...")

                adj_mat = adj_mat.todok()

                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time.time()
                print(f"costing {end - s}s, saved norm_mat...")
                sp.save_npz(self.path + '/s_pre_adj_mat.npz', norm_adj)
            self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self.Graph = self.Graph.coalesce().cuda(self.device)

        return self.Graph
    
    def get_not_candidate(self):
        if self.candidate:
            not_candidate_dict = {}
            with open('data/' + self.dataset + '/not_candidate.txt', 'r') as f:
                for line in f.readlines():
                    line = line.strip('\n').split(' ')
                    if len(line) == 0:
                        continue
                    line = [int(i) for i in line]
                    user = line[0]
                    items = line[1:]
                    not_candidate_dict[user] = items

            return not_candidate_dict
        else:
            return None

class TrainDataset(torch.utils.data.Dataset):

    def __init__(self, modeltype, users, train_user_list, user_pop_idx, item_pop_idx, neg_sample, \
                n_observations, n_items, sample_items, weights, infonce, items):
        self.modeltype = modeltype
        self.users = users
        self.train_user_list = train_user_list
        self.user_pop_idx = user_pop_idx
        self.item_pop_idx = item_pop_idx
        self.neg_sample = neg_sample
        self.n_observations = n_observations
        self.n_items = n_items
        self.sample_items = sample_items
        self.weights = weights
        self.infonce = infonce
        self.items = items

    def __getitem__(self, index):

        index = index % len(self.users)
        user = self.users[index]
        if self.train_user_list[user] == []:
            pos_items = 0
        else:
            pos_item = rd.choice(self.train_user_list[user])

        user_pop = self.user_pop_idx[user]
        pos_item_pop = self.item_pop_idx[pos_item]
        pos_weight = self.weights[pos_item]

        if self.infonce == 1 and self.neg_sample == -1:

            return user, pos_item, user_pop, pos_item_pop, pos_weight

        elif self.infonce == 1 and self.neg_sample != -1:
            neg_items = randint_choice(self.n_items, size=self.neg_sample, exclusion=self.train_user_list[user])
            neg_items_pop = self.item_pop_idx[neg_items]

            return user, pos_item, user_pop, pos_item_pop, pos_weight, torch.tensor(neg_items).long(), neg_items_pop

        else:
            while True:
                idx = rd.randint(0, self.n_items -1)
                neg_item = self.items[idx]
                
                if neg_item not in self.train_user_list[user]:
                    break
        
            neg_item_pop = self.item_pop_idx[neg_item]
            return user, pos_item, user_pop, pos_item_pop, pos_weight, neg_item, neg_item_pop

    def __len__(self):
        return self.n_observations

