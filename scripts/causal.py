#%%
import sys

sys.path.append("")
import unittest
from pickle import load

import numpy as np
import pandas as pd

from causallearn.search.FCMBased import lingam

import os
import pickle
import random
import torch

import matplotlib.pyplot as plt
from utils import *

# %%
fix_seeds(101)
# %%
# n_ids = 500
n_ids = 1000
item_per_page = 20
dataset_prefix = "ml-1m"

base_path = f"../storage/ml-1m/LightGCN/lgn_1000_5_4_1009"

original_path = f"../datasets/{dataset_prefix}/cf_data/"
movie_detail_path = "../datasets/ml-1m/simulation/movie_detail.csv"

bahavior_path = base_path + "/behavior/"
rankings_file_path = base_path + f"/rankings/full_rankings_{n_ids}.npy"
new_train_path = base_path + "/train.txt"
full_rankings = np.load(rankings_file_path)
movie_detail = pd.read_csv(movie_detail_path)

original_train_path = original_path + "train.txt"
original_valid_path = original_path + "valid.txt"
original_test_path = original_path + "test.txt"

train_user_dict_list, train_item_dict, train_item_dict_list, train_trainUser, train_trainItem = helper_load_train(original_train_path)

item_pop = {key: len(value) for key, value in train_item_dict_list.items()}
# %%
records = pd.DataFrame(columns=['item_id', 'position', 'popularity', 'exposure', 'propagation', 'quality', 'click', 'rating'])

for i in range(0, n_ids):
    file_path = bahavior_path + str(i) + ".pkl"
    # Read file.
    with open(file_path, "rb") as f:
        data = pickle.load(f)

    # break
    rankings = full_rankings[i][:20]
    # print(data)
    for key, value in data.items():
        if(key == 'interview'):
            continue
        if('recommended_id' not in value.keys()):
            continue
        recommended_id = value['recommended_id']
        watch_id = value['watch_id']
        rating = value['rating']
        rating_id = value['rating_id']
        rating_id_pos_dict = {rating_id[i]: i for i in range(len(rating_id))}
        if(len(rating) != len(rating_id)):
            # print("error")
            continue
        for i in range(len(recommended_id)):
            item_id = int(recommended_id[i])
            # print(item_id)
            if(item_id not in rating_id_pos_dict.keys()):
                one_record = [item_id, i, item_pop[item_id], 1, 0.1, movie_detail.loc[item_id]['rating'], 0, 0]
            else:
                rating_pos = rating_id_pos_dict[item_id]
                one_record = [item_id, i, item_pop[item_id], 1, 0.1, movie_detail.loc[item_id]['rating'], 1, rating[rating_pos]]
            records.loc[len(records)] = one_record

#%%
# If exposure is 1 and rating is 0, then set rating to 3.
records.loc[(records['exposure'] == 1) & (records['click'] == 1) & (records['rating'] == 0), 'rating'] = 3
records.exposure = records.exposure/n_ids
records = records.groupby('item_id').filter(lambda x: len(x) > 9) # 10 score
records
#%%
print(records.groupby("rating").size())
records.groupby("rating").size().plot()

#%%
records['click_rate'] = records['click']
#%%
group_data = records.groupby('item_id').agg({'position': 'mean', 'popularity': 'mean', 'exposure': 'sum', 'propagation': 'mean', 'quality': 'mean', 'click': 'sum', 'click_rate': 'mean', 'rating': 'sum'}).reset_index()
#%%
# group_data
interaction = records.query('rating > 0')
rated = interaction.groupby('item_id')['rating'].mean().reset_index()
group_data = group_data[['item_id', 'position', 'popularity', 'exposure', 'propagation', 'quality', 'click', 'click_rate']]

group_data = pd.merge(group_data, rated, on='item_id', how='left')
group_data = group_data.fillna(0)

#%%
# Remove data with a click rate of 1.
# group_data = group_data[group_data['click_rate'] != 1]

#%%
# Normalize each column of data to 0-1.
# group_data = (group_data - group_data.min()) / (group_data.max() - group_data.min())
# Bishounen transformation
group_data = (group_data - group_data.mean()) / group_data.std()
group_data
#%%
# x = group_data.position
# x = group_data.rating
# x = group_data.popularity
# x = group_data.exposure
# x = group_data.click
# x = group_data.click_rate
# x = group_data.propagation
x = group_data.quality
# y = group_data.popularity
# y = group_data.exposure
# y = group_data.quality
# y = group_data.propagation
# y = group_data.click_rate
# y = group_data.click
y = group_data.rating


plt.scatter(x, y)
# plt.scatter(y, x)

#%%
# index = ['exposure', 'propagation', 'click', 'rating']
# index = ['exposure', 'quality', 'click', 'rating']
# index = ['popularity', 'exposure', 'quality', 'click', 'rating']
# index = ['quality', 'popularity', 'exposure', 'click', 'rating']
# index = ['quality', 'popularity', 'exposure', 'click_rate']
# index = ['quality', 'popularity', 'exposure', 'click_rate', 'rating']
# index = ['quality', 'popularity', 'exposure', 'click', 'click_rate', 'rating']
index = ['quality', 'popularity', 'exposure', 'click', 'rating']
# index = ['quality', 'popularity', 'exposure', 'click_rate', 'rating']
# index = ['quality', 'popularity', 'exposure', 'click', 'rating']
# index = ['exposure', 'propagation', 'quality', 'click', 'rating']
# index = ['exposure', 'propagation', 'quality', 'rating']
# cd_data = cd_data[['exposure', 'propagation', 'quality', 'click', 'rating']]
cd_data = group_data[index]
cd_data


#%%
# Calculate correlation.
corr = cd_data.corr()
corr


#%%
# model = lingam.ICALiNGAM(42, 500)
# prior_knowledge = np.array([
#     [0, 0, 0, 0, 0],
#     [-1, 0, 0, 0, 0],
#     [-1, -1, 0, 0, 0],
#     [-1, -1, -1, 0, 0],
#     [-1, -1, -1, -1, 0]
# ])
# elimilate wrong causality
def prior_graph(n):
    prior_knowledge = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, n):
            if(i > j):
                prior_knowledge[i][j] = -1
    return prior_knowledge
prior_knowledge = prior_graph(len(index))

model = lingam.DirectLiNGAM(101, prior_knowledge, True, 'pwling')
model.fit(cd_data)

print(model.causal_order_)
# print(model.adjacency_matrix_)

adj_matrix = model.adjacency_matrix_

adj_matrix_pd = pd.DataFrame(adj_matrix, index=index, columns=index)
adj_matrix_pd

# %%
# Draw a graph.
import networkx as nx
import matplotlib.pyplot as plt

G = nx.DiGraph()
G.add_nodes_from(index)

count = 0
for i in range(0, len(index)):
    for j in range(0, len(index)):
        if(adj_matrix[i][j] != 0):
            G.add_edge(index[j], index[i])
            count += 1

# Set canvas size.
plt.figure(figsize=(8, 8))

# Draw a figure and add weight values on the edges.
pos = nx.spring_layout(G)
nx.draw_networkx(G, pos, with_labels=True, node_size=1000, node_color='white', edge_color='black', font_size=20, arrows=True)
# %%
