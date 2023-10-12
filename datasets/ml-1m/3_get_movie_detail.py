#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import torch
import pickle


def fix_seeds(seed=101):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # In order to disable hash randomization and make the experiment reproducible.
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
# %%
# Fixed seed
seed = 101
fix_seeds(seed)

# %%
# Read users.dat
raw_path = "raw_data/"
movies = pd.read_table(raw_path + 'movies.dat', encoding='ISO-8859-1', sep='::', header=None, names=['movie_id', 'title', 'genres'], engine='python')
ratings = pd.read_csv(raw_path + 'ratings.dat', sep='::', engine='python', header=None, names=['user_id', 'movie_id', 'rating', 'timestamp'])
users = pd.read_csv(raw_path + 'users.dat', sep='::', engine='python', header=None, names=['user_id', 'gender', 'age', 'occupation', 'zip-code'])
# %%
# average rating for each movie
avg_ratings = ratings[['movie_id', 'rating']].groupby("movie_id").mean().reset_index()
movies_rating = pd.merge(movies, avg_ratings, on='movie_id', how='left')

#%%
aug_movie = pd.read_csv(raw_path + 'movies_augmentation.csv')
aug_movie.rating = movies_rating.rating

# %%
item_id_map = pickle.load(open('raw_data/movie_id_map.pkl', 'rb'))
# %%
aug_movie = aug_movie[aug_movie.movie_id.isin(item_id_map.keys())]
# remap movie_id
aug_movie.movie_id = aug_movie.movie_id.apply(lambda x: item_id_map[x])
# %%
aug_movie.to_csv('simulation/movie_detail.csv', index=False)
# %%
