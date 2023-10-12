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
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
# %%
# 固定seed
seed = 101
fix_seeds(seed)

# %%
# 读取users.dat
raw_path = "raw_data/"
movies = pd.read_table(raw_path + 'movies.dat', encoding='ISO-8859-1', sep='::', header=None, names=['movie_id', 'title', 'genres'], engine='python')
ratings = pd.read_csv(raw_path + 'ratings.dat', sep='::', engine='python', header=None, names=['user_id', 'movie_id', 'rating', 'timestamp'])
users = pd.read_csv(raw_path + 'users.dat', sep='::', engine='python', header=None, names=['user_id', 'gender', 'age', 'occupation', 'zip-code'])

# %%
avg_num_movies = ratings.groupby('user_id').size().reset_index(name='activity_num')
percentile = np.percentile(avg_num_movies['activity_num'], [60, 90, 100])
# avg_num_movies['activity'] = pd.qcut(avg_num_movies['activity_num'], 3, labels=False) + 1
def activity(x):
    if x < percentile[0]:
        return 1
    elif x < percentile[1]:
        return 2
    else:
        return 3
avg_num_movies['activity'] = avg_num_movies['activity_num'].apply(activity)
#%%
#@ activity
# group by user_id and the first four characters of timestamp
# ratings['timestamp'] = ratings['timestamp'].apply(lambda x: str(x)[:5])
# ratings = ratings.groupby(['user_id', 'timestamp']).agg({'movie_id': lambda x: list(x), 'rating': lambda x: list(x)}).reset_index()
# ratings
# # %%
# ratings['num_movies'] = ratings['movie_id'].apply(lambda x: len(x))
# ratings
# # %%
# # group by user_id, save the average value of num_movies for each user_id
# ratings = ratings.groupby(['user_id']).agg({'movie_id': lambda x: list(x), 'rating': lambda x: list(x), 'num_movies': lambda x: list(x)}).reset_index()
# ratings
# # %%
# ratings['avg_num_movies'] = ratings['num_movies'].apply(lambda x: sum(x) / len(x))
# ratings['avg_num_movies']
# ratings
# #%%
# avg_num_movies = ratings['avg_num_movies'].values
# avg_num_movies
# # %%
# # 根据分位点将avg_num_movies分为3类
# percentile = np.percentile(avg_num_movies, [33, 66, 100])
# percentile 
# # %%
# ratings['activity'] = pd.qcut(ratings['avg_num_movies'], 3, labels=False) + 1
#%%
# 根据不同分类用不同颜色绘制直方图
plt.hist(avg_num_movies['activity_num'], bins=100, color='steelblue')
plt.axvline(percentile[0], color='red')
plt.axvline(percentile[1], color='red')
plt.axvline(percentile[2], color='red')
plt.show()
#%%
statistics = avg_num_movies[['user_id', 'activity']]
statistics_num = avg_num_movies[['user_id', 'activity_num']]
# %%


# %%
raw_path = "raw_data/"
movies = pd.read_table(raw_path + 'movies.dat', encoding='ISO-8859-1', sep='::', header=None, names=['movie_id', 'title', 'genres'], engine='python')
ratings = pd.read_csv(raw_path + 'ratings.dat', sep='::', engine='python', header=None, names=['user_id', 'movie_id', 'rating', 'timestamp'])
users = pd.read_csv(raw_path + 'users.dat', sep='::', engine='python', header=None, names=['user_id', 'gender', 'age', 'occupation', 'zip-code'])
# %%
# merge movies and ratings on movie_id
movies_ratings = pd.merge(movies, ratings, on='movie_id', how='inner')
movies_ratings
# %%
def count_genres(x):
    genres = x['genres'].str.split('|')
    genres = genres.explode()
    return genres.value_counts()

genres_count = movies_ratings.groupby('user_id').apply(count_genres).reset_index(name='count')
# %%
genres_count
# %%
# 计算每个用户每个电影种类的比例
genres_count['ratio'] = genres_count.groupby('user_id')['count'].transform(lambda x: x / x.sum())
genres_count['cum_ratio'] = genres_count.groupby('user_id')['ratio'].transform(lambda x: x.cumsum())
# 根据ratio按照降序排序
genres_count = genres_count.sort_values(by=['user_id', 'ratio'], ascending=[True, False]).reset_index(drop=True)
genres_count
# %%
genres_count.query("user_id == 1")
# %%
top_percent = 0.8
genres_count_top = genres_count[genres_count['cum_ratio'] <= top_percent]
# %%
genres_diversity = genres_count_top.groupby('user_id').size().reset_index(name='diversity_num')
# %%
percentile = np.percentile(genres_diversity.diversity_num, [33, 66, 100])
percentile 
#%%
# 根据不同分类用不同颜色绘制直方图
plt.hist(genres_diversity.diversity_num, bins=10, color='steelblue')
plt.axvline(percentile[0], color='red')
plt.axvline(percentile[1], color='red')
plt.axvline(percentile[2], color='red')
plt.show()
#%%
# 根据diversity num值为5, 6, 7进行分档
def diversity(x):
    if x < percentile[0]:
        return 1
    elif x < percentile[1]:
        return 2
    else:
        return 3
genres_diversity['diversity'] = genres_diversity['diversity_num'].apply(diversity)
# %%
statistics = pd.merge(statistics, genres_diversity[['user_id', 'diversity']], on='user_id', how='inner')
statistics_num = pd.merge(statistics_num, genres_diversity[['user_id', 'diversity_num']], on='user_id', how='inner')
# %%
#@ conformity
raw_path = "raw_data/"
movies = pd.read_table(raw_path + 'movies.dat', encoding='ISO-8859-1', sep='::', header=None, names=['movie_id', 'title', 'genres'], engine='python')
ratings = pd.read_csv(raw_path + 'ratings.dat', sep='::', engine='python', header=None, names=['user_id', 'movie_id', 'rating', 'timestamp'])
users = pd.read_csv(raw_path + 'users.dat', sep='::', engine='python', header=None, names=['user_id', 'gender', 'age', 'occupation', 'zip-code'])
# %%
# average rating for each movie
avg_ratings = ratings[['movie_id', 'rating']].groupby("movie_id").mean().reset_index()
avg_ratings.rename(columns={'rating':'avg_rating'}, inplace=True)
movies_rating = pd.merge(movies, avg_ratings, on='movie_id', how='inner')
# %%
# 统计信息
ratings_with_avg = pd.merge(ratings, avg_ratings, on = 'movie_id') # 每个pair的评分
ratings_with_avg['mse'] = (ratings_with_avg['rating'] - ratings_with_avg['avg_rating'])**2
ratings_with_avg
#%%
percentile = np.percentile(ratings_with_avg['mse'], [50, 90, 100])
plt.hist(ratings_with_avg['mse'], bins=100, color='steelblue')
plt.axvline(percentile[0], color='red')
plt.axvline(percentile[1], color='red')
plt.axvline(percentile[2], color='red')
plt.show()

#%%
# 计算用户的评分和平均的评分的偏离度
deviation = ratings_with_avg.groupby('user_id').mse.mean().reset_index(name='deviation')
# 计算deviation的5个分位数
# deviation['conformity'] = pd.qcut(deviation['deviation'], 3, labels=False) + 1
#%%
percentile = np.percentile(deviation['deviation'], [25, 80, 25])
percentile
#%%
plt.hist(deviation['deviation'], bins=100, color='steelblue')
plt.axvline(percentile[0], color='red')
plt.axvline(percentile[1], color='red')
plt.axvline(percentile[2], color='red')
plt.show()
#%%
def conformity(x):
    if x < percentile[0]:
        return 1
    elif x < percentile[1]:
        return 2
    else:
        return 3
deviation['conformity'] = deviation['deviation'].apply(conformity)
# %%
statistics = pd.merge(statistics, deviation[['user_id', 'conformity']], on='user_id', how='inner')
statistics_num = pd.merge(statistics_num, deviation[['user_id', 'deviation']], on='user_id', how='inner')
# %%
statistics


#%%
# 读取'raw_data/user_id_map.pkl'
user_id_map = pickle.load(open('raw_data/user_id_map.pkl', 'rb'))
# item_id_map = pickle.load(open('raw_data/movie_id_map.pkl', 'rb'))
#%%
statistics_remap = statistics[statistics.user_id.isin(user_id_map.keys())]
statistics_remap['user_id'] = statistics_remap['user_id'].apply(lambda x: user_id_map[x])
# %%
statistics_remap.to_csv("simulation/user_statistic.csv", index=False)

# %%
statistics_num_remap = statistics_num[statistics_num.user_id.isin(user_id_map.keys())]
statistics_num_remap['user_id'] = statistics_num_remap['user_id'].apply(lambda x: user_id_map[x])

# %%
statistics_num_remap.to_csv("simulation/user_statistic_num.csv", index=False)
# %%
