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

def describe_interactions(df):
    print('number of users: ', len(df.user_id.unique()))
    print('number of movies: ', len(df.movie_id.unique()))
    print('number of interactions: ', len(df))
    print('max user id', df.user_id.max())
    print('max video id', df.movie_id.max())
    print(' ')

def int_to_user_dict(interaction):
    """
    convert a list of interactions into a dictionary 
    that maps each user to a list of their interactions
    input: df with columns ['user_id', 'movie_id']
    output: dict with key: user_id, value: list of movie_id
    """
    user_dict = {}
    for u, v in interaction:
        if(u not in user_dict.keys()):
            user_dict[u] = [v]
        else:
            user_dict[u].append(v)
    # Sort according to key.
    user_dict = dict(sorted(user_dict.items(), key=lambda x: x[0]))
    return user_dict

def save_user_dict_to_txt(user_dict, base_path, filename):
    with open(base_path + filename, 'w') as f:
        for u, v in user_dict.items():
            f.write(str(int(u)))
            for i in v:
                f.write(' ' + str(int(i)))
            f.write('\n')
#%%
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
# Take the sentences with a rating greater than 3 as positive samples.
pairs = ratings[ratings['rating'] > 3]
pairs
# %%
# Users with more than 20 interactions.
filter_gate = 20
int_per_user = pairs.groupby('user_id').size().reset_index(name='counts')
filtered_users = int_per_user[int_per_user['counts'] >= filter_gate]
filtered_users
#%%
# Only take interaction data from users with more than 20 interactions.
pairs = pairs[pairs['user_id'].isin(filtered_users['user_id'])]
pairs = pairs.reset_index(drop=True)[['user_id', 'movie_id', 'rating']]
pairs = pairs.sample(frac=1, random_state=seed).reset_index(drop=True)
pairs = pairs.sort_values(by='user_id', axis=0, ascending=True).reset_index(drop=True)
pairs

#%%
# Randomly select 1000 users.
random_users = pairs.user_id.unique()
random_users = np.random.choice(random_users, size=1000, replace=False)
#pairs = pairs[pairs['user_id'].isin(random_users)]
pairs = ratings[ratings['user_id'].isin(random_users)]
# %%
pairs
# %%
# train valid test is divided into 4:3:3
train_pairs = pairs.groupby('user_id').sample(frac=0.4, random_state=seed)
valid_pairs = pairs[~pairs.index.isin(train_pairs.index)]
valid_pairs = valid_pairs.groupby('user_id').sample(frac=0.5, random_state=seed)
test_pairs = pairs[~(pairs.index.isin(train_pairs.index.append(valid_pairs.index)))]
print(len(train_pairs)/len(pairs), len(valid_pairs)/len(pairs), len(test_pairs)/len(pairs))

#%%
pos_items_train = train_pairs.movie_id.unique()
pos_users_train = train_pairs.user_id.unique()

# Only select the user and item that appear in the train.
valid_pairs = valid_pairs[valid_pairs.user_id.isin(pos_users_train) & valid_pairs.movie_id.isin(pos_items_train)]
test_pairs = test_pairs[test_pairs.user_id.isin(pos_users_train) & test_pairs.movie_id.isin(pos_items_train)]
print(len(train_pairs)/len(pairs), len(valid_pairs)/len(pairs), len(test_pairs)/len(pairs))

movies = movies[movies.movie_id.isin(pos_items_train)].reset_index(drop=True)
users = users[users.user_id.isin(pos_users_train)].reset_index(drop=True)

#%%
# Remap user_id and movie_id.
user_id_map = {}
movie_id_map = {}

for i, u in enumerate(sorted(pos_users_train)):
    user_id_map[u] = i

for i, v in enumerate(sorted(pos_items_train)):
    movie_id_map[v] = i

train_pairs['user_id'] = train_pairs['user_id'].map(user_id_map)
train_pairs['movie_id'] = train_pairs['movie_id'].map(movie_id_map)

valid_pairs['user_id'] = valid_pairs['user_id'].map(user_id_map)
valid_pairs['movie_id'] = valid_pairs['movie_id'].map(movie_id_map)

test_pairs['user_id'] = test_pairs['user_id'].map(user_id_map)
test_pairs['movie_id'] = test_pairs['movie_id'].map(movie_id_map)

movies['movie_id'] = movies['movie_id'].map(movie_id_map)
users['user_id'] = users['user_id'].map(user_id_map)

#%%
# Save user_id_map using pickle.
with open('raw_data/user_id_map.pkl', 'wb') as f:
    pickle.dump(user_id_map, f)
with open('raw_data/movie_id_map.pkl', 'wb') as f:
    pickle.dump(movie_id_map, f)

#%%
describe_interactions(train_pairs)
describe_interactions(valid_pairs)
describe_interactions(test_pairs)

plt.figure(figsize=(30, 10))
plt.scatter(train_pairs.user_id, train_pairs.movie_id, s=0.1)

# %%
train_user_dict = int_to_user_dict(train_pairs.values[:,0:2])
valid_user_dict = int_to_user_dict(valid_pairs.values[:,0:2])
test_user_dict = int_to_user_dict(test_pairs.values[:,0:2])
#%%

# %%
base_path = 'cf_data/'
if not os.path.exists(base_path):
    os.makedirs(base_path)
else:
    # Remove all files in the directory.
    files = os.listdir(base_path)
    for file in files:
        os.remove(base_path + file)

save_user_dict_to_txt(train_user_dict, base_path, 'train.txt')
save_user_dict_to_txt(valid_user_dict, base_path, 'valid.txt')
save_user_dict_to_txt(test_user_dict, base_path, 'test.txt')

#%%
n_for_init = 25
init_profile = pd.merge(train_pairs, movies, on='movie_id')
init_profile = init_profile.groupby('user_id').sample(frac=1, random_state=seed)
init_profile
#%%
top_N_like = init_profile.groupby('user_id').head(n_for_init) # The first n_for_init
top_N_like['rating'] = top_N_like['rating'].astype(str)
top_N_like = top_N_like.sort_values(by='user_id', axis=0, ascending=True).reset_index(drop=True)
top_N_like

#%%
def agg_func(x):
    return pd.Series({
        "movie_title_list": "; ".join((x["title"])),
        "movie_genres_list": "; ".join((x["genres"])),
        "rating_list":"; ".join((x["rating"])),
    })

agg_top_N_like = top_N_like.groupby(['user_id']).apply(agg_func).reset_index()
agg_top_N_like
# %%
gender_dict = {
    'F': 'Female', 'M': 'Male'
}
age_dict = {
    1:  "Under 18", 18:  "18-24", 25:  "25-34", 35:  "35-44", 45:  "45-49", 50:  "50-55", 56:  "56+"
}
occupation_dict = {
    0:  "other", 1:  "academic/educator", 2:  "artist", 3:  "clerical/admin", 4:  "college/grad student", 5:  "customer service",
    6:  "doctor/health care", 7:  "executive/managerial", 8:  "farmer", 9:  "homemaker", 10:  "K-12 student", 11:  "lawyer",
    12:  "programmer", 13:  "retired", 14:  "sales/marketing", 15:  "scientist", 16:  "self-employed",
    17:  "technician/engineer", 18:  "tradesman/craftsman", 19:  "unemployed", 20:  "writer"
}
# %%
users['occupation'] = users['occupation'].apply(lambda x: occupation_dict[x])
users['age'] = users['age'].apply(lambda x: age_dict[x])
users['gender'] = users['gender'].apply(lambda x: gender_dict[x])
# %%
users
# %%
agg_top_N_like = pd.merge(agg_top_N_like, users, on='user_id')
agg_top_N_like

#%%
agg_top_N_like.to_csv(f'raw_data/agg_top_{n_for_init}.csv', index=False)
