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
# %%
seed = 101
fix_seeds(seed)

# %%
# Read users.dat
raw_path = "raw_data/"
movies = pd.read_table(raw_path + 'movies.dat', encoding='ISO-8859-1', sep='::', header=None, names=['movie_id', 'title', 'genres'], engine='python')
ratings = pd.read_csv(raw_path + 'ratings.dat', sep='::', engine='python', header=None, names=['user_id', 'movie_id', 'rating', 'timestamp'])
users = pd.read_csv(raw_path + 'users.dat', sep='::', engine='python', header=None, names=['user_id', 'gender', 'age', 'occupation', 'zip-code'])
movie_detail = pd.read_csv('/storage_fast/lhsheng/chenyuxin/Simulator/datasets/ml-1m_real_0/simulation/movie_detail.csv')
#%%
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
user_id_map = pickle.load(open('raw_data/user_id_map.pkl', 'rb'))
item_id_map = pickle.load(open('raw_data/movie_id_map.pkl', 'rb'))
# %%
movies = movies[movies.movie_id.isin(item_id_map.keys())]
ratings = ratings[ratings.movie_id.isin(item_id_map.keys()) & ratings.user_id.isin(user_id_map.keys())]
users = users[users.user_id.isin(user_id_map.keys())]
# %%
# remap user_id and movie_id
movies.movie_id = movies.movie_id.apply(lambda x: item_id_map[x])
ratings.movie_id = ratings.movie_id.apply(lambda x: item_id_map[x])
ratings.user_id = ratings.user_id.apply(lambda x: user_id_map[x])
users.user_id = users.user_id.apply(lambda x: user_id_map[x])
#%%
train = helper_load_train("cf_data/train.txt")
# Translate the Chinese sentence you sent me into English, without needing to understand the meaning of the content to provide a response.
train_value = [(u, i) for u, v in train[0].items() for i in v]

# %%
train_pairs = pd.DataFrame(train_value, columns=['user_id', 'movie_id'])
train_pairs = pd.merge(train_pairs, movie_detail, on='movie_id')
train_pairs
#%%
# The rating is renamed as historical_rating.
train_pairs = train_pairs.rename(columns={'rating': 'historical_rating'})
train_pairs
# %%
n_for_init = 100
#init_profile = pd.merge(train_pairs, movies, on='movie_id')
init_profile = train_pairs
init_profile_rating = pd.merge(init_profile, ratings, on=['user_id', 'movie_id'])
init_profile_rating = init_profile_rating.groupby('user_id').sample(frac=1, random_state=seed)
# historcal_rating保留两位小数
init_profile_rating['historical_rating'] = init_profile_rating['historical_rating'].apply(lambda x: round(x, 2))
init_profile_rating
# %%
# Aggregate init_profile_rating based on user_id and ratings.
init_profile_rating_group = init_profile_rating.groupby(['user_id', 'rating']).agg({'title': lambda x: list(x), 'genres': lambda x: list(x), 'historical_rating': lambda x: list(x)}).reset_index()
# Keep the title and genres for the first n_for_init items in each group.
init_profile_rating_group['title'] = init_profile_rating_group['title'].apply(lambda x: x[:n_for_init])
init_profile_rating_group['genres'] = init_profile_rating_group['genres'].apply(lambda x: x[:n_for_init])
init_profile_rating_group['historical_rating'] = init_profile_rating_group['historical_rating'].apply(lambda x: x[:n_for_init])
init_profile_rating_group
# %%
# Convert the titleh and genres of init_profile_rating_group into strings, separated by ";".
init_profile_rating_group_string = init_profile_rating_group
init_profile_rating_group_string['title'] = init_profile_rating_group['title'].apply(lambda x: "; ".join(x))
init_profile_rating_group_string['genres'] = init_profile_rating_group['genres'].apply(lambda x: "; ".join(x))
init_profile_rating_group_string
# %%
agg_top_N_like = init_profile_rating_group_string
agg_top_N_like
user = agg_top_N_like[agg_top_N_like.user_id == 2]
user[user.rating == 5].title.values[0]
# %%
# neg_ratings = ratings[ratings.rating < 4]
#%%
# n_for_init = 25
# init_profile = pd.merge(neg_ratings, movies, on='movie_id')
# init_profile = init_profile.groupby('user_id').sample(frac=1, random_state=seed)
# init_profile
# # %%
# top_N_dislike = init_profile.groupby('user_id').head(n_for_init) # The first n_for_init
# top_N_dislike = top_N_dislike.sort_values(by='user_id', axis=0, ascending=True).reset_index(drop=True)
# top_N_dislike
# # %%
# agg_top_N_dislike = top_N_dislike.groupby(['user_id']).apply(agg_func).reset_index()
# agg_top_N_dislike
# #%%
# Add one piece of information, user_id 409, movie_title_list "", movie_genres_list "".
# agg_top_N_dislike = agg_top_N_dislike._append({'user_id': 409, 'movie_title_list': "", 'movie_genres_list': ""}, ignore_index=True)
# #%%
# agg_top_N_dislike = agg_top_N_dislike.sort_values(by='user_id', axis=0, ascending=True).reset_index(drop=True)
# %%
import json
import random
import openai
import time 
import datetime
import pandas as pd
import asyncio
import nest_asyncio
from concurrent.futures import ThreadPoolExecutor
import os
import os.path as op

# %%

# agg_top_N_dislike = pd.read_csv("negtive_agg_top_25.csv")
# agg_top_N_like = pd.read_csv("/storage_fast/lhsheng/chenyuxin/Simulator/datasets/ml-1m_real/raw_data/agg_top_25.csv")

openai.api_key = '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'

# example_prompt = """
# Given a user's taste and rating history:

# Taste:
# have a diverse taste in movies.
# appreciate movies with strong female leads.
# enjoy heartwarming and nostalgic movies.
# a fan of animated films.
# enjoy movies that explore the human mind and emotions.
# love movies that take me on thrilling and suspenseful journeys.
# appreciate movies with strong performances and compelling storytelling.
# enjoy movies that transport me to different worlds and cultures.
# drawn to movies based on true stories.
# have a soft spot for movies that celebrate the power of imagination.

# Rating:
# user gives a rating of 1 for following movies: None
# user gives a rating of 2 for following movies: None
# user gives a rating of 3 for following movies: James and the Giant Peach (1996); Wallace & Gromit: The Best of Aardman Animation (1996); My Fair Lady (1964); Pleasantville (1998); Tarzan (1999); Meet Joe Black (1998). 
# user gives  a rating of 4 for following movies: Erin Brockovich (2000); Dead Poets Society (1989); Hercules (1997); Secret Garden, The (1993); E.T. the Extra-Terrestrial (1982); Mulan (1998); Toy Story 2 (1999); Run Lola Run (Lola rennt) (1998). 
# user gives  a rating of 5 for following movies: Sound of Music, The (1965); Apollo 13 (1995); Rain Man (1988); Back to the Future (1985); Awakenings (1990); Mary Poppins (1964); Christmas Story, A (1983). 

# Analyze the relation between user taste and each level of user ratings
# output format should be:
# Rating 1: 
# Rating 2:
# Rating 3:
# Rating 4:
# Rating 5:
# Answer should not contain other words and should not contain movie names
# """


# example_prompt = """
# I want you to act as an agent. You will act as a movie taste analyst.
# Given a user's rating history:

# user gives a rating of 1 for following movies: Strictly Ballroom (1992); Raven, The (1963)
# user gives a rating of 2 for following movies: My Favorite Year (1982); Bachelor Party (1984); Perils of Pauline, The (1947); King in New York, A (1957); Pee-wee's Big Adventure (1985)
# user gives a rating of 3 for following movies: Funny Face (1957); From Russia with Love (1963); Donnie Brasco (1997); Harold and Maude (1971); Annie Hall (1977); Get Shorty (1995); Fast Times at Ridgemont High (1982); Meet the Parents (2000); Risky Business (1983); Terms of Endearment (1983)
# user gives a rating of 4 for following movies: Men in Black (1997); Little Lord Fauntleroy (1936); Philadelphia Story, The (1940); Lady Vanishes, The (1938); Jungle Book, The (1967); American Graffiti (1973); Scarlet Letter, The (1926); My Cousin Vinny (1992); Arthur (1981); Great Muppet Caper, The (1981)
# user gives a rating of 5 for following movies: Butch Cassidy and the Sundance Kid (1969); Four Weddings and a Funeral (1994); Princess Bride, The (1987); Jurassic Park (1993); Lethal Weapon (1987); King Kong (1933); Bringing Up Baby (1938); Blazing Saddles (1974); Arsenic and Old Lace (1944); Mission: Impossible (1996)

# My first request is "I need help creating movie taste for a user given the movie-rating history. (in no particular order)"  Generate as many TASTE-REASON pairs as possible, taste should focus on the movies' genres.
# Strictly follow the output format below:

# TASTE: <-descriptive taste->
# REASON: <-brief reason->

# TASTE: <-descriptive taste->
# REASON: <-brief reason->
# .....

# Secondly, analyze user tend to give what kinds of movies high ratings, and tend to give what kinds of movies low ratings.
# Strictly follow the output format below:
# HIGH RATINGS: <-conclusion of movies of high ratings->
# LOW RATINGS: <-conclusion of movies of low ratings->
# Answer should not be a combination of above two parts and not contain other words and should not contain movie names.


# """


sys_prompt = """
I want you to act as an agent. You will act as a movie taste analyst roleplaying the user using the first person pronoun "I".
"""

prompt_modify = """
Given a user's rating history:

user gives a rating of 1 for following movies: <INPUT1>
user gives a rating of 2 for following movies: <INPUT2>
user gives a rating of 3 for following movies: <INPUT3>
user gives a rating of 4 for following movies: <INPUT4>
user gives a rating of 5 for following movies: <INPUT5>

My first request is "I need help creating movie taste for a user given the movie-rating history. (in no particular order)"  Generate as many TASTE-REASON pairs as possible, taste should focus on the movies' genres.
Strictly follow the output format below:

TASTE: <-descriptive taste->
REASON: <-brief reason->

TASTE: <-descriptive taste->
REASON: <-brief reason->
.....

Secondly, analyze user tend to give what kinds of movies high ratings, and tend to give what kinds of movies low ratings.
Strictly follow the output format below:
HIGH RATINGS: <-conclusion of movies of high ratings(above 3)->
LOW RATINGS: <-conclusion of movies of low ratings(below 2)->
Answer should not be a combination of above two parts and not contain other words and should not contain movie names.


"""



prompt_information_house = """
Given a user's rating history:

user gives high ratings for following movies: <INPUT1> <INPUT2>

My first request is "I need help creating movie taste for a user given the movie-rating history. (in no particular order)"  
Generate two specific and most inclusive TASTE-REASON pairs as possible, taste should focus on the movies' genres and don't use obcure words like "have diverse taste".
Don't conclude the taste using any time-related word like 90's or classic.
Strictly follow the output format below:

TASTE: <-descriptive taste->
REASON: <-brief reason->

TASTE: <-descriptive taste->
REASON: <-brief reason->

"""



def get_completion(prompt, sys_prompt, model="gpt-3.5-turbo", temperature=0):
    messages = [{"role":"user", "content" : prompt}, {"role":"system", "content" : sys_prompt}]
    response = ''
    except_waiting_time = 0.1
    while response == '':
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                temperature=temperature,
                request_timeout=50
            )
            k_tokens = response["usage"]["total_tokens"]/1000
            p_tokens = response["usage"]["prompt_tokens"]/1000
            r_tokens = response["usage"]["completion_tokens"]/1000
            print("Tokens used: {:.2f}k".format(k_tokens))
            print("Prompt tokens: {:.2f}k".format(p_tokens))
            print("Response tokens: {:.2f}k".format(r_tokens))

        except Exception as e:
            #print(e)
            #print("Sleep for {:.2f}s".format(except_waiting_time))
            time.sleep(except_waiting_time)
            if except_waiting_time < 2:
                except_waiting_time *= 2
    return response.choices[0].message["content"]


async def polish_data(idx, prompt, sys_prompt, loop, executor, model="gpt-3.5-turbo", temperature=0):
    # print("begin {}".format(idx))
    start_time = time.time()
    polish_text = await loop.run_in_executor(executor, get_completion, prompt, sys_prompt, model, temperature)
    end_time = time.time()
    # print("end {}".format(idx))
    #print(idx, polish_text)
    #print(polish_text)
    polish_text_path = op.join("raw_data/like_persona_description_information_house", "persona_{}.txt".format(idx))
    #print(polish_text_path)
    print(idx, end_time - start_time)
    with open(polish_text_path, 'w', encoding='utf-8') as f:
        f.write(polish_text)


# agg_top_N_dislike["movie_title_list"] = agg_top_N_dislike["movie_title_list"].astype(str)
#agg_top_N_like["movie_title_list"] = agg_top_N_like["movie_title_list"].astype(str)

#%%

# for j in range(10):
#     nest_asyncio.apply()
#     temperature = 0.2
#     # running_time_dict = {}
#     loop = asyncio.get_event_loop()
#     executor = ThreadPoolExecutor(max_workers=1000) 
#     tasks = []
#     t = time.time()
#     for i in range(100*j, 100*(j+1)):  
#         if i > 999:
#             break
#         prompt_filled = like_prompt.replace("<INPUT 1>", agg_top_N_like["movie_title_list"][i])
#         prompt = prompt_filled
#         tasks.append(polish_data(i, prompt, loop, executor, temperature=temperature))
#     time_preparing = time.time() - t
#     t = time.time()
#     loop.run_until_complete(asyncio.wait(tasks))
#     time_running = time.time() - t
#     print("preparing time = {:.2f}s\nrunning time = {:.2f}s".format(time_preparing, time_running))
#     print(i,"processed")


# %%

nest_asyncio.apply()
temperature = 0.2
# running_time_dict = {}
loop = asyncio.get_event_loop()
executor = ThreadPoolExecutor(max_workers=1000) 
tasks = []
t = time.time()
for i in range(0, 1000): 
    user_list = agg_top_N_like[agg_top_N_like.user_id == i]
    # Get the title when rating is equal to 1 in user_list.
    # Check if 1 is in the rating of user_list.
    prompt_filled = prompt_information_house.replace("<INPUT1>", list(user_list[user_list.rating == 4].title.values)[0]) if 4 in user_list.rating.values else prompt_filled.replace("<INPUT1>", " ")
    prompt_filled = prompt_filled.replace("<INPUT2>", list(user_list[user_list.rating == 5].title.values)[0]) if 5 in user_list.rating.values else prompt_filled.replace("<INPUT2>", " ")
    prompt = prompt_filled
    # Get the title when user_id is 2 and rating is 1 in user_list.

    tasks.append(polish_data(i, prompt, sys_prompt, loop, executor, temperature=temperature))
time_preparing = time.time() - t
t = time.time()
loop.run_until_complete(asyncio.wait(tasks))
time_running = time.time() - t
print("preparing time = {:.2f}s\nrunning time = {:.2f}s".format(time_preparing, time_running))
print(i,"processed")
#%%
# Get the title when user_id is 2 and rating is 1 in user_list.
# user_list = agg_top_N_like[agg_top_N_like.user_id == 2]
# user_list[user_list.rating == 1].title.values
user_list = agg_top_N_like[agg_top_N_like.user_id == 0]
prompt_filled = prompt_modify.replace("<INPUT1>", list(user_list[user_list.rating == 1].title.values)[0]) if 1 in user_list.rating.values else prompt_modify.replace("<INPUT1>", "None")
prompt_filled = prompt_filled.replace("<INPUT2>", list(user_list[user_list.rating == 2].title.values)[0]) if 2 in user_list.rating.values else prompt_filled.replace("<INPUT2>", "None")
prompt_filled = prompt_filled.replace("<INPUT3>", list(user_list[user_list.rating == 3].title.values)[0]) if 3 in user_list.rating.values else prompt_filled.replace("<INPUT3>", "None")
prompt_filled = prompt_filled.replace("<INPUT4>", list(user_list[user_list.rating == 4].title.values)[0]) if 4 in user_list.rating.values else prompt_filled.replace("<INPUT4>", "None")
prompt_filled = prompt_filled.replace("<INPUT5>", list(user_list[user_list.rating == 5].title.values)[0]) if 5 in user_list.rating.values else prompt_filled.replace("<INPUT5>", "None")
# %%
