#%%
import re
import pandas as pd
import os.path as op
import os

#%%
def generate_init_info(s):
    taste = re.findall(r'TASTE:(.+)', s)
    reason = re.findall(r'REASON:(.+)', s)
    # high_rating = re.findall(r'HIGH RATINGS:(.+)', s)
    # low_rating = re.findall(r'LOW RATINGS:(.+)', s)
    # return taste, reason, movie
    return "| ".join(taste), "| ".join(reason)#, "| ".join(high_rating), "| ".join(low_rating)

base_path = "raw_data/like_persona_description_information_house"

# Get all file names under the folder.
len_file_names = len(sorted(os.listdir(base_path)))
file_names = ["persona_"+str(i)+".txt" for i in range(len_file_names)]

# df = pd.DataFrame(index=range(len(file_names)), columns=["avatar_name", "age", "occupation", "traits", "description"])

#df = pd.DataFrame(index=range(len(file_names)), columns=["taste", "reason", "high_rating", "low_rating"])
df = pd.DataFrame(index=range(len(file_names)), columns=["taste", "reason"])

#%%
avatars_info = {}
for idx, file_name in enumerate(file_names):
    with open(base_path + "/" + file_name, "r") as f:
        persona = f.read()
    # taste, reason, high_rating, low_rating = generate_init_info(persona)
    taste, reason = generate_init_info(persona)
    avatars_info[idx] = {
        "taste": taste,
        "reason": reason,
        # "high_rating": high_rating,
        # "low_rating": low_rating
    }
    print(idx)

    # df.loc[idx] = [taste, reason, high_rating, low_rating]
    df.loc[idx] = [taste, reason]
    # break

#%%
df.to_csv("simulation/all_personas_like_information_house.csv", index=False)
# %%
