import pickle
import numpy as np
import os


data_path = 'data_plus_plus/'


with open(data_path + 'halfcheetah-expert-v2.pkl', 'rb') as f:
    expert = pickle.load(f)

with open(data_path + 'halfcheetah-medium-v2.pkl', 'rb') as f:
    medium = pickle.load(f)


# get half the dataset of each
np.random.seed(42069)
expert = list(np.random.choice(expert, 500, replace=False))
medium = list(np.random.choice(medium, 500, replace=False))

# shuffle the data
expert_medium = list(np.random.choice(expert + medium, 1000, replace=False))

outfile = open(data_path + 'halfcheetah-expert_medium_split-v2.pkl','wb')
pickle.dump(expert_medium, outfile)
outfile.close()