
#%%

import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
from tqdm import tqdm

#%%

env = gym.make('Hopper-v3')

train_data = dict()
envs = ['halfcheetah', 'hopper', 'walker2d']
datasets = ['expert', 'medium_replay', 'medium']
data_features = ['observations', 'next_observations', 'actions', 'rewards', 'terminals']

for e in envs:
    for d in datasets:
        with open(f'./{e}-{d}-v2.pkl', 'rb') as f:
            train_data[f'{e}-{d}'] = pickle.load(f)


# %%

data_set = train_data['hopper-expert']

for data in data_set:
    
    init_state = data['observations'][0]
    action_generator = (action for action in data['actions'])







# %%
