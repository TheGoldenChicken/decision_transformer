#%% 
# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
from tqdm import tqdm
from sklearn.linear_model import LinearRegression

# %%
""" Loading dataset """
train_data = dict()

envs = ['halfcheetah', 'hopper', 'walker2d']
datasets = ['expert', 'medium_replay', 'medium']
data_features = ['observations', 'next_observations', 'actions', 'rewards', 'terminals']

for e in envs:
    for d in datasets:
        with open(f'./data/{e}-{d}-v2.pkl', 'rb') as f:
            train_data[f'{e}-{d}'] = pickle.load(f)
            

# %%
""" Define functions got getting multiple reward coefficients """

# Calculating forward reward
def get_forward_reward(positions):
    return positions[1:] - positions[:-1]

# Calculating control cost
def get_control_cost(actions):
    return np.sum(np.square(actions), axis=1)

# calculate the weights of the individual parts of the reward using linear regression
def get_coef(data_temp, use_healthy_reward=False, use_intercept=False):
    
    forward_reward = get_forward_reward(data_temp['infos/qpos'][:,0])
    ctrl_cost = get_control_cost(data_temp['actions'])

    target_reward = data_temp['rewards'][:-1]

    if use_healthy_reward: 
        healthy_reward = np.array(~data_temp['terminals'], dtype=int)
        X = np.stack((forward_reward, ctrl_cost[:-1], healthy_reward[:-1]), axis=0).T
    else:
        X = np.stack((forward_reward, ctrl_cost[:-1]), axis=0).T

    # Do linear regression
    reg = LinearRegression(fit_intercept=False).fit(X, target_reward)
    
    return reg, X, target_reward

# %%
""" Perform linear regression and plot residuals """


data_temp = train_data['halfcheetah-medium_replay'] # Get dataset

results = []
for e, i in enumerate(data_temp):
    reg, X, target_reward = get_coef(i, use_healthy_reward=True, use_intercept=False) # Do linear regression
    results.append([*reg.coef_, reg.score(X, target_reward),5])
    predictions = reg.predict(X)
    plt.scatter(predictions, predictions - target_reward, c='C0')

print("Mean: (Forward, control, healthy, score):",np.mean(results,axis=0)) # Print coefficients

print("Last: (Forward, control, healthy, score):",results[-1]) # Print coefficients

# Do prediction

plt.hlines(0, min(predictions), max(predictions), linestyles='--', colors='C1')
plt.xlabel("Prediction")
plt.ylabel("Error")
plt.title("Raw Trajectories")
plt.show()

# # mask for points with low error
# tolerance = 0.002
# mask = abs(predictions - target_reward ) < tolerance

# data_masked = data_temp
# reg, X, target_reward = get_coef(data_masked, use_healthy_reward=True, use_intercept=False) # Do linear regression

# print("(Forward, control, healthy):",reg.coef_,"| Score:" ,reg.score(X, target_reward)) # Print coefficients

# plt.scatter(predictions, predictions - target_reward, c='C0')
# plt.xlabel("Prediction")
# plt.ylabel("Error")
# plt.title("Removed data with error > 0.002")
# plt.hlines(0, min(predictions), max(predictions), linestyles='--', colors='C1')
# plt.show()


#%%
data_temp = train_data['hopper-medium_replay'] # Get dataset

for i, e in enumerate(data_temp):
    if len(data_temp[i]["actions"]) != len(data_temp[i]["rewards"]):
        print(i)