import pickle
import os
import numpy as np
from sklearn.linear_model import LinearRegression

datasets = ['halfcheetah-expert-v2.pkl',
            'halfcheetah-medium-v2.pkl',
            'halfcheetah-medium-replay.pkl',
            'hopper-expert-v2.pkl',
            'hopper-medium-v2.pkl',
            'hopper-medium-replay-v2.pkl',
            'walker2d-expert-v2.pkl',
            'walker2d-medium-v2.pkl',
            'walker2d-medium-replay-v2.pkl']

data_string = datasets[0]

with open('gym\\data\\' + data_string, 'rb') as f:
    trajectories = pickle.load(f)


dt = 1


rewards = []
dt = 0.05
coef = []
score = []
for i, traj in enumerate(trajectories):

    forward = traj['observations'][:,13]
    alive = np.array(~traj['terminals'], dtype=int)
    control = np.square(traj['actions'][:]).sum(axis = 1)
    reward = traj['rewards']
    
    X = np.stack((forward, control, alive), axis=0).T

    reg = LinearRegression().fit(X, reward)

    r_score = reg.score(X, reward)
    
    coef.append(reg.coef_)
    score.append(r_score)

coef = np.array(coef)
print(score)
print(coef.mean(axis=0))

# rewards = np.array(rewards)

# print(np.array(rewards[0]).shape)
# print(trajectories[:]['rewards'])

# def get_coef(data_temp):

#     forward_reward = data_temp['observations'][:,5]
#     ctrl_cost = (data_temp['actions'] ** 2).sum(axis=1)
#     healthy_reward = np.array(~data_temp['terminals'], dtype=int)

#     target_reward = data_temp['rewards']


#     X = np.stack((forward_reward, ctrl_cost, healthy_reward), axis=0).T

#     reg = LinearRegression().fit(X, target_reward)

#     r_score = reg.score(X, target_reward)
    
#     return reg.coef_, r_score # (forward_reward_weight, ctrl_cost_weight, healthy_reward), r_score

# print(get_coef(trajectories[0]))
