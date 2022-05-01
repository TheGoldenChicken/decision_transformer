import pickle
import os
import numpy as np

datasets = ['halfcheetah-expert-v2.pkl',
            'halfcheetah-medium-v2.pkl',
            'halfcheetah-medium-replay.pkl',
            'hopper-expert-v2.pkl',
            'hopper-medium-v2.pkl',
            'hopper-medium-replay.pkl',
            'walker2d-expert-v2.pkl',
            'walker2d-medium-v2.pkl',
            'walker2d-medium-replay-v2.pkl']

data_string = datasets[3]

with open('gym\\data\\' + data_string, 'rb') as f:
    trajectories = pickle.load(f)

dt = 1
for i, traj in enumerate(trajectories):
    rewards = []
    for ii in range(len(traj['rewards'])):
        forward = (traj['next_observations'][ii][0] - traj['observations'][ii][0])/dt
        alive = 0
        control = -0.1 * np.square(traj['actions'][ii]).sum()
        rewards.append([traj['rewards'][ii], forward + alive + control])
    
    trajectories[i]['rewards'] = rewards
    
print(trajectories[0]['rewards'])