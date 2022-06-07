import pickle
import numpy as np
import os


def get_forward_reward(positions):
    return positions[1:] - positions[:-1]


def get_control_cost(actions):
    return np.sum(np.square(actions), axis=1)

data_path = 'data_plus_plus/'
forward_coefs = [.05, 0.008, 0.008]
ctrl_coefs = [0.1, 1e-3, 1e-3] # Halfcheetah, hopper, walker2d
alive_coefs = [0, 1, 1]       # Halfcheetah, hopper, walker2d
environments = [['halfcheetah-expert-v2.pkl','halfcheetah-medium-v2.pkl', 'halfcheetah-medium_replay-v2.pkl'],
                ['hopper-expert-v2.pkl', 'hopper-medium-v2.pkl', 'hopper-medium_replay-v2.pkl'],
                ['walker2d-expert-v2.pkl', 'walker2d-medium-v2.pkl', 'walker2d-medium_replay-v2.pkl']]

for env_type, forward_coef, ctrl_coef, alive_coef in zip(environments, forward_coefs, ctrl_coefs, alive_coefs):
    for env in env_type:
        # Unpickle the dataset
        with open(data_path + env, 'rb') as f:
            dataset = pickle.load(f)
            
        # For all trajectories in the dataset
        for i, traj in enumerate(dataset):

            # calculate forward reward
            forward_reward = forward_coef * get_forward_reward(traj['infos/qpos'][:,0])

            # remove last datapoint for each trajectory (forward_reward has dim(traj) - 1)
            for k, v in traj.items():
                traj[k] = v[:-1]

            # save reward splits
            ctrl_cost = ctrl_coef * get_control_cost(traj['actions'])
            alive_bonus = alive_coef * np.array(~traj['terminals'], dtype=int)
            
            # Redefine reward
            if env.split('-')[0] == 'halfcheetah': # halfcheetah does not use alive bonus
                dataset[i]['multi_rewards'] = np.array([forward_reward, ctrl_cost])
            else:
                dataset[i]['multi_rewards'] = np.array([forward_reward, ctrl_cost, alive_bonus])
        
        filename = env.split('.')[0] + '-split_reward.pkl'
        outfile = open(data_path + filename,'wb')
        pickle.dump(dataset, outfile)
        outfile.close()