import pickle
import numpy as np



ctrl_coef = [-0.1,-1e-3,-1e-3] # Halfcheetah, hopper, walker2d
alive_bonuses = [0, 1, 1] # Halfcheetah, hopper, walker2d
environments = [['halfcheetah-expert-v2.pkl','halfcheetah-medium-v2.pkl', 'halfcheetah-medium_replay-v2.pkl'],
                ['hopper-expert-v2.pkl', 'hopper-medium-v2.pkl', 'hopper-medium_replay-v2.pkl'],
                ['walker2d-expert-v2.pkl', 'walker2d-medium-v2.pkl', 'walker2d-medium_replay-v2.pkl']]

for i, env_type in enumerate(environments):
    for env in env_type:
        # Unpickle the dataset
        with open('gym\\data\\' + env, 'rb') as f:
            dataset = pickle.load(f)
            
        # For all trajectories in the dataset
        for iii, traj in enumerate(dataset):
            reward = traj['rewards'] # Reward
            alive = alive_bonuses[i]*np.ones(len(reward)) # Alive bonus
            control = ctrl_coef[i]*np.square(traj['actions']).sum(axis = 1) # Control reward
            
            # Caulculate forward reward
            forward = reward - control - alive 
            
            # Redefine reward
            dataset[iii]['reward'] = [reward,forward,control]
        
        filename = env[:-4] + '-split-reward'
        outfile = open(filename,'wb')
        pickle.dump(dataset,outfile)
        outfile.close()
