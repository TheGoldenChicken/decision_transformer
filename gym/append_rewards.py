import pickle
import numpy as np
import os



ctrl_coef = [-0.1,-1e-3,-1e-3] # Halfcheetah, hopper, walker2d
alive_bonuses = [0, 1, 1] # Halfcheetah, hopper, walker2d
environments = [['halfcheetah-expert-v2.pkl','halfcheetah-medium-v2.pkl', 'halfcheetah-medium-replay-v2.pkl'],
                ['hopper-expert-v2.pkl', 'hopper-medium-v2.pkl', 'hopper-medium-replay-v2.pkl'],
                ['walker2d-expert-v2.pkl', 'walker2d-medium-v2.pkl', 'walker2d-medium-replay-v2.pkl']]

for i, env_type in enumerate(environments):
    for env in env_type:
        # Unpickle the dataset
<<<<<<< HEAD
        directory_path = os.path.dirname(os.path.abspath(__file__))
        with open(f'{directory_path}/data/{env}', 'rb') as f:
=======
        with open('data\\' + env, 'rb') as f:
>>>>>>> 5ed7c286ee6c14504779f5f6f4ac687df6cc0a3b
            dataset = pickle.load(f)
            
        # For all trajectories in the dataset
        for iii, traj in enumerate(dataset):
            reward = traj['rewards'] # Reward
            alive = alive_bonuses[i]*np.ones(len(reward)) # Alive bonus
            control = ctrl_coef[i]*np.square(traj['actions']).sum(axis = 1) # Control reward
            
            # Caulculate forward reward
            forward = reward - control - alive 
            
            # Redefine reward
            dataset[iii]['reward'] = np.array([reward,forward,control])
        
        filename = env[:-4] + '-split-reward.pkl'
        outfile = open(filename,'wb')
        pickle.dump(dataset,outfile)
        outfile.close()