import pickle
import numpy as np

envs = ['halfcheetah', 'hopper', 'walker2d']
levels = ['expert', 'medium-replay', 'random', 'medium']



for env in envs:
    for level in levels:
        dataset = f'data/{env}-{level}-v2.pkl'
        with (open(dataset, 'rb')) as data:
            curr_data = pickle.load(data)
        print(f'{env}-{level} num trajectories is: {len(curr_data)}')
        print(f'{env}-{level} avg. len of trajectories is:', np.mean([len(curr_data[i]['observations']) for i in range(len(curr_data))]))


