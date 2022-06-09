import numpy as np
import pickle
def get_cumsum(dataset):
    cum_sum = 0
    total_len = 0
    for traj in dataset:
        cum_summ = np.sum(traj['rewards'])
        cum_sum += cum_summ
        total_len += 1

    cum_sum = cum_sum / total_len
    return cum_sum

with open('halfcheetah-random-v2.pkl', 'rb') as f:
    trajectories1 = pickle.load(f)

with open('hopper-random-v2.pkl', 'rb') as f:
    trajectories2 = pickle.load(f)
with open('walker2d-random-v2.pkl', 'rb') as f:
    trajectories3 = pickle.load(f)


print(get_cumsum(trajectories1))
print(get_cumsum(trajectories2))
print(get_cumsum(trajectories3))

