#!/usr/bin/env python3
"""
Shows how to toss a capsule to a container.
"""
from mujoco_py import load_model_from_path, MjSim, MjViewer
import gym
import os
import random
import pickle
import numpy as np
import pandas
import time

ass_dir = '/home/dayman/miniconda3/envs/decision-transformer-gym/lib/python3.8/site-packages/gym/envs/mujoco/assets/'

dataset_path = f'data/halfcheetah-expert-v2.pkl'
with open(dataset_path, 'rb') as f:
    trajectories = pickle.load(f)

#Load the model and environment from its xml file
#model = load_model_from_path("../xmls/tosser.xml")
#model = load_model_from_path('/home/dayman/miniconda3/envs/decision-transformer-gym/lib/python3.8/site-packages/gym/envs/mujoco/assets/half_cheetah.xml')
#sim = MjSim(model)
env = gym.make('HalfCheetah-v2')
# The following doesn't work, gym.make returns some kind of weird timelimit object, a wrapper for gym time_limits
# model2 = gym.make('Ant') # Remember capitalization
# sim2 = MjSim(model2)

#the time for each episode of the simulation
sim_horizon = 1000

#initialize the simulation visualization
#viewer = MjViewer(sim)

#get initial state of simulation
#sim_state = sim.get_state()

step = 0
#ob = env.reset()

for traj in trajectories:
    step = 0
    #ob = env.manual_reset(traj['infos/qpos'][0], traj['infos/qvel'][0])
    ob = env.reset()
    print('resetting')
    env.unwrapped.set_state(traj['infos/qpos'][0], traj['infos/qvel'][0])
    print(len(traj['infos/qpos']))
    for r, actionss in enumerate(traj['actions']):
        action = actionss
        ob, reward, done, info = env.step(action)
        env.render()
        time.sleep(0.01)
        step += 1

        if step % 100 == 0:
            print('observation ', ob)
            print('should be ', traj['observations'][r])
        #if step % 1000 == 0:
        #    env.reset()

#repeat indefinitely
# while True:
#     #set simulation to initial state
#     sim.set_state(sim_state)
#
#     #sim.data.qpos[:] = trajectories[0]['infos/qpos'][0]
#
#
#     #for the entire simulation horizon
#     for i in range(sim_horizon):
#
#         sim.data.ctrl[:] = trajectories[0]['actions'][i] * 1000
#
#         # #trigger the lever within the 0 to 150 time period
#         # if i < 150:
#         #     #sim.data.ctrl[:] = 0.0
#         #     for i, ct in enumerate(sim.data.ctrl):
#         #         sim.data.ctrl[:] = random.uniform(-1000,1000)
#         #
#         # else:
#         #     for i, ct in enumerate(sim.data.ctrl):
#         #         sim.data.ctrl[:] = random.uniform(-1000,1000)
#
#             #sim.data.ctrl[:] = -1.0
#         #move one time step forward in simulation
#         sim.step()
#         viewer.render()
#
#     if os.getenv('TESTING') is not None:
#         break
