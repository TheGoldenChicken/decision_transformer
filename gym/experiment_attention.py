import gym
import numpy as np
import torch
import wandb

import argparse
import pickle
import random
import sys
import os

from decision_transformer_multiple_rewards.evaluation.evaluate_episodes import evaluate_episode_rtg
from decision_transformer_multiple_rewards.models.decision_transformer_multiple_rewards import DecisionTransformer
from decision_transformer_multiple_rewards.models.mlp_bc import MLPBCModel
from decision_transformer_multiple_rewards.training.act_trainer import ActTrainer
from decision_transformer_multiple_rewards.training.seq_trainer import SequenceTrainer

from tqdm import tqdm

def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)): # Why do this reversed?? No idea
        # Because they need the discount cumsum of t+1, u dounce!
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum


def experiment(
        exp_prefix,
        variant,
):

    
    # seeding
    random.seed(variant['seed'])
    np.random.seed(variant['seed'])
    torch.manual_seed(variant['seed'])

    # Device and Wands and biases settings
    device = variant.get('device', 'cuda')
    log_to_wandb = variant.get('log_to_wandb', False)

    # Getting datasets and creating names
    env_name, dataset = variant['env'], variant['dataset']
    group_name = f'{exp_prefix}-{env_name}-{dataset}'
    exp_prefix = f"{group_name}-{variant['seed']}"

    # load dataset
    directory_path = os.path.dirname(os.path.abspath(__file__))
    dataset_path = f'{directory_path}/data_split_reward/{env_name}-{dataset}-v2.pkl'
    with open(dataset_path, 'rb') as f:
        trajectories = pickle.load(f)
        # trajectories is a list of dicts
        # observations, next_observations, actions, rewards, terminals (is_done)
        # next_observations is just observations at t+1 (why is this necessary?)
        # Ok, turns out not, there is some difference, don't know what though

    # save all path information into separate lists
    states, traj_lens, returns = [], [], []
    # States is basically a flattened list of observations from all trajectories
    print('Creating states list')
    for path in tqdm(trajectories):
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum()) # SKAL MÅSKE ÆNDRES
    traj_lens, returns = np.array(traj_lens), np.array(returns)
    # Traj lens are obviously just lens of single trajectory
    # Returns are end reward for all trajectories

    reward_dim = trajectories[0]['multi_rewards'].shape[0] # Calculates the dimension of the reward from the data

################## DEFINE ENV ##################

    if env_name == 'hopper':
        env = gym.make('Hopper-v3')
        max_ep_len = 1000
        step = 100
        # env_targets = np.swapaxes(np.array([np.arange(0, 1800+100, step) for _ in range(reward_dim)]), 0, 1)
        # env_targets = [3600, 1800]  # evaluation conditioning targets
        scale = 1000.  # normalization for rewards/returns
    elif env_name == 'halfcheetah':
        env = gym.make('HalfCheetah-v3')
        max_ep_len = 1000
        step = 400
        # env_targets = np.swapaxes(np.array([np.arange(0, 6000+400, step) for _ in range(reward_dim)]), 0, 1)
        # env_targets = [12000, 6000]
        scale = 1000.
    elif env_name == 'walker2d':
        env = gym.make('Walker2d-v3')
        max_ep_len = 1000
        step = 200
        # env_targets = np.swapaxes(np.array([np.arange(0, 2500+200, step) for _ in range(reward_dim)]), 0, 1)
        # env_targets = [5000, 2500]
        scale = 1000.
    else:
        raise NotImplementedError

    # vi laver grid 10x10 for target rewards
    max_forward_reward, max_ctrl_cost = np.array([traj['multi_rewards'].sum(axis=1) for traj in trajectories]).max(axis=0)

    forward_rewards = np.linspace(0, 1.5 * max_forward_reward, 10, endpoint=True)
    ctrl_costs = np.linspace(0, 1.5 * max_ctrl_cost, 10, endpoint=True)
    env_targets = np.array([[forward_reward, ctrl_cost] for forward_reward in forward_rewards for ctrl_cost in ctrl_costs])

    state_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    ################

    # used for input normalization
    states = np.concatenate(states, axis=0) # Flattens states along axis 0? # Check this code in debugger
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
    # state_dim number of means and stds

    num_timesteps = sum(traj_lens)

    print('=' * 50)
    print(f'Starting new experiment: {env_name} {dataset}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print('=' * 50)

    # used to reweight sampling so we sample according to timesteps instead of trajectories
    # makes longer trajectories more likely to be sampled
    p_sample = traj_lens / sum(traj_lens)

    def get_batch(batch_size=256, max_len=variant['K']):
        batch_inds = np.random.choice(
            np.arange(len(traj_lens)),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps - longer trajs more likely to sample
        )

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            # So this is a bit weird, because the way we create p_sample from traj_lens[sorted_inds], we are actually
            # sampling based on final return. That is trajectories[sorted_inds[900]] will have lower reward than if batch_inds = 1000...
            traj = trajectories[batch_inds[i]]
            si = random.randint(0, traj['multi_rewards'].shape[1] - 1) # Random number between 0 and len of trajectory - 1

            # get sequences from dataset
            # Based on context length
            s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
            a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim)) # Just reshapes to have 'extra dimension'
            r.append(np.expand_dims(traj['multi_rewards'][:,si:si + max_len], 0))
            r[-1] = np.swapaxes(r[-1], 1, 2) # Switches the last two axis

            # Not really sure about this? Probably some envs call them 'dones' rather than terminals... stupid, maybe gym vs mujoco?
            if 'terminals' in traj:
                d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            else:
                d.append(traj['dones'][si:si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len-1  # padding cutoff # remove stuff that is longer than max_ep_len allows

            rtg_i = []
            for j in range(r[0].shape[2]):
                rtg_i.append(discount_cumsum(traj['multi_rewards'][j,si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1)) # Only get reward to go in context length (+1 context length for some reason...)
            rtg.append(np.squeeze(np.stack(rtg_i, axis=2), axis=3))
            if rtg[-1].shape[1] <= s[-1].shape[1]: # Some shape correction here.. don't know when states would ever be longer than reward to go
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, reward_dim))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            # reward_dim = r[-1].shape[2]
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, reward_dim)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, reward_dim)), rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        # For at teste om flere rewards virker
        # rtg = torch.cat([rtg, rtg], 2)

        return s, a, r, d, rtg, timesteps, mask

    def eval_episodes(target_rew):
        def fn(model):
            returns, lengths, attentions, cross_attentions = [], [], [], []
            for _ in range(variant['num_eval_episodes']):
                with torch.no_grad():
                    ret, length, att, cross_att = evaluate_episode_rtg(
                        env,
                        state_dim,
                        act_dim,
                        reward_dim,
                        model,
                        max_ep_len=max_ep_len,
                        scale=scale,
                        target_return=target_rew/scale,
                        mode=variant.get('mode', 'normal'),
                        state_mean=state_mean,
                        state_std=state_std,
                        device=device,
                        return_atttenions_interval=0
                    )
                returns.append(ret)
                lengths.append(length)
                attentions.append(att)
                cross_attentions.append(cross_att)
            return {
                f'{target_rew}_returns': returns,
                f'{target_rew}_lenghts': lengths,
                f'{target_rew}_attentions': attentions,
                f'{target_rew}_cross_attentions': cross_attentions

                # f'target_{target_rew}_return_mean': np.mean(returns),
                # f'target_{target_rew}_return_std': np.std(returns),
                # f'target_{target_rew}_length_mean': np.mean(lengths),
                # f'target_{target_rew}_length_std': np.std(lengths),
            }
        return fn

    if variant['model_name'] != '':
        path = f"{variant['save_path']}{env_name}/{dataset}/{variant['model_name']}"

        file_to_read = open(path + '-kwargs', 'rb')
        model_kwargs = pickle.load(file_to_read)

        model = DecisionTransformer(**model_kwargs)
        model.load_state_dict(torch.load(path + '-model'))
    else:
        model_kwargs = {
            'state_dim'           : state_dim,
            'act_dim'             : act_dim,
            'reward_dim'          : reward_dim,
            'max_length'          : variant['K'],
            'max_ep_len'          : max_ep_len,
            'hidden_size'         : variant['embed_dim'],
            'n_layer'             : variant['n_layer'],
            'n_head'              : variant['n_head'],
            'n_inner'             : 4*variant['embed_dim'],
            'activation_function' : variant['activation_function'],
            'n_positions'         : 1024,
            'resid_pdrop'         : variant['dropout'],
            'attn_pdrop'          : variant['dropout']
        }

        model = DecisionTransformer(**model_kwargs)

    model = model.to(device=device)

    warmup_steps = variant['warmup_steps']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps+1)/warmup_steps, 1)
    )

    trainer = SequenceTrainer(
        model=model,
        optimizer=optimizer,
        batch_size= variant['batch_size'],
        get_batch=get_batch,
        scheduler=scheduler,
        loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a)**2),
        eval_fns=[eval_episodes(tar) for tar in env_targets],
    )

    if log_to_wandb:
        wandb.init(
            name=exp_prefix,
            group=group_name,
            project='decision-transformer',
            config=variant
        )
        # wandb.watch(model)  # wandb has some bug


    save_iters = set(map(int, variant['save_iters'].split(','))) if variant['save_iters'] != '' else []
    eval_iters = set(map(int, variant['eval_iters'].split(','))) if variant['eval_iters'] != '' else []

    for iter in range(1, variant['max_iters'] + 1):
        outputs = dict()
        outputs = trainer.train_iteration(num_steps=variant['num_steps_per_iter'], iter_num=iter, print_logs=True)

        if iter in save_iters:
            path = f"{directory_path}/{variant['save_path']}{env_name}/{dataset}/iter{iter}-{exp_prefix}"
            file = open(path + '-kwargs', 'wb')
            pickle.dump(model_kwargs, file)
            file.close()
            torch.save(trainer.model.state_dict(), path + '-model')


        if iter in eval_iters:
            eval_outputs = trainer.evaluate(num_steps=variant['num_steps_per_iter'], iter_num=iter, print_logs=True)
            file = open(f'{directory_path}/evaluation_data/{env_name}/{dataset}/iter{iter}-{exp_prefix}', 'wb')
            pickle.dump(eval_outputs, file)
            file.close()

        if log_to_wandb:
            if iter in eval_iters:
                for k, v in eval_outputs.items():
                    if k == 'time/evaluation':
                        outputs[k] = v
                    else:
                        target, statistic = k.split('_')
                        v = torch.stack(v, dim=1)
                        outputs[f'evaluation/target_{target}_{statistic[:-1]}_mean'] = torch.mean(v, dim=1)
                        outputs[f'evaluation/target_{target}_{statistic[:-1]}_std'] = torch.std(v, dim=1)

            wandb.log(outputs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, default='')

    parser.add_argument('--save_iters', type=str, default='') # string like '5,10,15'
    parser.add_argument('--save_path', type=str, default='./saved_models/')
    
    parser.add_argument('--eval_iters', type=str, default='1') # string like '5,10,15'

    parser.add_argument('--env', type=str, default='hopper')
    parser.add_argument('--dataset', type=str, default='medium-replay')  # medium, medium-replay, medium-expert, expert
    parser.add_argument('--num_eval_episodes', type=int, default=100) # 100
    parser.add_argument('--max_iters', type=int, default=10)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=False)


    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--embed_dim', type=int, default=128)

    parser.add_argument('--num_steps_per_iter', type=int, default=10_000) #10000
    
    parser.add_argument('--K', type=int, default=20) # contect window
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10_000)
    parser.add_argument('--mode', type=str, default='normal')
    parser.add_argument('--seed', type=int, default=random.randint(int(1e5), int(1e6) - 1))

    parser.add_argument('--split_reward', type=bool, default=False)

    args = parser.parse_args()

    experiment('gym-experiment_multi', variant=vars(args))
