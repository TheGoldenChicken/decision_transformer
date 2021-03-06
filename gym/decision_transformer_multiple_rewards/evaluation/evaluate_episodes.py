import numpy as np
import torch


def evaluate_episode_rtg(
        env,
        state_dim,
        act_dim,
        reward_dim,
        model,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
        return_atttenions_interval = 0
    ):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()
    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros((0, reward_dim), device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, reward_dim)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    sim_states = []

    episode_attentions = []
    episode_cross_attentions = []
    episode_return, episode_length = [], 0
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros((1, reward_dim), device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
            return_attentions = False, # Change this if no attention return is needed
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        x_position_before = env.env.data.qpos[0]
        state, reward, done, _ = env.step(action)
        x_position_after = env.env.data.qpos[0]
        forward_reward = env.env._forward_reward_weight / env.env.dt * (x_position_after - x_position_before)
        ctrl_cost = env.env._ctrl_cost_weight * np.sum(np.square(action))

        reward = torch.tensor([forward_reward, ctrl_cost]).to(device=device)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        if mode != 'delayed':
            pred_return = target_return[0,-1] - (reward/scale) # M??SKE FEJL HER
        else:
            pred_return = target_return[0,-1]
        target_return = torch.cat(
            [target_return, pred_return.reshape(1, reward_dim)], dim=1)
        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        episode_return += [[forward_reward, ctrl_cost]]
        episode_length += 1

        # if return_atttenions_interval % t == 0:
        #     episode_attentions.append(attentions)
        #     episode_cross_attentions.append(cross_attentions)

        if done:
            break

    return episode_return, episode_length