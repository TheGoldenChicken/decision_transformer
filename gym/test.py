from decision_transformer.evaluation.evaluate_episodes import evaluate_episode_rtg

ret, length = evaluate_episode_rtg(
    env,
    state_dim,
    act_dim,
    model,
    max_ep_len=max_ep_len,
    scale=scale,
    target_return=target_rew/scale,
    mode=variant.get('mode', 'normal'),
    state_mean=state_mean,
    state_std=state_std,
    device=device,
)