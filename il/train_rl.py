"""Fine-tune an obs2 BC checkpoint with PPO in the rope simulator."""

import argparse
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.distributions import Independent, Normal

from policy import load_agent, save_checkpoint
from scene import EVAL_SEEDS, make_env


HORIZON = 600
# Scene seed blocks reserved for PPO sampling and checkpoint selection.
TRAIN_SEEDS = (910_000, 930_000)
DEV_SEEDS = range(940_000, 940_010)


@torch.no_grad()
def collect_episode(agent, critic, seed, noise_std, final_bonus):
    env, obs = make_env(seed, horizon=HORIZON)
    agent.reset()
    states, actions, log_probs, values, rewards = [], [], [], [], []

    for step in range(HORIZON):
        agent.observe(obs)
        state = agent.policy_input()
        distribution = Independent(Normal(agent.model(state[None])[0], noise_std), 1)
        action = distribution.sample()
        # PPO tracks the sampled normalized action before the actuator clamp.
        command = np.clip(agent.y_norm.inverse(action.numpy()), -1, 1)
        obs, reward, done, info = env.step(command)
        states.append(state)
        actions.append(action)
        log_probs.append(distribution.log_prob(action))
        values.append(critic(state).squeeze())
        # Reward success at the official time limit as well as progress during the episode.
        if step + 1 == HORIZON and info['success'] and not info.get('bad', False):
            reward += final_bonus
        rewards.append(reward)
        if done:
            break

    advantages = torch.zeros(len(rewards))
    next_value = 0.0
    gae = 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + 0.995 * next_value - values[t]
        gae = delta + 0.995 * 0.95 * gae
        advantages[t] = gae
        next_value = values[t]
    returns = advantages + torch.stack(values)
    return (
        torch.stack(states), torch.stack(actions), torch.stack(log_probs),
        advantages, returns,
    ), sum(rewards)


def update_policy(actor, reference, critic, optimizer, episodes, noise_std, bc_weight):
    states, actions, old_log_probs, advantages, returns = (
        torch.cat(column) for column in zip(*episodes)
    )
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    with torch.no_grad():
        reference_actions = reference(states)
        old_means = actor(states)

    for _ in range(4):
        for batch in torch.randperm(len(states)).split(256):
            predicted = actor(states[batch])
            distribution = Independent(Normal(predicted, noise_std), 1)
            ratio = (distribution.log_prob(actions[batch]) - old_log_probs[batch]).exp()
            advantage = advantages[batch]
            policy_loss = -torch.minimum(ratio * advantage, ratio.clamp(0.8, 1.2) * advantage).mean()
            value_loss = (critic(states[batch]).squeeze(-1) - returns[batch]).square().mean()
            # Keep the fine-tuned policy close to BC on the states it visits.
            bc_loss = (predicted - reference_actions[batch]).square().mean()
            loss = policy_loss + 0.5 * value_loss + bc_weight * bc_loss
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite PPO loss')
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
            nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
            optimizer.step()
        with torch.no_grad():
            kl = ((actor(states) - old_means) / noise_std).square().sum(dim=1).mean() / 2
        if kl.item() > 0.02:
            break
    return kl.item()


def evaluate(agent, seeds=DEV_SEEDS):
    outcomes = {}
    for seed in seeds:
        env, obs = make_env(seed, horizon=HORIZON)
        agent.reset()
        ever = False
        for _ in range(HORIZON):
            obs, _, done, info = env.step(agent.act(obs))
            ever |= bool(info['success'])
            if done:
                break
        outcomes[seed] = {'success': bool(info['success']), 'ever': ever}
    return outcomes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bc-ckpt', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=20)
    parser.add_argument('--episodes-per-update', type=int, default=4)
    parser.add_argument('--eval-every', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--actor-lr', type=float, default=1e-5)
    parser.add_argument('--noise-std', type=float, default=0.10)
    parser.add_argument('--bc-weight', type=float, default=5.0)
    parser.add_argument('--final-bonus', type=float, default=5.0)
    args = parser.parse_args()
    if min(args.updates, args.episodes_per_update, args.eval_every) < 1 or args.seed < 0:
        parser.error('update counts must be positive and seed must be nonnegative')
    if args.out.exists():
        parser.error('--out must be a new directory')
    settings = (args.actor_lr, args.noise_std, args.bc_weight, args.final_bonus)
    if not all(np.isfinite(settings)) or min(settings[:2]) <= 0 or min(settings[2:]) < 0:
        parser.error('learning rate and noise must be positive, weights must be nonnegative')
    if set(EVAL_SEEDS) & (set(range(*TRAIN_SEEDS)) | set(DEV_SEEDS)):
        parser.error('PPO seeds overlap with the official evaluation seeds')

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    reference_agent = load_agent(args.bc_ckpt)
    if (reference_agent.To, reference_agent.Ta) != (2, 1):
        parser.error('--bc-ckpt must use obs-horizon 2 and act-horizon 1')
    agent = deepcopy(reference_agent)
    actor = agent.model
    critic = nn.Sequential(nn.Linear(agent.cfg['model']['in_dim'], 128), nn.Tanh(), nn.Linear(128, 1))
    optimizer = torch.optim.Adam([
        {'params': actor.parameters(), 'lr': args.actor_lr},
        {'params': critic.parameters(), 'lr': 3e-4},
    ])
    args.out.mkdir(parents=True, exist_ok=False)

    metadata = dict(vars(args), bc_ckpt=str(args.bc_ckpt.resolve()), out=str(args.out),
                    horizon=HORIZON, train_seed_range=list(TRAIN_SEEDS), dev_seeds=list(DEV_SEEDS))
    (args.out / 'rl_config.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    log_path = args.out / 'rl_log.jsonl'

    def log(entry):
        with log_path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(entry) + '\n')

    print('Evaluating the original BC policy on development seeds...', flush=True)
    baseline = evaluate(reference_agent)
    best = sum(item['success'] for item in baseline.values())
    log(dict(update=0, dev_successes=best, dev_outcomes=baseline))
    print(f'BC baseline on development seeds: {best}/{len(DEV_SEEDS)}', flush=True)
    save_checkpoint(
        args.out / 'ckpt.pt', actor, agent.cfg, agent.x_norm, agent.y_norm,
        rl_update=0, dev_successes=best, training_method='bc',
    )
    for update in range(1, args.updates + 1):
        seeds = rng.integers(*TRAIN_SEEDS, size=args.episodes_per_update)
        print(f'Collecting episodes for PPO update {update}/{args.updates}...', flush=True)
        episodes, rewards = zip(*(collect_episode(
            agent, critic, int(seed), args.noise_std, args.final_bonus
        ) for seed in seeds))
        kl = update_policy(actor, reference_agent.model, critic, optimizer, episodes,
                           args.noise_std, args.bc_weight)
        entry = dict(update=update, train_seeds=seeds.tolist(),
                     mean_reward=float(np.mean(rewards)), update_kl=kl)
        if update % args.eval_every == 0 or update == args.updates:
            outcomes = evaluate(agent)
            score = sum(item['success'] for item in outcomes.values())
            gained = sum(item['success'] and not baseline[seed]['success'] for seed, item in outcomes.items())
            lost = sum(baseline[seed]['success'] and not item['success'] for seed, item in outcomes.items())
            entry.update(dev_successes=score, gained=gained, lost=lost, dev_outcomes=outcomes)
            print(f'Development success: {score}/{len(DEV_SEEDS)}, gained {gained}, lost {lost}', flush=True)
            for name in ('latest.pt', 'ckpt.pt') if score > best else ('latest.pt',):
                save_checkpoint(
                    args.out / name, actor, agent.cfg, agent.x_norm, agent.y_norm,
                    rl_update=update, dev_successes=score, training_method='bc_ppo',
                )
            best = max(best, score)
        log(entry)


if __name__ == '__main__':
    main()
