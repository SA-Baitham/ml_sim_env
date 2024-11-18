import torch
import numpy as np
import os
import pickle
from copy import deepcopy
from tqdm import tqdm
from omegaconf import OmegaConf

import hydra
from torch.utils.tensorboard import SummaryWriter
from environments.mujoco_envs.pick_cube_env import PickCubeEnv
import datetime

from utils import load_data  # data functions
from utils import (
    set_seed,
    make_policy,
    get_image,
    pre_process,
    post_process,
)  # helper functions


# Set print options to avoid scientific notation
np.set_printoptions(suppress=True, precision=4)
torch.set_printoptions(sci_mode=False, precision=4)

def evaluation_sequence(env, policy, stats, temporal_agg, is_render):
    obs, _ = env.reset()  # TODO: observation 형식 통일?

    max_timesteps = env.step_limit
    chunk_size = policy.chunk_size
    state_dim = policy.state_dim
    camera_names = policy.camera_names

    if temporal_agg:
        query_frequency = 1
    else:
        query_frequency = chunk_size

    ### evaluation loop
    if temporal_agg:
        all_time_actions = torch.zeros(
            [max_timesteps, max_timesteps + chunk_size, state_dim]
        ).cuda()

    qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
    image_list = []  # for visualization
    qpos_list = []
    action_list = []
    rewards = []
    with torch.inference_mode():
        for t in range(max_timesteps):
            ### process previous timestep to get qpos and image_list
            if "images" in obs:
                image_list.append(obs["images"])
            else:
                image_list.append({"main": obs["image"]})
            qpos_numpy = np.array(obs["qpos"])
            qpos = pre_process(qpos_numpy, stats)
            qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
            qpos_history[:, t] = qpos
            curr_image = get_image(obs, policy.camera_names)

            ### query policy
            if t % query_frequency == 0:
                all_actions = policy(qpos, curr_image)
            if temporal_agg:
                all_time_actions[[t], t : t + chunk_size] = all_actions
                actions_for_curr_step = all_time_actions[:, t]
                actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                actions_for_curr_step = actions_for_curr_step[actions_populated]
                k = 0.01
                exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                exp_weights = exp_weights / exp_weights.sum()
                exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                raw_action = (actions_for_curr_step * exp_weights).sum(
                    dim=0, keepdim=True
                )
            else:
                raw_action = all_actions[:, t % query_frequency]

            ### post-process actions
            raw_action = raw_action.squeeze(0).cpu().numpy()
            action = post_process(raw_action, stats)

            ### step the environment
            print(f"Step {t}: {action}")
            obs, reward, terminated, truncated, info = env.step(action)

            if is_render:
                env.render()

            ### for visualization
            qpos_list.append(qpos_numpy)
            action_list.append(action)
            rewards.append(reward)

    rewards = np.array(rewards)
    episode_return = np.sum(rewards[rewards != None])
    episode_highest_reward = np.max(rewards)

    is_success = episode_highest_reward==env.env_max_reward

    return rewards, episode_return, episode_highest_reward, is_success


@hydra.main(config_path="config", config_name="eval_default")
def eval(conf: OmegaConf):
    set_seed(conf.seed)

    task_config = conf.task_config
    policy_config = conf.policy_config

    # load policy and stats
    policy = make_policy(policy_config)
    try:
        loading_status = policy.load_state_dict(torch.load(conf.ckpt_path))
    except:
        raise FileNotFoundError("ckpt file not found")

    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f"Loaded: {conf.ckpt_path}")
    ckpt_folder_path = os.path.dirname(conf.ckpt_path)
    ckpt_file_name = os.path.basename(conf.ckpt_path)
    stats_path = os.path.join(ckpt_folder_path, f"dataset_stats.pkl")
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)

    env = PickCubeEnv()  # TODO: Make gym style(ex: gym.make)

    num_rollouts = conf.rollout_num
    episode_returns = []
    highest_rewards = []
    for rollout_id in range(num_rollouts):
        rewards, episode_return, episode_highest_reward, is_success = (
            evaluation_sequence(env, policy, stats, conf.temporal_agg, conf.is_render)
        )

        episode_returns.append(episode_return)
        highest_rewards.append(episode_highest_reward)
        print(
            f"Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env.env_max_reward=}, Success: {is_success}"
        )

        # if save_episode:
        #     save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video{rollout_id}.mp4'))

    success_rate = np.mean(np.array(highest_rewards) == env.env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f"\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n"
    for r in range(env.env_max_reward + 1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f"Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n"

    print(summary_str)

    # save success rate to txt
    result_file_name = "result_" + ckpt_file_name.split(".")[0] + ".txt"
    with open(os.path.join(ckpt_folder_path, result_file_name), "w") as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write("\n\n")
        f.write(repr(highest_rewards))


if __name__ == "__main__":
    eval()
