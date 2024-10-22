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


def evaluation_sequence(env, policy, stats, temporal_agg, is_render, option):
    obs, _ = env.reset(option)  # TODO: observation 형식 통일?

    max_timesteps = env.step_limit
    chunk_size = policy.chunk_size
    state_dim = policy.state_dim
    camera_names = policy.camera_names
    # print("state_dim", state_dim)

    if temporal_agg:
        query_frequency = 1
    else:
        query_frequency = chunk_size

    ### evaluation loop
    if temporal_agg:
        all_time_actions = torch.zeros(
            [max_timesteps, max_timesteps + chunk_size, 8]
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

    is_success = episode_highest_reward == env.env_max_reward

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

    option_list = [
        {
            "generated_cube_pose": [
                0.08195033797814436,
                0.649710261653528,
                0.16393482690393846,
                0.17599400266154233,
                0.0,
                0.0,
                -0.9843912388004928,
            ]
        },
        {
            "generated_cube_pose": [
                -0.07853849890252396,
                0.6365104979102101,
                0.14941354251523398,
                0.713968271059121,
                0.0,
                0.0,
                -0.7001780544410469,
            ]
        },
        {
            "generated_cube_pose": [
                -0.03505245719860538,
                0.6261015104773942,
                0.1345764977219966,
                0.994176760191427,
                0.0,
                0.0,
                -0.10776163276082068,
            ]
        },
        {
            "generated_cube_pose": [
                0.02183554870148921,
                0.48749156625300716,
                0.10356992646399223,
                0.017334495853050715,
                0.0,
                0.0,
                -0.999849746338679,
            ]
        },
        {
            "generated_cube_pose": [
                0.018059701830922675,
                0.4581745615750393,
                0.19467086316049487,
                0.12075275740665972,
                0.0,
                0.0,
                0.9926826137183468,
            ]
        },
        {
            "generated_cube_pose": [
                0.09780622359908822,
                0.6166732625955806,
                0.1953778466469305,
                0.999456131842487,
                0.0,
                0.0,
                0.032976363087113554,
            ]
        },
        {
            "generated_cube_pose": [
                -0.01092387892112294,
                0.5355359712963073,
                0.16871939872188652,
                0.8713227571093622,
                0.0,
                0.0,
                -0.4907103554474262,
            ]
        },
        {
            "generated_cube_pose": [
                -0.036540901235268916,
                0.6178259482353987,
                0.11540269683066838,
                0.999092765009305,
                0.0,
                0.0,
                0.04258693351324703,
            ]
        },
        {
            "generated_cube_pose": [
                -0.03302006154067316,
                0.6109059153268466,
                0.1337183420279534,
                0.950734324316642,
                0.0,
                0.0,
                -0.3100068459988881,
            ]
        },
        {
            "generated_cube_pose": [
                -0.044864817299097616,
                0.5622519560767583,
                0.18126903060127636,
                0.5180405101092281,
                0.0,
                0.0,
                0.8553560836784705,
            ]
        },
        {
            "generated_cube_pose": [
                0.0031374236762943264,
                0.4733525356601857,
                0.17100873566527747,
                0.8131954163950792,
                0.0,
                0.0,
                -0.5819907342510136,
            ]
        },
        {
            "generated_cube_pose": [
                0.08171012744791609,
                0.5616523225075731,
                0.10469795274671828,
                0.2447992238498158,
                0.0,
                0.0,
                0.9695737929639641,
            ]
        },
        {
            "generated_cube_pose": [
                -0.03373576480158012,
                0.5844747911775428,
                0.11059911024721175,
                0.02227067536263199,
                0.0,
                0.0,
                -0.9997519777519284,
            ]
        },
        {
            "generated_cube_pose": [
                0.02703848224435007,
                0.5733483407374296,
                0.10475223580863145,
                0.998736437782748,
                0.0,
                0.0,
                0.0502546300854667,
            ]
        },
        {
            "generated_cube_pose": [
                0.006467805959647466,
                0.5550325719492091,
                0.13775535675077677,
                0.9649682759169335,
                0.0,
                0.0,
                -0.26236658795262197,
            ]
        },
        {
            "generated_cube_pose": [
                -0.03347744241634419,
                0.4903082403147443,
                0.10411598720708816,
                0.17824977415685855,
                0.0,
                0.0,
                -0.9839852732704026,
            ]
        },
        {
            "generated_cube_pose": [
                -0.0782545596064716,
                0.4686139611803118,
                0.10981957838170958,
                0.5466107033043258,
                0.0,
                0.0,
                -0.8373868514809332,
            ]
        },
        {
            "generated_cube_pose": [
                0.054159099763507884,
                0.5081344998912407,
                0.16409726192159857,
                0.9860544548681068,
                0.0,
                0.0,
                0.16642299130456872,
            ]
        },
        {
            "generated_cube_pose": [
                0.08568705084361364,
                0.5404232422945867,
                0.15649044290436315,
                0.7018071576487116,
                0.0,
                0.0,
                -0.7123669794937414,
            ]
        },
        {
            "generated_cube_pose": [
                0.07606883636514136,
                0.6217889839347049,
                0.19328649756244615,
                0.7110595540629139,
                0.0,
                0.0,
                -0.7031317874878436,
            ]
        },
        {
            "generated_cube_pose": [
                -0.025330000488050994,
                0.6467413481402475,
                0.1161624528267785,
                0.8362835397894038,
                0.0,
                0.0,
                0.5482972196512623,
            ]
        },
        {
            "generated_cube_pose": [
                0.03406424836361549,
                0.6183925432692076,
                0.1774783748690097,
                0.07858686047780293,
                0.0,
                0.0,
                -0.9969072701912863,
            ]
        },
        {
            "generated_cube_pose": [
                -0.07489745699807025,
                0.4887717390024993,
                0.10854277000938539,
                0.7949439036085699,
                0.0,
                0.0,
                0.6066829403531705,
            ]
        },
        {
            "generated_cube_pose": [
                0.08863482748195192,
                0.46091478364435035,
                0.13576259719062356,
                0.42734261077130203,
                0.0,
                0.0,
                0.9040897593818699,
            ]
        },
        {
            "generated_cube_pose": [
                -0.011717103559862219,
                0.5024267805671534,
                0.18601463217217756,
                0.9090058362301285,
                0.0,
                0.0,
                -0.4167833846251127,
            ]
        },
        {
            "generated_cube_pose": [
                0.006018533993329453,
                0.4866833353921736,
                0.10121501970950221,
                0.11571719625698329,
                0.0,
                0.0,
                -0.9932822008323832,
            ]
        },
        {
            "generated_cube_pose": [
                -0.014973838592936883,
                0.4941960700951688,
                0.11651400846919577,
                0.9828137155864806,
                0.0,
                0.0,
                -0.18460010957498493,
            ]
        },
        {
            "generated_cube_pose": [
                0.0036798391408977144,
                0.48705297786530083,
                0.19255715759471806,
                0.22494385593962687,
                0.0,
                0.0,
                0.9743717266397934,
            ]
        },
        {
            "generated_cube_pose": [
                -0.07992881028365567,
                0.567364000865142,
                0.10957831282641611,
                0.29151466462866127,
                0.0,
                0.0,
                0.9565663595937498,
            ]
        },
        {
            "generated_cube_pose": [
                0.059673045316092854,
                0.6275818288929993,
                0.199392029177131,
                0.36504357167622314,
                0.0,
                0.0,
                -0.9309904353847391,
            ]
        },
    ]

    num_rollouts = conf.rollout_num
    episode_returns = []
    highest_rewards = []
    for rollout_id in range(num_rollouts):
        rewards, episode_return, episode_highest_reward, is_success = (
            evaluation_sequence(
                env,
                policy,
                stats,
                conf.temporal_agg,
                conf.is_render,
                option_list[rollout_id],
            )
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
