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

from eval_act import evaluation_sequence
from utils import load_data  # data functions
from utils import (
    compute_dict_mean,
    set_seed,
    detach_dict,
    make_policy,
    forward_pass,
)  # helper functions


@hydra.main(config_path="config", config_name="train_default")
def train(conf: OmegaConf):
    dataset_path = conf.task_config.dataset_dir
    
    # get last folder name from dataset path
    dataset_category = dataset_path.split("/")[-1]
    set_seed(conf.seed)

    task_config = conf.task_config
    policy_config = conf.policy_config

    # load dataset
    train_dataloader, val_dataloader, stats, _ = load_data(task_config, conf.batch_size)

    # make policy and optimizer
    policy = make_policy(policy_config)
    if conf.ckpt_path:
        policy.load_state_dict(torch.load(conf.ckpt_path))
    policy.cuda()
    optimizer = policy.configure_optimizers()

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None

    start_date = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    current_file_path = os.path.dirname(
        os.path.realpath(__file__)
    )  # TODO Path 잘 만들기
    log_path = os.path.join(
        current_file_path, "logs/{}/{}".format(start_date, task_config.name, dataset_category)
    )
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    writer = SummaryWriter(log_path)  # TODO Logger를 따로 만들어 필요함수 생성

    # save dataset stats
    stats_path = os.path.join(log_path, f"dataset_stats.pkl")
    with open(stats_path, "wb") as f:
        pickle.dump(stats, f)

    env = PickCubeEnv()  # TODO 쉽게 변경 할 수 있게 수정

    for epoch in tqdm(range(conf.num_epochs)):
        print(f"\nEpoch {epoch}")
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary["loss"]
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        writer.add_scalar("Loss/Val", epoch_val_loss, epoch)
        print(f"Val loss:   {epoch_val_loss:.5f}")
        summary_string = ""
        for k, v in epoch_summary.items():
            summary_string += f"{k}: {v.item():.3f} "
        print(summary_string)

        # training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            # backward
            loss = forward_dict["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        epoch_summary = compute_dict_mean(
            train_history[(batch_idx + 1) * epoch : (batch_idx + 1) * (epoch + 1)]
        )
        epoch_train_loss = epoch_summary["loss"]
        writer.add_scalar("Loss/train", epoch_train_loss, epoch)
        print(f"Train loss: {epoch_train_loss:.5f}")
        summary_string = ""
        for k, v in epoch_summary.items():
            summary_string += f"{k}: {v.item():.3f} "
        print(summary_string)

        # evaluation
        # if epoch % conf.eval_interval == 0:
        #     success_num = 0
        #     for i in range(conf.rollout_num):
        #         rewards, episode_return, episode_highest_reward, is_success = (
        #             evaluation_sequence(env, policy, stats, False, False)
        #         )

        #         if(is_success):
        #             success_num += 1
        #     success_rate = success_num/conf.rollout_num
        #     writer.add_scalar("Eval/success_rate", success_rate, epoch)


        if epoch % conf.ckpt_save_interval == 0:
            ckpt_path = os.path.join(
                log_path, f"policy_epoch_{epoch}_seed_{conf.seed}.ckpt"
            )
            torch.save(policy.state_dict(), ckpt_path)

    ckpt_path = os.path.join(log_path, f"policy_last.ckpt")
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(
        log_path, f"policy_best_epoch_{best_epoch}_seed_{conf.seed}.ckpt"
    )
    torch.save(best_state_dict, ckpt_path)
    print(
        f"Training finished:\nSeed {conf.seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}"
    )


if __name__ == "__main__":
    train()
