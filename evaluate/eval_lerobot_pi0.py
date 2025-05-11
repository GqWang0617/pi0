import os
from aloha_scripts.utils import *
import torch
from torchvision import transforms
import cv2
import numpy as np
import time
from transformers import AutoTokenizer
from data_utils.dataset import set_seed
import sys

from policy_heads import *
from qwen2_vla.utils.image_processing_qwen2_vla import *
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy


def get_obs(deplot_env_obs, camera_views=4):
    cur_traj_data = dict()
    # (480, 270, 4)

    cur_bottom_rgb = deplot_env_obs['images']['cam_bottom']  # camera_extrinsics image
    cur_top_rgb = deplot_env_obs['images']['cam_top']  # camera_extrinsics image
    cur_left_rgb = deplot_env_obs['images']['cam_left_wrist']  # camera_extrinsics image
    cur_right_rgb = deplot_env_obs['images']['cam_right_wrist']  # camera_extrinsics image

    cur_bottom_rgb = cv2.resize(cv2.cvtColor(cur_bottom_rgb, cv2.COLOR_BGRA2BGR), (320, 240))[:, :, ::-1]
    cur_top_rgb = cv2.resize(cv2.cvtColor(cur_top_rgb, cv2.COLOR_BGRA2BGR), (320, 240))[:, :, ::-1]
    cur_left_rgb = cv2.resize(cv2.cvtColor(cur_left_rgb, cv2.COLOR_BGRA2BGR), (320, 240))[:, :, ::-1]
    cur_right_rgb = cv2.resize(cv2.cvtColor(cur_right_rgb, cv2.COLOR_BGRA2BGR), (320, 240))[:, :, ::-1]

    # cv2.imshow('cur_rgb', cv2.hconcat([cur_left_rgb, cur_right_rgb, cur_bottom_rgb, cur_top_rgb]))
    # cv2.waitKey(1)

    cur_right_depth = np.zeros_like(cur_right_rgb) - 1.0
    cur_right_depth = cur_right_depth[..., :1]
    cur_left_depth = np.zeros_like(cur_left_rgb) - 1.0
    cur_left_depth = cur_left_depth[..., :1]

    cur_joint_positions = deplot_env_obs['qpos']

    cur_state_np = cur_joint_positions

    # [128, 128, 3] np array
    right_rgb_img = cur_right_rgb  # deplot_env_obs['front']
    right_depth_img = cur_right_depth
    left_rgb_img = cur_left_rgb  # deplot_env_obs['wrist_1']
    left_depth_img = cur_left_depth
    # cur_high_rgb = cur_top_rgb

    cur_state = cur_state_np  # deplot_env_obs['state']
    cur_state = np.expand_dims(cur_state, axis=0)

    # [2, 1, 128, 128, 3]
    # [2, 480, 480, 3]
    if camera_views == 4:
        traj_rgb_np = np.array([cur_bottom_rgb, cur_top_rgb, left_rgb_img, right_rgb_img])
    else:
        traj_rgb_np = np.array([cur_top_rgb, left_rgb_img, right_rgb_img])


    traj_rgb_np = np.expand_dims(traj_rgb_np, axis=1)
    traj_rgb_np = np.transpose(traj_rgb_np, (1, 0, 4, 2, 3))

    traj_depth_np = np.array([right_depth_img, left_depth_img])
    traj_depth_np = np.expand_dims(traj_depth_np, axis=1)
    traj_depth_np = np.transpose(traj_depth_np, (1, 0, 4, 2, 3))

    print("#" * 50)
    print(traj_rgb_np.shape)

    return cur_joint_positions, cur_state, traj_rgb_np, traj_depth_np


def prepare_inputs(observations, state, task):
    observations = observations / 255.0
    input_item = {}
    img_idx = 0
    for k in ['observation.state', 'observation.images.cam_high', 'observation.images.cam_left_wrist', 'observation.images.cam_right_wrist', 'task']:
        if 'images' in k:
            input_item[k] = torch.from_numpy(observations[0][img_idx]).unsqueeze(0).to(dtype=torch.bfloat16, device='cuda')
            img_idx += 1
        elif k == 'task':
            input_item[k] = [task]
        else:
            input_item[k] = torch.from_numpy(state).unsqueeze(0).to(dtype=torch.float32, device='cuda')

    return input_item

def eval_bc(policy, deploy_env, policy_config, raw_lang=None):
    assert raw_lang is not None, "raw lang is None!!!!!!"
    set_seed(0)

    query_frequency = 15
    rand_crop_resize = False
    state_dim = 14

    policy.eval()
    policy = policy.to("cuda")

    max_timesteps = int(1000 * 10)  # may increase for real-world tasks
    for rollout_id in range(1000):
        rollout_id += 0
        print(f"env has reset!")
        robot_state_history = np.zeros((max_timesteps, state_dim))
        image_list = []  # for visualization
        depth_list = []

        for t in range(max_timesteps):

            if (t + 2) % 200000 == 0:
                a = input(f"q means specify {RED}new Sub Step instructions{RESET}:")
                if a == 'q':
                    lang_in = input("Input new instructions(q and enter mean using default):")
                    if lang_in != 'q' or lang_in != '':
                        raw_lang = lang_in
                        print(raw_lang)

            obs = deploy_env.get_obs()

            cur_state_np_raw, robot_state, traj_rgb_np, traj_depth_np = get_obs(obs, camera_views=policy_config['camera_views'])

            depth_list.append(traj_depth_np)
            robot_state_history[t] = cur_state_np_raw

            robot_state = torch.from_numpy(robot_state).float().cuda()

            # todo add resize&crop to wrist camera
            if t % query_frequency == 0:
                curr_image = torch.from_numpy(traj_rgb_np).float().cuda()
                if rand_crop_resize:
                    print('rand crop resize is used!')
                    original_size = curr_image.shape[-2:]
                    ratio = 0.95
                    curr_image = curr_image[...,
                                 int(original_size[0] * (1 - ratio) / 2): int(original_size[0] * (1 + ratio) / 2),
                                 int(original_size[1] * (1 - ratio) / 2): int(original_size[1] * (1 + ratio) / 2)]
                    curr_image = curr_image.squeeze(0)
                    resize_transform = transforms.Resize(original_size, antialias=True)
                    curr_image = resize_transform(curr_image)
                    curr_image = curr_image.unsqueeze(0)

            image_list.append(curr_image)
            # control_timestamps["policy_start"] = time_ms()
            # if t == 0:
            #     # warm up
            #     for _ in range(2):
            #         with torch.inference_mode():
            #             batch = prepare_inputs(traj_rgb_np, cur_state_np_raw, raw_lang)
            #             action = policy.select_action(batch)
            #     print('network warm up done')
            #     continue

            if t % query_frequency == 0:
                policy.reset()

            process_time1 = time.time()
            with torch.inference_mode():
                batch = prepare_inputs(traj_rgb_np, cur_state_np_raw, raw_lang)
                action = policy.select_action(batch)

            process_time2 = time.time()

            process_t = process_time2 - process_time1
            # print(f"{RED} Execute >>{query_frequency}<< action costs {time_cur - time_pre - process_t}s. Model forward takes {process_t}s {RESET}")

            print(f'step {t}, pred action: {action}')
            if len(action.shape) == 2:
                action = action[0]
            # action[7:] = 0
            action_info = deploy_env.step(action.tolist(), mode=policy_config['control_mode'])

if __name__ == '__main__':
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>hyper parameters<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    action_head = 'unet_diffusion_policy'  # 'droid_diffusion'
    policy_config = {
        "model_path": "/home/eai/MAD-1/wjj/lerobot_pi0_results/pretrain_aloha_all_all_data_diffusion/checkpoint-100000",
        "pretrain_vlm_path": "/home/eai/Documents/wjj/results/lerobot_pi0_results/paligemma-3b-pt-224-tokenizer", # for loading tokenizer
        "enable_lora": True,
        "temp_agg": False,
        "action_head": action_head,
        "camera_views": 3,
        'control_mode': 'absolute',  # absolute

    }
    global im_size
    im_size = 480  # default 480
    raw_lang = 'Grasp the left hem and left sleeve, then fold them forward.'
    # raw_lang = 'Pull back to flatten the fabric.'
    # raw_lang = 'Grasp the right hem and right sleeve, then fold them backward.'
    # raw_lang = 'Fold to the left.'
    # raw_lang = 'Wrap to the right.'
    # raw_lang = 'Move the folded cloth to right.'
    raw_lang = 'Fold the t-shirt on the table.'
    # raw_lang = 'The crumpled shirts are in the basket. Pick it and fold it.'


    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>hyper parameters<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    policy = PI0Policy.from_pretrained(policy_config['model_path'])
    policy.language_tokenizer = AutoTokenizer.from_pretrained(policy_config['pretrain_vlm_path'])

    sys.path.insert(0, "/home/eai/Dev-Code/mirocs")
    from run.agilex_robot_env import AgilexRobot

    agilex_bot = AgilexRobot()

    eval_bc(policy, agilex_bot, policy_config, raw_lang=raw_lang)

    print()
    exit()

