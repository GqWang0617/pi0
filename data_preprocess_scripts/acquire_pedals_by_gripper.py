import h5py
import os
from visualize_recover_data import load_hdf5
import numpy as np
from tqdm import tqdm

def rules_hang_cups_waibao(subtask_idx_left, subtask_idx_right, length):
    # subtask_idx_left = [e for e in subtask_idx_right]
    pedals = []
    # if subtask_idx_right[0] < 80:
    #     pedals = subtask_idx_left[1:3]
    #
    # else:
    #     pedals = subtask_idx_left[0::2]
    # pedals[0] += 6
    # pedals[1] -= 6
    pedals.append(subtask_idx_left[2] + 6)
    pedals.append(subtask_idx_right[2] + 6)

    return pedals

def rules_pick_cup_and_pour_coffee(subtask_idx_left, subtask_idx_right, length):
    # subtask_idx_left = [e for e in subtask_idx_right]
    pedals = []
    # if subtask_idx_right[0] < 80:
    #     pedals = subtask_idx_left[1:3]
    #
    # else:
    #     pedals = subtask_idx_left[0::2]
    # pedals[0] += 6
    # pedals[1] -= 6
    pedals.append(subtask_idx_left[2] + 6)
    pedals.append(subtask_idx_right[2] + 6)

    return pedals

def modify_subtask_in_h5py(ep_p, pedal, length):
    subtask = [[0]] * length
    with h5py.File(ep_p, 'a') as f:
        # subtask = f['subtask']
        for p in pedal:
            subtask[p] = [10]
        f['subtask'][()] = subtask


def main_acquire_pedals_by_gripper():
    data_path = "/home/jovyan/tzb/wjj/data/aloha_bimanual/aloha_4views/aloha_data/"

    tasks = [
        # 'hang_cups_waibao',
        # 'pick_cars_from_moving_belt_zhumj_1227',
        # 'pick_cars_from_moving_belt_waibao_1227',

        # 'pick_cup_and_pour_water_wjj_weiqing_coffee' # the gripper value may bigger than mean_gripper when the bottle almost equals to the width of gripper
        'pick_cup_and_pour_water_wjj_weiqing_coke'
    ]

    for task in tasks:
        task_path = os.path.join(data_path, task)
        episodes = os.listdir(task_path)
        episodes = [f for f in episodes if f.endswith('.hdf5')]
        episodes = sorted(episodes, key=lambda x: int(x.split('.')[0].split('_')[-1]))

        for episode in tqdm(episodes[20:]):
            qpos, qvel, effort, action, subtask_labels, image_dict = load_hdf5(task_path, episode)
            griper_action_np = action[:, 6::7]  # 6 means left gripper, 13 means right gripper
            mean_gripper = np.mean(griper_action_np, axis=0) /2  # calculate the mean value of gripper
            index_griper = griper_action_np > mean_gripper
            index_griper = index_griper.astype(int)
            xor_diff = np.abs(index_griper[:-1] - index_griper[1:])
            subtask_idx_left = np.where(xor_diff[:, 0] == 1)[0] + 1
            subtask_idx_right = np.where(xor_diff[:, 1] == 1)[0] + 1

            # print(subtask_idx_left, subtask_idx_right)
            # exit(0)

            if task == 'hang_cups_waibao':
                assert len(
                    subtask_idx_right) == 4, f'subtask_idx_right should be a list with 4 elements::{subtask_idx_right}::{task_path}/{episode}'
                pedal = rules_hang_cups_waibao(subtask_idx_left, subtask_idx_right, len(index_griper))
            elif task == 'pick_cars_from_moving_belt_zhumj_1227' or task == 'pick_cars_from_moving_belt_waibao_1227':
                assert len(
                    subtask_idx_right) == 4, f'subtask_idx_right should be a list with 4 elements::{subtask_idx_right}::{task_path}/{episode}'
                pedal = rules_hang_cups_waibao(subtask_idx_left, subtask_idx_right, len(index_griper))
            elif task == 'pick_cup_and_pour_water_wjj_weiqing_coffee' or task == 'pick_cup_and_pour_water_wjj_weiqing_coke':
                # print(np.round(griper_action_np, 3))
                # print(mean_gripper)
                assert len(
                    subtask_idx_right) == 4, f'subtask_idx_right should be a list with 4 elements::{subtask_idx_right}::{task_path}/{episode}'
                pedal = rules_pick_cup_and_pour_coffee(subtask_idx_left, subtask_idx_right, len(index_griper))

            modify_subtask_in_h5py(os.path.join(task_path, episode), pedal, len(index_griper))

if __name__ == '__main__':
    main_acquire_pedals_by_gripper()

