import copy
import os.path
import time
import h5py
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('/home/jovyan/tzb/wjj/projects/dvla_mh_qwen2_vla')
import os
from tqdm import tqdm
# from aloha_scripts.constants import TASK_CONFIGS
# from aloha_scripts.constants import DATA_DIR
import concurrent.futures
from PIL import Image, ImageDraw, ImageFont
import functools
import cv2
from visualize_recover_data import load_hdf5
from aloha_scripts.utils import *
import imageio
import json
TABLEWARE = "plate,mug,white cup,knife,bowl,spoon".split(',')
RUBBISH = "empty bottle,coke can,plastic bag,paper cup,spirit can,paper ball".split(',')

def deal_subtask_label(path:str='', sub_reasonings=None, file_name=None, raw_lang=None, save_video=False)->None:

    save_path = os.path.join(path, "videos")
    os.makedirs(save_path, exist_ok=True)

    qpos, qvel, effort, action, subtask_labels, image_dict=load_hdf5(path, file_name)
    print("load successfully.")

    targets = []

    try:
        diff_idxs = np.where(subtask_labels != 0)[0]
        diff_idxs = [-1] + diff_idxs.tolist() + [len(subtask_labels) - 1]
    except Exception as e:
        print(e)
    assert len(diff_idxs) == 1 + len(sub_reasonings), f"{RED}The num of substeps does not match the num of pedals.|{diff_idxs}|{path}/{file_name}{RESET}"

    for i in range(len(diff_idxs) - 1):
        start = diff_idxs[i] + 1
        end = diff_idxs[i + 1] + 1
        targets.extend([sub_reasonings[i]] * (end - start))

    if save_video:
        imgs = []
        font = ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationSans-BoldItalic.ttf', size=16)
        # os.makedirs(os.path.join(save_path, file_name.split('.')[0]), exist_ok=True)
        fills = [(255,0,0), (0,0, 255)]

        for i,img in enumerate(image_dict['cam_high'][:]):
            # image_pil = Image.open(os.path.join(path, f"{file_name.split('.')[0]}.jpg"))
            image_pil = Image.fromarray(img)
            draw = ImageDraw.Draw(image_pil)

            color = fills[0]

            draw.text((120, 160), targets[i], font=font, fill=color)
            imgs.append(image_pil)

        imageio.mimsave(os.path.join(save_path, f"{file_name.split('.')[0]}.mp4"), imgs, fps=25)


    with h5py.File(os.path.join(path, file_name), 'a') as hdf5_file:
        if 'substep_reasonings' in hdf5_file.keys():
            del hdf5_file['substep_reasonings']
        hdf5_file.create_dataset("substep_reasonings", data=targets)

        if "language_raw" not in hdf5_file.keys():
            hdf5_file.create_dataset('language_raw', data=[raw_lang])
        else:
            hdf5_file['language_raw'][()] = raw_lang

if __name__=="__main__":

    template_substeps = [
        'Open the rice cooker.',
        'Pour water and rice.',
        'Close it.'
    ]

    template_instruction = "Cook rice."
    # template_substeps = [
    #     'Pick up the paper cup and place it on table.',
    #     'Pour the coffee into the cup.',
    #     'Done.'
    # ]
    #
    # template_instruction = "Serve a cup of coffee."
    # template_substeps = [
    #     'Pick up left green cup and hang it on the rack.',
    #     'Pick up right pink cup and hang it on the rack.',
    #     'Done.'
    # ]
    # template_instruction = "Hang the cups on the rack."

    # template_substeps = [
    #     'Pick up the first chess and place it into box.',
    #     'Pick up another chess and place it into box.',
    #     'Done.'
    # ]
    # template_instruction = "Storage the chess on the moving conveyor belt."

    tasks = [
        # 'get_papercup_and_pour_coke_yichen_1224', #check
        # "hang_cups_waibao",
        # "pick_cars_from_moving_belt_zhumj_1227",
        # 'pick_cars_from_moving_belt_waibao_1227',
        # 'pick_cup_and_pour_water_wjj_weiqing_coffee' # the gripper value may bigger than mean_gripper when the bottle almost equals to the width of gripper
        # 'pick_cup_and_pour_water_wjj_weiqing_coke'
        'weiqing_kitchen_cook_rice_wjj_0118',
    ]
    DATA_DIR = '/home/jovyan/tzb/wjj/data/aloha_bimanual/aloha_4views/7z_1_18_extract'

    OUTPUT=os.path.join(DATA_DIR)

    for t in tasks:
        print(f">>>>>>>>>>>>>>>>>>>Processing {t}<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        task_dir = os.path.join(DATA_DIR, t)
        episodes_path = os.path.join(DATA_DIR, t)
        episodes = os.listdir(episodes_path)
        episodes = [f for f in episodes if f.endswith('.hdf5')]
        episodes = sorted(episodes, key=lambda x: int(x.split('.')[0].split('_')[-1]))
        i = 0


        for i, episode in enumerate(tqdm(episodes[0:])):
            print(f"Processing {episode}")

            # substeps = []

            if i%5 == 0:
                save_or_not = True
            else:
                save_or_not = False

            # template_substeps.append('All clear, waiting for next task.')

            deal_subtask_label(path=episodes_path, sub_reasonings=template_substeps, file_name=episode, raw_lang=template_instruction,
                               save_video=save_or_not)

            i += 1
            # deal_pkl_label(path=file_path,file_name=file_name,save_path=save_path)


