import copy
import os.path
import time
import h5py
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('/gpfs/private/tzb/wjj/projects/dvla_mh_qwen2_vla')
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

    assert len(diff_idxs) == 6, f"{RED} The diff_idxs must be  | {diff_idxs} | {file_name} {RESET}"

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

    template_substeps = dict(
        rubbish="The <obj> belongs to rubbish, pick it into right rubbish bin.",
        tableware="The <obj> belongs to tableware, pick it into white tray.",
    )

    template_instruction = "Clean table."
    tasks = [
        # 'clean_table_zmj_1217_green_plate_coke_can_brown_mug_bottle', #check
        # 'clean_table_ljm_1217', #check
        # 'clean_table_lxy_1220_blue_plate_pink_paper_cup_plastic_bag_knife', #check
        # 'clean_table_zzy_1220_green_paper_cup_wulong_bottle_pink_bowl_brown_spoon', #check
        'clean_table_zmj_1220_green_cup_blue_paper_ball_pink_plate_sprite', #check
    ]
    DATA_DIR = '/gpfs/private/tzb/wjj/data/aloha_bimanual/aloha_4views/'

    OUTPUT=os.path.join(DATA_DIR)

    for t in tasks:
        print(f">>>>>>>>>>>>>>>>>>>Processing {t}<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        task_dir = os.path.join(DATA_DIR, t)
        episodes_path = os.path.join(DATA_DIR, t)
        episodes = os.listdir(episodes_path)
        episodes = [f for f in episodes if f.endswith('.hdf5')]
        episodes = sorted(episodes, key=lambda x: int(x.split('.')[0].split('_')[-1]))
        i = 0

        with open(os.path.join(task_dir, 'annotations', 'annotations.json'), 'r') as f:
            all_results = json.load(f)

        for i, episode in enumerate(tqdm(episodes[:])):
            print(f"Processing {episode}")
            # print(file_name)
            # if "hdf5" not in file_name:
            #     continue
            substeps = []
            objects = all_results[episode]['object_sequence']
            objects = [each[0] for each in objects]

            if i%5 == 0:
                save_or_not = True
            else:
                save_or_not = False
            for obj in objects:
                obj = obj.strip()
                if obj in RUBBISH:
                    substeps.append(template_substeps['rubbish'].replace('<obj>', obj))
                elif obj in TABLEWARE:
                    substeps.append(template_substeps['tableware'].replace('<obj>', obj))
                else:
                    raise f"{obj} is not a rubbish or tableware"
            substeps.append('All clear, waiting for next task.')
            assert len(substeps) == 5, f"{substeps} | The substeps must be 5. | {objects}"

            with open(os.path.join(episodes_path, "annotations", 'annotations.json'), 'r') as f:
                all_results = json.load(f)
            deal_subtask_label(path=episodes_path, sub_reasonings=substeps, file_name=episode, raw_lang=template_instruction,
                               save_video=save_or_not)

            i += 1
            # deal_pkl_label(path=file_path,file_name=file_name,save_path=save_path)


