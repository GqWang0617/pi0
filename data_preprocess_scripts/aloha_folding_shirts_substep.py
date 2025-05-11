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
import concurrent.futures
import shutil

def is_normal_data(subtask_labels):
    if not 10 in subtask_labels and max(subtask_labels) <= 5:
        return True
    elif 10 in subtask_labels and max(subtask_labels) != 100:
        return True

    return False

def add_substep_for_double_pedal(subtask_labels, sub_reasonings, recover=False):
    targets = []
    subtask_idxs = np.where(subtask_labels != 0)[0]
    subtask_idxs = [-1] + subtask_idxs.tolist() + [len(subtask_labels) - 1]
    substep_idx = 0
    for i in range(len(subtask_idxs) - 1):
        start = subtask_idxs[i]
        end = subtask_idxs[i + 1]
        if recover and subtask_labels[start][0] == 100:
            if substep_idx <= 2:  # first 3 steps mainly recover on grasp failed
                targets.extend([sub_reasonings['recover_steps'][0]] * (end - start))
            else:
                targets.extend([sub_reasonings['recover_steps'][1]] * (end - start))
        else:
            targets.extend([sub_reasonings['standard_substeps'][substep_idx]] * (end - start))
            substep_idx += 1
    return targets

def add_substep_for_double_pedal_v2(subtask_labels, sub_reasonings, recover):
    """ add more substeps for double pedal such as lift up and check!!!Note in v2 folding shirt, recover label 100 means split. And no recover."""
    targets = []
    subtask_idxs = np.where(subtask_labels != 0)[0]
    split_idxs = np.where(subtask_labels == 100)[0] # this is using for split trajs
    subtask_idxs = [-1] + subtask_idxs.tolist() + [len(subtask_labels) - 1]
    substep_idx = 0
    for i in range(len(subtask_idxs) - 1):
        start = subtask_idxs[i]
        end = subtask_idxs[i + 1]

        if start == split_idxs[0]:
            targets.extend([sub_reasonings['recurrent_step'][0]] * (split_idxs[-1] - split_idxs[0]))
            pass
            ## recurrent substeps
        elif start > split_idxs[0] and end <= split_idxs[-1]:
            pass
        else:
            targets.extend([sub_reasonings['standard_substeps'][substep_idx]] * (end - start))
            substep_idx += 1
    return targets

def deal_subtask_label(path:str='/Volumes/PSSD-8/ljm/5_28_put_rubbish_can_succ_t0001_s-0-0/episode_12.hdf5',recover_labels='', sub_reasonings=None, file_name=None,save_path=None, raw_lang=None, save_imgs=False, output=None)->None:
    s1 = time.time()
    # paths = path.split('/')
    # paths[-1] = f"low_res_{paths[-1]}"
    # save_path = '/'.join(paths)
    save_path = os.path.join(path, "videos")
    os.makedirs(save_path, exist_ok=True)
    # if os.path.exists(os.path.join(save_path, f"{file_name}")):
    #     return 0
    qpos, qvel, effort, action, subtask_labels, image_dict=load_hdf5(path, file_name)
    print("load successfully.")
    # return 0
    sub_id = 0
    targets = []
    save_video = True
    recover = False
    if is_normal_data(subtask_labels): # standard
        if 10 in subtask_labels:
            print(f'{RED}>>Double<<{RESET} No recover label for {path}/{file_name}')
            targets = add_substep_for_double_pedal_v2(subtask_labels, sub_reasonings, recover)
        else:
            print(f'>>Single<< No recover label for {path}/{file_name}')
            diff = subtask_labels[1:] - subtask_labels[:-1]
            diff_idxs = np.argwhere(diff != 0)
            diff_idxs = [[-1, 0]] + diff_idxs.tolist() + [[len(subtask_labels) - 1, 0]]
            for i in range(len(diff_idxs) - 1):
                start = diff_idxs[i][0] + 1
                end = diff_idxs[i + 1][0] + 1
                try:
                    targets.extend([sub_reasonings['standard_substeps'][i]] * (end - start))
                except:
                    print(f"{RED} {i}-{end}-{start}:{diff_idxs}:{max(subtask_labels)}:{sum(subtask_labels)} {RESET}")
                    exit(0)

    else: # need recover
        recover = True
        if 100 in subtask_labels:
            print(f'This is {RED} double step label {RESET} {path}/{file_name}')
            targets = add_substep_for_double_pedal_v2(subtask_labels, sub_reasonings, recover)

        else:
            print(f'This is {BLUE} single step label {RESET} {path}/{file_name}')
            recover_labels = recover_labels.split(',')

            diff = subtask_labels[1:] - subtask_labels[:-1]
            diff_idxs = np.argwhere(diff != 0)
            diff_idxs = [[-1, 0]] + diff_idxs.tolist() + [[len(subtask_labels) - 1, 0]]
            substep_idx = 0

            if len(recover_labels) + 6 < len(diff_idxs) - 1:
                print(f'{RED} Missing labels for {path}/{file_name} {RESET}')
                return
            # print(diff_idxs, recover_labels)
            # exit(0)
            recover_idxs = []
            for i in range(len(diff_idxs) - 1):
                start = diff_idxs[i][0] + 1
                end = diff_idxs[i + 1][0] + 1
                if str(i+1) in recover_labels: # needs recover
                    if substep_idx <= 3: # first 3 steps mainly recover on grasp failed default to 2
                        recover_id = 0
                        recover_idxs.append([substep_idx, end, diff_idxs[i + 2][0] + 1])

                    else:
                        recover_id = 1
                    if recover_id == 0:
                        targets.extend([sub_reasonings['standard_substeps'][substep_idx]] * (end - start - 8))
                        targets.extend([sub_reasonings['recover_steps'][recover_id]] * 8)
                    else:
                        targets.extend([sub_reasonings['recover_steps'][recover_id]] * (end - start))

                else:
                    try:
                        targets.extend([sub_reasonings['standard_substeps'][substep_idx]] * (end - start))
                    except:
                        print(f"{RED} {substep_idx}-{end}-{start}:{diff_idxs}|recover_labels:{recover_labels}|max:{max(subtask_labels)} {RESET}")
                        exit(0)
                    substep_idx += 1

            for substep, s, e in recover_idxs:
                if substep <= 2:
                    recover_label = sub_reasonings['recover_steps'][0]
                else:
                    recover_label = sub_reasonings['recover_steps'][1]
                for j in range(s, e):
                    targets[j] = recover_label

    if save_video and (int(file_name.split('.')[0].split('_')[-1]) % 10 == 0):
        imgs = []
        font = ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationSans-BoldItalic.ttf', size=16)
        # os.makedirs(os.path.join(save_path, file_name.split('.')[0]), exist_ok=True)
        fills = [(255,0,0), (0,0, 255)]

        for i,img in enumerate(image_dict['cam_high'][:]):
            # image_pil = Image.open(os.path.join(path, f"{file_name.split('.')[0]}.jpg"))
            image_pil = Image.fromarray(img)
            draw = ImageDraw.Draw(image_pil)
            if 'recover_steps' in sub_reasonings.keys() and targets[i] in sub_reasonings['recover_steps']:
                color = fills[0]
            else:
                color = fills[1]

            draw.text((120, 160), targets[i], font=font, fill=color)
            imgs.append(image_pil)

            # image_pil.save(os.path.join(save_path, file_name.split('.')[0], f"{i}.jpg"))
        if recover:
            imageio.mimsave(os.path.join(save_path, f"recover_{file_name.split('.')[0]}.mp4"), imgs, fps=25)
        else:
            imageio.mimsave(os.path.join(save_path, f"{file_name.split('.')[0]}.mp4"), imgs, fps=25)

    s2 = time.time()

    # 在循环外创建线程池

    s3 = time.time()

    with h5py.File(os.path.join(path, file_name), 'a') as hdf5_file:
        if 'substep_reasonings' in hdf5_file.keys():
            del hdf5_file['substep_reasonings']
        hdf5_file.create_dataset("substep_reasonings", data=targets)

        if "language_raw" not in hdf5_file.keys():
            hdf5_file.create_dataset('language_raw', data=[raw_lang])
        else:
            hdf5_file['language_raw'][()] = raw_lang


    # generate_h5(dict_data, os.path.join(save_path,f"{file_name}"))
    s4 = time.time()

    # print(f'{file_name} done. Time taken: Copy h5 to new: {s4 - s3:.2f} seconds, Resize imgs:{s3 - s2:.2f} seconds, Adding Substep reasoning: {s2 - s1:.2f} seconds')

def deal_subtask_label_without_padel(path='', sub_reasonings=None, file_name=None,save_path=None, raw_lang=None, save_imgs=False, output=None)->None:
    s1 = time.time()

    save_path = os.path.join(path, "videos")
    os.makedirs(save_path, exist_ok=True)

    qpos, qvel, effort, action, subtask_labels, image_dict=load_hdf5(path, file_name)
    print("load successfully.")
    # return 0
    sub_id = 0
    targets = []
    save_video = True

    print(f'{RED}>>None Padel<<{RESET} for {path}/{file_name}')
    griper_action_np = action[:, 6::7]  # 6 means left gripper, 13 means right gripper
    mean_gripper = np.mean(griper_action_np, axis=0)  # calculate the mean value of gripper
    index_griper = griper_action_np > mean_gripper
    index_griper = index_griper.astype(int)
    xor_diff = np.abs(index_griper[:-1] - index_griper[1:])
    subtask_idx_left = np.where(xor_diff[:, 0] == 1)[0] + 1
    subtask_idx_right = np.where(xor_diff[:, 1] == 1)[0] + 1

    diff_idxs = subtask_idx_left + subtask_idx_right
    diff_idxs = sorted(diff_idxs)

    for i in range(len(diff_idxs) - 1):
        start = diff_idxs[i][0] + 1
        end = diff_idxs[i + 1][0] + 1
        try:
            targets.extend([sub_reasonings['standard_substeps'][i]] * (end - start))
        except:
            print(f"{RED} {i}-{end}-{start}:{diff_idxs}:{max(subtask_labels)}:{sum(subtask_labels)} {RESET}")
            exit(0)

    if save_video:
        imgs = []
        font = ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationSans-BoldItalic.ttf', size=16)
        # os.makedirs(os.path.join(save_path, file_name.split('.')[0]), exist_ok=True)
        fills = [(255,0,0), (0,0, 255)]

        for i,img in enumerate(image_dict['cam_high'][:]):
            # image_pil = Image.open(os.path.join(path, f"{file_name.split('.')[0]}.jpg"))
            image_pil = Image.fromarray(img)
            draw = ImageDraw.Draw(image_pil)
            if targets[i] in sub_reasonings['recover_steps']:
                color = fills[0]
            else:
                color = fills[1]

            draw.text((120, 160), targets[i], font=font, fill=color)
            imgs.append(image_pil)

            # image_pil.save(os.path.join(save_path, file_name.split('.')[0], f"{i}.jpg"))

        imageio.mimsave(os.path.join(save_path, f"{file_name.split('.')[0]}.mp4"), imgs, fps=25)

    s2 = time.time()

    # 在循环外创建线程池

    s3 = time.time()

    with h5py.File(os.path.join(path, file_name), 'a') as hdf5_file:
        if 'substep_reasonings' in hdf5_file.keys():
            del hdf5_file['substep_reasonings']
        hdf5_file.create_dataset("substep_reasonings", data=targets)

        if "language_raw" not in hdf5_file.keys():
            hdf5_file.create_dataset('language_raw', data=[raw_lang])
        else:
            hdf5_file['language_raw'][()] = raw_lang


    # generate_h5(dict_data, os.path.join(save_path,f"{file_name}"))
    s4 = time.time()

    # print(f'{file_name} done. Time taken: Copy h5 to new: {s4 - s3:.2f} seconds, Resize imgs:{s3 - s2:.2f} seconds, Adding Substep reasoning: {s2 - s1:.2f} seconds')

def linear_process(file_list, file_path, template_substeps, template_instruction, OUTPUT, using_recover_label):
    i = 0
    recover_label_path = os.path.join(RECOVER_PATH, f"{t}_label.txt")
    if using_recover_label:
        recover_labels = {}
        with open(os.path.join(recover_label_path), 'r') as f:
            for line in f:
                k, v = line.strip().split(":")
                recover_labels[k] = v.strip()
    for file_name in tqdm(file_list[:]):
        # print(file_name)
        # if "hdf5" not in file_name:
        #     continue
        try:
            r_l = recover_labels[file_name.split('.')[0]]
        except Exception as e:
            # print(e)
            r_l = None
        save_or_not = True
        # with h5py.File(os.path.join(file_path, file_name)) as root:
        #     if 'substep_reasonings' in root:
        #         print(f'{file_name} already done.')
        #         continue
        # for episodes with padel signals
        deal_subtask_label(path=file_path, sub_reasonings=template_substeps, file_name=file_name,
                           raw_lang=template_instruction, recover_labels=r_l, save_imgs=save_or_not, output=OUTPUT)
        #
        # deal_subtask_label_without_padel(path=file_path, sub_reasonings=template_substeps, file_name=file_name, raw_lang=template_instruction,save_imgs=save_or_not, output=OUTPUT)
        i += 1
        # deal_pkl_label(path=file_path,file_name=file_name,save_path=save_path)

def mulit_process(file_list, file_path, template_substeps, template_instruction, OUTPUT, using_recover_label):
    i = 0
    recover_label_path = os.path.join(RECOVER_PATH, f"{t}_label.txt")
    if using_recover_label:
        recover_labels = {}
        with open(os.path.join(recover_label_path), 'r') as f:
            for line in f:
                k, v = line.strip().split(":")
                recover_labels[k] = v.strip()

    def _process_file(file_name):
        try:
            r_l = recover_labels.get(file_name.split('.')[0], None)
        except Exception as e:
            # print(f"Error in recovering labels for {file_name}: {e}")
            r_l = None

        try:
            deal_subtask_label(
                path=file_path,
                sub_reasonings=template_substeps,
                file_name=file_name,
                raw_lang=template_instruction,
                recover_labels=r_l,
                save_imgs=True,  # Directly pass True
                output=OUTPUT
            )
            return "None"
        except Exception as e:
            print(f"{RED} Error {RESET} in processing {file_name}: {e}")
            temps = file_path.split("/")
            wrong_path = os.path.join("/".join(temps[:-1]), "wrong", temps[-1])
            os.makedirs(wrong_path, exist_ok=True)
            shutil.move(os.path.join(file_path, file_name), os.path.join(wrong_path, file_name))
            print(f"Moved {file_name} to {wrong_path}")
            return file_name

    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = []
        for episode in tqdm(file_list[:]):
            futures.append(executor.submit(_process_file, episode))

        # Optional: Wait for all tasks to complete
        for future in futures:
            try:
                result = future.result()  # Retrieve exceptions from threads
                if result != "None":
                    print(f"{BLUE} ERROR in {result} {RESET}")
            except Exception as e:
                print(f"Error in thread: {e}")


if __name__=="__main__":

    from template_substeps import template_substeps
    using_recover_label = False
    template_instruction = "Fold the shirt on the table."
    if 'raw_language' in template_substeps.keys():
        template_instruction = template_substeps['raw_language']
    tasks = [
        # 'fold_shirt_wjj1213_meeting_room',
        # 'fold_shirt_zzy1213',
        # 'fold_shirt_zmj1212',
        # 'fold_shirt_zmj1213',
        # 'fold_shirt_lxy1213',
        # 'fold_shirt_lxy1214'
        # 'fold_tshirts_129' # for local debug
        # 'folding_junjie_1224',
        # 'folding_zhongyi_1224'.
        # 'folding_shirt_12_28_zzy_right_first',
        # 'folding_shirt_12_27_office'
        # 'folding_shirt_12_30_wjj_weiqing_recover',
        # 'folding_shirt_12_30_yichen_weiqing_recover', # only 3
        # 'folding_shirt_12_31_wjj_lab_marble_recover',
        # 'folding_shirt_12_31_zhouzy_lab_marble',
        # 'folding_blue_tshirt_yichen_0102',
        # 'folding_blue_tshirt_xiaoyu_0103',
        # 'folding_blue_tshirt_yichen_0103',
        # "0107_ljm_folding_blue_shirt",
        # "0107_wjj_folding_blue_shirt",

        # "folding_second_tshirt_yichen_0108",
        # "folding_second_tshirt_wjj_0108",
        # "folding_random_yichen_0109",
        # "folding_random_tshirt_yichen_0104", # unused
        # "folding_random_table_right_wjj_0109",
        # "folding_basket_two_tshirt_yichen_0109",
        # "folding_basket_second_tshirt_yichen_0110",
        # "folding_basket_second_tshirt_yichen_0109",
        # "folding_basket_second_tshirt_wjj_0110",
        # 'folding_basket_second_tshirt_yichen_0111',
        # 'folding_basket_second_tshirt_wjj_0113',
        # 'folding_basket_second_tshirt_wjj_0111',

        # 'folding_basket_second_tshirt_yichen_0114'

        # "weiqing_folding_basket_first_tshirt_dark_blue_yichen_0116",
        # "weiqing_folding_basket_first_tshirt_pink_wjj_0115",
        # "weiqing_folding_basket_second_tshirt_blue_yichen_0115",
        # "weiqing_folding_basket_second_tshirt_dark_blue_yichen_0116",
        # "weiqing_folding_basket_second_tshirt_red_lxy_0116",
        # "weiqing_folding_basket_second_tshirt_red_wjj_0116",
        # "weiqing_folding_basket_second_tshirt_shu_red_yellow_wjj_0116",
        # "weiqing_folding_basket_second_tshirt_yellow_shu_red_wjj_0116",
        # "weiqing_folding_basket_second_yellow_blue_wjj_0117",
        # "weiqing_folding_basket_first_yellow_blue_wjj_0117",
        # "weiqing_folding_basket_second_dark_blue_polo_to_blue_shirt_lxy_0117",
        # "weiqing_folding_basket_second_dark_blue_shirt_to_polo_lxy_0118"
        # "folding_random_short_first_wjj_0121",
        # "folding_random_short_second_wjj_0121"
        # "folding_random_short_first_wjj_0122",
        # "folding_random_short_second_wjj_0122",
        "folding_random_tshirt_second_wjj_0124",
        "folding_random_tshirt_first_wjj_0124",
    ]
    DATA_DIR = '/home/jovyan/tzb/wjj/data/aloha_bimanual/aloha_4views/1_24_folding_7z_extract'
    RECOVER_PATH = '/home/jovyan/tzb/wjj/data/aloha_bimanual/aloha_4views/fold_shirt_yichen_label_1217/'


    OUTPUT=os.path.join(DATA_DIR)
    for t in tasks:
        s = time.time()
        # task_dir = os.path.join(DATA_DIR, t)
        file_path = os.path.join(DATA_DIR, t)
        file_list = os.listdir(file_path)
        file_list = [f for f in file_list if f.endswith('.hdf5')]
        file_list = sorted(file_list, key=lambda x: int(x.split('.')[0].split('_')[-1]))
        if 'second' in t:
            template_substeps['standard_substeps'].extend(template_substeps['stack_steps'])
        else:
            template_substeps['standard_substeps'].extend(template_substeps['push_steps'])
        # print(template_substeps['standard_substeps'])
        # exit(0)
        # linear_process(file_list[0:2], file_path, template_substeps, template_instruction, OUTPUT, using_recover_label)
        mulit_process(file_list, file_path, template_substeps, template_instruction, OUTPUT, using_recover_label)
        print(f"{BLUE} Done. Time taken: {time.time() - s:.2f} seconds {RESET}")

