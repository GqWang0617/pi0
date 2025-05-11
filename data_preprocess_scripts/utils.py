import copy
import os.path
import time
import h5py
import numpy as np
from tqdm import tqdm
import fnmatch
import sys
sys.path.append('/gpfs/private/tzb/wjj/projects/dvla_mh_qwen2_vla')
from aloha_scripts.constants import TASK_CONFIGS
import os
from tqdm import tqdm
import json

def down_sample_episodes(path, step):
    data=h5py.File(path,'a')

    stack = [(data)]

    while stack:
        cur_data = stack.pop()
        for name, item in cur_data.items():

            if isinstance(item, h5py.Group):
                stack.append((item))
            else:
                if item[:].shape[0] > 300:
                    sample_data = item[0::step]
                    del cur_data[name]
                    cur_data.create_dataset(name, data=sample_data)
                else:
                    print(item[:].shape)
    data.close()

def main_down_sample(DATA_DIR, ignore):
    files = os.listdir(DATA_DIR)

    target = [
        # '1105_2358_stack_cup', '1105_2059_stack_cup'
        'low_res_plate_blue_plastic_waste_spoon_paper_drink_box'
              ]
    for file in tqdm(files):
        if file not in target:
            continue
        print(f"Current task: {file}")
        file_path = os.path.join(DATA_DIR, file)
        if not os.path.isdir(os.path.join(DATA_DIR, file)):
            continue
        if file in ignore:
            continue
        file_list = os.listdir(file_path)
        file_list = [f for f in file_list if f.endswith('.hdf5')]
        file_list = sorted(file_list, key=lambda x: int(x.split('.')[0].split('_')[-1]))
        i = 0
        for file_name in tqdm(file_list[:]):
            try:
                down_sample_episodes(os.path.join(file_path, file_name), step=3)
            except Exception as e:
                print(e)
                print(file, file_name)
            i += 1

def get_norm_stats(dataset_path_list):
    all_qpos_data = []
    all_action_data = []
    all_episode_len = []

    for dataset_path in dataset_path_list:
        try:
            with h5py.File(dataset_path, 'r') as root:
                qpos = root['/observations/qpos'][()]
        except Exception as e:
            print(f'Error loading {dataset_path} in get_norm_stats')
            print(e)
            quit()
        all_episode_len.append(len(qpos))

    return None, all_episode_len
def find_all_hdf5(dataset_dir, skip_mirrored_data):
    hdf5_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for filename in fnmatch.filter(files, '*.hdf5'):
            if 'features' in filename: continue
            if skip_mirrored_data and 'mirror' in filename:
                continue
            hdf5_files.append(os.path.join(root, filename))
    print(f'Found {len(hdf5_files)} hdf5 files')
    return hdf5_files
def sum_episodes_length():
    tasks = TASK_CONFIGS['11_1_reasoning_all_tasks']['dataset_dir']
    skip_mirrored_data = False
    data = {}
    for task in tasks:
        dataset_path_list = find_all_hdf5(task, skip_mirrored_data)

        print("*"*60)
        task_name= dataset_path_list[0].split('/')[-2]
        print(task_name)
        _, all_episode_len = get_norm_stats(dataset_path_list)
        # print(all_episode_len)
        # print("*"*60)
        data[task_name] = all_episode_len

    target_task = ['bread', 'pick_pot', 'pick_kettle', 'tape']
    target_task = ['pot_left', 'pot_right']
    target_task = ['arrange', 'types']
    target_task = ['blue_cube_yellow_box']
    target_task = ['cube_right', 'cube_left']

    temp = []
    for k, v in data.items():
        if any(x in k for x in target_task):
            temp += v
    # print(temp)
    print(f"Episodes: {len(temp)}, Average length: {np.mean(np.array(temp))}")

def add_language_to_hdf5(file_path, raw_lang): # add language instructions
    # Open the HDF5 file in read/write mode
    print(file_path)
    with h5py.File(file_path, 'a') as hdf5_file:
        if "language_raw" not in hdf5_file.keys():
            hdf5_file.create_dataset("language_raw", data=[raw_lang])
        else:
            # hdf5_file.attrs["language_raw"] = raw_lang
            # del hdf5_file.attrs['language_raw']
            hdf5_file.attrs['language_raw'] = hdf5_file['language_raw'][()]
            hdf5_file['language_raw'][()] = [raw_lang]


def add_reasoning_language_to_hdf5(file_path, reasoning): # add language reasonings
    print(file_path)
    with h5py.File(file_path, 'a') as hdf5_file:
        if "reasoning" not in hdf5_file.keys():
            hdf5_file.create_dataset("reasoning", data=[reasoning])
        else:
            # hdf5_file.attrs["language_raw"] = raw_lang
            # del hdf5_file.attrs['language_raw']
            hdf5_file['reasoning'][()] = reasoning
    # hdf5_file.attrs["reasoning"] =
    pass

def add_distilbert_embedding_to_hdf5(file_path, encoded_lang): # add language distillbert embedding
    # Open the HDF5 file in read/write mode
    print(file_path)
    with h5py.File(file_path, 'a') as hdf5_file:
        if "language_distilbert" not in hdf5_file.keys():
            # hdf5_file.create_dataset("language_embed", data=language_embed)
            hdf5_file.create_dataset("language_distilbert", data=encoded_lang.cpu().detach().numpy())
        else:
            # hdf5_file.attrs["language_raw"] = raw_lang
            # del hdf5_file.attrs['language_raw']
            del hdf5_file["language_distilbert"]
            hdf5_file.create_dataset("language_distilbert", data=encoded_lang.cpu().detach().numpy())

def extract_meta_info(root='/gpfs/private/tzb/wjj/data/droid_with_distilbert_lang_three_view'):
    droid_path = f'{root}/droid_with_distilbert_lang_three_view_succ_t0001_s-0-0'
    qa_path = f'{root}/results_all/'

    import pickle
    pkls = os.listdir(qa_path)

    meta_info = {}

    h5pys = os.listdir(droid_path)
    for h in tqdm(h5pys):
        ep_p = os.path.join(droid_path, h)
        with h5py.File(ep_p, 'r') as hdf5_file:
            meta = hdf5_file['episode_metadata_file_path'][:][0].decode('utf-8')
        # print(meta)
        meta_info[h] = meta

    with open(os.path.join(root, 'meta.json'), 'w') as f:
        json.dump(meta_info, f, indent=4)

def show_hdf5(file_path, attr='language_raw'):
    # Open the HDF5 file in read/write mode
    print(file_path)
    with h5py.File(file_path, 'r') as hdf5_file:
        pass
        raw_lang = hdf5_file[attr][0].decode('utf-8')
        print(raw_lang, hdf5_file.keys())

if __name__=="__main__":

    DATA_DIR = '/gpfs/private/tzb/wjj/data/aloha_bimanual/'

    ignore = ['1029_place_cup_on_the_shelf', '1030_hide_spiderman', '1030_magic_cube', '1031_sweep_trash', '1031_unpack_bag_put_ball', '1030_put_light_bulb']

    # OUTPUT=os.path.join(DATA_DIR)
    main_down_sample(DATA_DIR, ignore)

