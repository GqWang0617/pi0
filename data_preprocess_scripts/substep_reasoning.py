import os.path

import h5py
import numpy as np
from tqdm import tqdm
import sys
sys.path.append('/gpfs/private/tzb/wjj/projects/dvla_mh_qwen2_vla')
import os
from tqdm import tqdm
from aloha_scripts.constants import TASK_CONFIGS
from aloha_scripts.constants import DATA_DIR

def nms_del_repeat(seq):
    if len(seq) == 4:
        return seq
    if len(seq) < 4:
        return 'le5'
    print('......nmsing.....', seq)
    xor_diff = seq[1:] - seq[:-1]
    xor_diff = np.where(xor_diff < 50)[0]
    for i in xor_diff:
        seq = np.delete(seq, i)
        if len(seq) == 4:
            return seq
    return seq


def deal_subtask_label(path:str='/Volumes/PSSD-8/ljm/5_28_put_rubbish_can_succ_t0001_s-0-0/episode_12.hdf5',sub_reasonings=None, file_name=None,save_path=None, raw_lang=None)->None:
    data=h5py.File(os.path.join(path,file_name),'a')
    griper_action_np=data["action"][:,9]  # 0:open 1:close
    index_griper=griper_action_np > 0.80
    index_griper = index_griper.astype(int)
    xor_diff = np.abs(index_griper[:-1] - index_griper[1:])
    subtask_idx = np.where(xor_diff == 1)[0] + 1
    subtask_idx = subtask_idx[1::2]
    subtask_idx = nms_del_repeat(subtask_idx)

    subtask_idx = np.append(np.array(0), subtask_idx)
    # return 0
    sub_id = 0
    targets = []
    for i in range(subtask_idx.shape[0] - 1):
        start_idx = subtask_idx[i]
        end_idx = subtask_idx[i+1]
        s_r = sub_reasonings[sub_id]
        # for j in range(start_idx, end_idx):
        targets.extend([s_r] * (end_idx-start_idx))
        sub_id += 1

    targets.extend([sub_reasonings[-1]] * (griper_action_np.shape[0] - subtask_idx[-1]))
    if "language_raw" not in data.keys():
        data.create_dataset("language_raw", data=[raw_lang])
    else:
        # hdf5_file.attrs["language_raw"] = raw_lang
        # del hdf5_file.attrs['language_raw']
        data.attrs['language_raw'] = data['language_raw'][()]
        data['language_raw'][()] = [raw_lang]

    if "substep_reasonings" not in data.keys():
        data.create_dataset("substep_reasonings", data=targets)
    else:
        data.attrs['substep_reasonings'] = data['substep_reasonings'][()]
        data['substep_reasonings'][()] = targets

    data.close()



if __name__=="__main__":
    # name = '10_28_reasoning_8mt'
    # print(TASK_CONFIGS.keys())
    # target_tasks = TASK_CONFIGS[name]['dataset_dir']
    DATA_DIR = "/home/jovyan/tzb/wjj/data/aloha_bimanual/aloha_4views/7z_1_18_extract"
    raw_lang = {
        '4_types_pikachu_blue_van_hex_key_glove_480_640': 'Classifying all objects and place to corresponding positions.',
        '2_types_blue_van_pink_bus_hex_key_480_640': 'Classifying all objects and place to corresponding positions.',
        '3types_bird_pikachu_white_car_hex_key_480_640': 'Classifying all objects and place to corresponding positions.',
        '3types_pink_car_hex_key2_gloves_480_640': 'Classifying all objects and place to corresponding positions.',
        '1type_blue_van_pink_bus_green_van_white_car_2_480_640': 'Classifying all objects and place to corresponding positions.',
        '4types_pig_cyan_trunk_hex_key_gloves_480_640': 'Classifying all objects and place to corresponding positions.',
        'weiqing_kitchen_cook_rice_wjj_0118': "Cook rice."
    }
    task_require_substep_reasonings = {
        # '4_types_pikachu_blue_van_hex_key_glove_480_640': [
        #     'The toy pikachu belongs to stuffed toys which locate at top-right of box. Place it on top-right of box.',
        #     'The toy blue van belongs to car toys which locate at bottom-left of box. Place it on bottom-left of box.',
        #     'The hex key belongs to hardware tools which locate at bottom-right of box. Place it on bottom-right of box.',
        #     'The gloves belong to protective equipments which locate at top-left of box. Place it on top-left of box.',
        #     'Wait for next task.'],
        # '2_types_blue_van_pink_bus_hex_key_480_640': [
        #     'The toy blue van belongs to car toys which locate at bottom-left of box. Place it on bottom-left of box.',
        #     'The toy pink bus belongs to car toys which locate at bottom-left of box. Place it on bottom-left of box.',
        #     'The hex key belongs to hardware tools which locate at bottom-right of box. Place it on bottom-right of box.',
        #     'The hex key belongs to hardware tools which locate at bottom-right of box. Place it on bottom-right of box.',
        #     'Wait for next task.']
        # '3types_bird_pikachu_white_car_hex_key_480_640': [
        #     'The toy bird belongs to stuffed toys which locate at top-right of box. Place it on top-right of box.',
        #     'The toy pikachu belongs to stuffed toys which locate at top-right of box. Place it on top-right of box.',
        #     'The toy white car belongs to car toys which locate at bottom-left of box. Place it on bottom-left of box.',
        #     'The hex key belongs to hardware tools which locate at bottom-right of box. Place it on bottom-right of box.',
        #     'Wait for next task.'],
        # '1types_blue_van_pink_bus_green_van_white_car_480_640': [
        #     'The toy blue van belongs to car toys which locate at bottom-left of box. Place it on bottom-left of box.',
        #     'The toy pink bus belongs to car toys which locate at bottom-left of box. Place it on bottom-left of box.',
        #     'The toy green van belongs to car toys which locate at bottom-left of box. Place it on bottom-left of box.',
        #     'The toy white car belongs to car toys which locate at bottom-left of box. Place it on bottom-left of box.',
        #     'Wait for next task.'],
        # '3types_pink_car_hex_key2_gloves_480_640': [
        #     'The toy pink bus belongs to car toys which locate at bottom-left of box. Place it on bottom-left of box.',
        #     'The hex key belongs to hardware tools which locate at bottom-right of box. Place it on bottom-right of box.',
        #     'The hex key belongs to hardware tools which locate at bottom-right of box. Place it on bottom-right of box.',
        #     'The gloves belong to protective equipments which locate at top-left of box. Place it on top-left of box.',
        #     'Wait for next task.'],
        # '1type_blue_van_pink_bus_green_van_white_car_2_480_640': [
        #     'The toy blue van belongs to car toys which locate at bottom-left of box. Place it on bottom-left of box.',
        #     'The toy pink bus belongs to car toys which locate at bottom-left of box. Place it on bottom-left of box.',
        #     'The toy green van belongs to car toys which locate at bottom-left of box. Place it on bottom-left of box.',
        #     'The toy white car belongs to car toys which locate at bottom-left of box. Place it on bottom-left of box.',
        #     'Wait for next task.'],
        # '4types_pig_cyan_trunk_hex_key_gloves_480_640': [
        #     'The toy pink pig belongs to stuffed toys which locate at top-right of box. Place it on top-right of box.',
        #     'The toy cyan truck belongs to car toys which locate at bottom-left of box. Place it on bottom-left of box.',
        #     'The hex key belongs to hardware tools which locate at bottom-right of box. Place it on bottom-right of box.',
        #     'The gloves belong to protective equipments which locate at top-left of box. Place it on top-left of box.',
        #     'Wait for next task.'],
    }
    for trsr, sub_r in task_require_substep_reasonings.items():
        file_path = os.path.join(DATA_DIR, trsr, f"{trsr}_succ_t0001_s-0-0")
        file_list = os.listdir(file_path)
        file_list = [f for f in file_list if f.endswith('.hdf5')]
        file_list = sorted(file_list, key=lambda x: int(x.split('.')[0].split('_')[-1]))
        r_l = raw_lang[trsr]
        for file_name in tqdm(file_list[:]):
            # print(file_name)
            # if "hdf5" not in file_name:
            #     continue
            try:
                deal_subtask_label(path=file_path, sub_reasonings=sub_r,file_name=file_name, raw_lang=r_l)
            except Exception as e:
                print(e)
                print(trsr, file_name)
            # deal_pkl_label(path=file_path,file_name=file_name,save_path=save_path)


