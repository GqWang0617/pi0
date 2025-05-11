from visualize_recover_data import load_hdf5
import os
import numpy as np
import sys
from tqdm import tqdm
import h5py
# from utils import *

sys.path.append('/home/jovyan/tzb/wjj/projects/dvla_mh_qwen2_vla')
from aloha_scripts.constants import TASK_CONFIGS
from aloha_scripts.utils import *

# template_substeps=['Grasp the left hem and left sleeve, then fold them forward.',
#        'Pull back to flatten the fabric.',
#        'Grasp the right hem and right sleeve, then fold them backward.',
#        'Fold to the left.',
#        'Wrap to the right.',
#        'Move the folded cloth to right.']
tasks = [
    # "folding_second_tshirt_yichen_0108",
    # "folding_second_tshirt_wjj_0108",
    # "folding_random_yichen_0109",
    # "folding_random_tshirt_yichen_0104", # unused
    # "folding_random_table_right_wjj_0109", # check

    # "folding_basket_two_tshirt_yichen_0109", # check
    # "folding_basket_second_tshirt_yichen_0110", # check
    # "folding_basket_second_tshirt_yichen_0109", # check
    # "folding_basket_second_tshirt_wjj_0110", # check

    # 'folding_basket_second_tshirt_yichen_0111', # check
    # 'folding_basket_second_tshirt_wjj_0113', # check
    # 'folding_basket_second_tshirt_wjj_0111', # check
]
DATA_DIR = '/home/jovyan/tzb/wjj/data/aloha_bimanual/aloha_4views/7z_1_10_extract/'

tasks = [os.path.join(DATA_DIR, t) for t in tasks]

# raw_lang_replace = "The crumpled shirts are in the basket. Pick it out and fold it."
# raw_lang_replace = "Fold the crumpled shirt on the middle of table."
raw_lang_replace = "The crumpled shirt is on the left side. Pull it over and fold it."

steps_need_replace={
    'Grasp the left hem then find the left sleeve. Then fold them forward.': 'Find left hem and left sleeve. Then lift up and fold them forward.',
}

if __name__ == '__main__':
    # tasks = TASK_CONFIGS['4_cameras_only_folding_shirt']['dataset_dir']
    # print(tasks)
    for t_p in tqdm(tasks[:], desc='modify substeps'):

        file_list = os.listdir(t_p)
        # print(file_list)
        # exit(0)
        file_list = [f for f in file_list if f.endswith('.hdf5')]
        file_list = sorted(file_list, key=lambda x: int(x.split('.')[0].split('_')[-1]))
        sub_bar = tqdm(total=len(file_list))
        for episode in file_list:
            dataset_path = os.path.join(t_p, episode)
            if not os.path.isfile(dataset_path):
                print(f'Dataset does not exist at \n{dataset_path}\n')
                exit()

            new_substeps = []

            with h5py.File(dataset_path, 'r') as root:
                substeps = root["substep_reasonings"][:]

            for substep in substeps:
                substep = substep.decode('utf-8')
                if substep not in steps_need_replace.keys():
                    new_substeps.append(substep)
                else:
                    new_substeps.append(steps_need_replace[substep])
            with h5py.File(dataset_path, 'a') as root2:
                root2["substep_reasonings"][()] = new_substeps
                root2['language_raw'][()] = [raw_lang_replace]

            sub_bar.set_description(f"{RED}{t_p}/{episode}{RESET}")
            sub_bar.update(1)
            # exit(0)
