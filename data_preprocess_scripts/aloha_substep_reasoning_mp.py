import copy
import os.path
import time
import h5py
import numpy as np
from tqdm import tqdm
import sys
# sys.path.append('/gpfs/private/tzb/wjj/projects/dvla_mh_qwen2_vla')
import os
from tqdm import tqdm
# from aloha_scripts.constants import TASK_CONFIGS
# from aloha_scripts.constants import DATA_DIR
import concurrent.futures
from PIL import Image, ImageDraw, ImageFont
import functools
import cv2
def nms_del_repeat(seq):
    if len(seq) == 2:
        return seq
    if len(seq) < 2:
        return 'le5'
    print('......nmsing.....', seq)
    xor_diff = seq[1:] - seq[:-1]
    xor_diff = np.where(xor_diff < 50)[0]
    for i in xor_diff:
        seq = np.delete(seq, i)
        if len(seq) == 2:
            return seq
    return seq

def generate_h5(data, path):
    with h5py.File(path, 'w') as f:
        def recurisve_save_to_h5(group, d):
            for k,v in d.items():
                if isinstance(v, dict):
                    sub_group = group.create_group(k)
                    recurisve_save_to_h5(sub_group, v)
                else:
                    group.create_dataset(k, data=v)

        recurisve_save_to_h5(f, data)

def recursive_h5_to_dict(group):

    def _recursive_h5_to_dict(d):
        data = {}
        for k, v in d.items():
            if isinstance(v, h5py.Group):
                data[k] = _recursive_h5_to_dict(v)
            else:
                data[k] = np.array(v)
        return data

    return _recursive_h5_to_dict(group)
def deal_subtask_label(path:str='/Volumes/PSSD-8/ljm/5_28_put_rubbish_can_succ_t0001_s-0-0/episode_12.hdf5',sub_reasonings=None, file_name=None,save_path=None, raw_lang=None, save_imgs=False, output=None)->None:
    s1 = time.time()
    paths = path.split('/')
    paths[-1] = f"ttlow_res_{paths[-1]}"
    save_path = '/'.join(paths)

    if os.path.exists(os.path.join(save_path, f"{file_name}")):
        return 0
    data=h5py.File(os.path.join(path,file_name),'r')
    griper_action_np=data["action"][:,6::7]  # 6 means left gripper, 13 means right gripper
    mean_gripper = np.mean(griper_action_np,axis=0) # calculate the mean value of gripper
    index_griper=griper_action_np > mean_gripper
    index_griper = index_griper.astype(int)
    xor_diff = np.abs(index_griper[:-1] - index_griper[1:])
    subtask_idx_left = np.where(xor_diff[:, 0] == 1)[0] + 1
    subtask_idx_right = np.where(xor_diff[:, 1] == 1)[0] + 1

    prepare = subtask_idx_left[0]
    subtask_idx_left = subtask_idx_left[2::2]
    subtask_idx_right = subtask_idx_right[2::2]
    # subtask_idx_left = nms_del_repeat(subtask_idx_left)
    # subtask_idx_right = nms_del_repeat(subtask_idx_right)

    subtask_idx = np.append(0, subtask_idx_left)
    subtask_idx = np.append(subtask_idx, subtask_idx_right)
    subtask_idx = sorted(subtask_idx)
    # return 0
    sub_id = 0
    targets = []

    for i in range(len(subtask_idx) - 1):
        start_idx = subtask_idx[i]
        end_idx = subtask_idx[i+1]
        s_r = sub_reasonings[sub_id]
        # for j in range(start_idx, end_idx):
        targets.extend([s_r] * (end_idx-start_idx))
        sub_id += 1
    for i in range(prepare):

        targets[i] = 'Preparing for the task.'
    targets.extend([sub_reasonings[-1]] * (griper_action_np.shape[0] - subtask_idx[-1]))
    if save_imgs:
        imgs = []
        font = ImageFont.truetype('/usr/share/fonts/truetype/abyssinica/AbyssinicaSIL-Regular.ttf', size=16)
        os.makedirs(os.path.join(save_path, file_name.split('.')[0]), exist_ok=True)
        fills = [(255,0,0), (0,0, 255)]

        for i,img in enumerate(data['observations']['images']['cam_high'][:]):
            # image_pil = Image.open(os.path.join(path, f"{file_name.split('.')[0]}.jpg"))
            image_pil = Image.fromarray(img)
            draw = ImageDraw.Draw(image_pil)
            draw.text((120, 160), targets[i], font=font, fill=fills[0] if 'left' in targets[i] else fills[1])
            imgs.append(image_pil)

            image_pil.save(os.path.join(save_path, file_name.split('.')[0], f"{i}.jpg"))

    resized_imgs = {k:[] for k in data['observations']['images'].keys()}
    s2 = time.time()

    def _resize_image(img, key):
        if 'wrist' in key:
            resized_image = cv2.resize(img, (56, 56))
        else:
            resized_image = cv2.resize(img, (320, 240))
        return np.array(resized_image)

    # 在循环外创建线程池
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        for k, v in resized_imgs.items():
            try:
                # 直接传递 key 到 _resize_image 函数
                results_list = list(executor.map(lambda img: _resize_image(img, k), data['observations']['images'][k]))
            except Exception as e:
                print(f"Error occurred: {e}")

            resized_imgs[k] = results_list

    s3 = time.time()

    os.makedirs('/'.join(paths), exist_ok=True)
    with h5py.File(os.path.join(save_path,f"{file_name}"), 'w') as dest_f:
        stack = [(data, dest_f)]

        while stack:
            cur_data, cur_f = stack.pop()
            for name, item in cur_data.items():
                if name in resized_imgs.keys():
                    cur_f.create_dataset(name, data=resized_imgs[name])
                    continue

                if isinstance(item, h5py.Group):
                    new_group = cur_f.create_group(name)
                    stack.append((item, new_group))
                else:
                    cur_f.create_dataset(name, data=item[()])

        dest_f.create_dataset("substep_reasonings", data=targets)
        dest_f.create_dataset('language_raw', data=[raw_lang])


    # generate_h5(dict_data, os.path.join(save_path,f"{file_name}"))
    data.close()
    s4 = time.time()

    print(f'{file_name} done. Time taken: Copy h5 to new: {s4 - s3:.2f} seconds, Resize imgs:{s3 - s2:.2f} seconds, Adding Substep reasoning: {s2 - s1:.2f} seconds')





if __name__=="__main__":
    raw_lang = {
        'plate_red_can_bowl_paper': 'Sorting the tablewares and rubbish on the table.',
        'bowl_red_can_brown_mug_purple_papercup': 'Sorting the tablewares and rubbish on the table.',
        'pot_blue_plastic_waste_shovel_green_paper_cup': 'Sorting the tablewares and rubbish on the table.',
        'bowl_fenda_plate_paper': 'Sorting the tablewares and rubbish on the table.',
        'plate_orange_can_bowl_paper': 'Sorting the tablewares and rubbish on the table.',
        'plate_blue_plastic_waste_spoon_paper_drink_box': 'Sorting the tablewares and rubbish on the table.',
        'plate_paper_drink_box_pot_bowl': 'Sorting the tablewares and rubbish on the table.',
        'mug_red_can_paper_cup_paper': 'Sorting the tablewares and rubbish on the table.',
    }

    task_require_substep_reasonings = {
        # 'plate_red_can_bowl_paper': [
        #     'pick plate with left hand and place it to left container.',
        #     'pick red can with right hand and place it to right rubbish bin.',
        #     'pick bowl with left hand and place it to left container.',
        #     'pick paper rubbish with right hand and place it to right rubbish bin.',
        #     'waiting for next task.'
        # ],
        # 'bowl_red_can_brown_mug_purple_papercup': [
        #     'pick bowl with left hand and place it to left container.',
        #     'pick red can with right hand and place it to right rubbish bin.',
        #     'pick brown mug with left hand and place it to left container.',
        #     'pick rubbish paper cup with right hand and place it to right rubbish bin.',
        #     'waiting for next task.'
        # ],
        # 'pot_blue_plastic_waste_shovel_green_paper_cup': [
        #     'pick pot with left hand and place it to left container.',
        #     'pick blue plastic waste with right hand and place it to right rubbish bin.',
        #     'pick shovel with left hand and place it to left container.',
        #     'pick rubbish paper cup with right hand and place it to right rubbish bin.',
        #     'waiting for next task.'
        # ],
        # 'bowl_fenda_plate_paper': [
        #     'pick bowl with left hand and place it to left container.',
        #     'pick orange can with right hand and place it to right rubbish bin.',
        #     'pick plate with left hand and place it to left container.',
        #     'pick paper rubbish with right hand and place it to right rubbish bin.',
        #     'waiting for next task.'
        # ],
        # 'plate_orange_can_bowl_paper': [
        #     'pick plate with left hand and place it to left container.',
        #     'pick orange can with right hand and place it to right rubbish bin.',
        #     'pick bowl with left hand and place it to left container.',
        #     'pick paper rubbish with right hand and place it to right rubbish bin.',
        #     'waiting for next task.'
        # ],
        # 'plate_blue_plastic_waste_spoon_paper_drink_box': [
        #     'pick plate with left hand and place it to left container.',
        #     'pick blue plastic waste with right hand and place it to right rubbish bin.',
        #     'pick spoon with left hand and place it to left container.',
        #     'pick paper drink box with right hand and place it to right rubbish bin.',
        #     'wating for next task.'
        # ],
        # 'plate_paper_drink_box_pot_bowl': [
        #     'pick plate with left hand and place it to left container.',
        #     'pick paper drink box with right hand and place it to right rubbish bin.',
        #     'pick pot with left hand and place it to left container.',
        #     'pick bowl with right hand and place it to left rubbish bin.',
        #     'wating for next task.'
        # ],
        'mug_red_can_paper_cup_paper': [
            'pick plate with left hand and place it to left container.',
            'pick paper drink box with right hand and place it to right rubbish bin.',
            'pick pot with left hand and place it to left container.',
            'pick bowl with right hand and place it to right rubbish bin.',
            'wating for next task.'
        ],
    }
    DATA_DIR = '/home/rl/PSSD-6/wjj/aloha'

    OUTPUT=os.path.join(DATA_DIR)
    for trsr, sub_r in task_require_substep_reasonings.items():
        file_path = os.path.join(DATA_DIR, trsr)
        file_list = os.listdir(file_path)
        file_list = [f for f in file_list if f.endswith('.hdf5')]
        file_list = sorted(file_list, key=lambda x: int(x.split('.')[0].split('_')[-1]))
        r_l = raw_lang[trsr]
        i = 0
        for file_name in tqdm(file_list[:]):
            # print(file_name)
            # if "hdf5" not in file_name:
            #     continue

            if i == 5:
                save_or_not = True
            else:
                save_or_not = False
            try:
                deal_subtask_label(path=file_path, sub_reasonings=sub_r,file_name=file_name, raw_lang=r_l, save_imgs=save_or_not, output=OUTPUT)
            except Exception as e:
                print(e)
                print(trsr, file_name)
            i += 1
            # deal_pkl_label(path=file_path,file_name=file_name,save_path=save_path)

