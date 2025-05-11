import os

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import h5py
import argparse
import imageio
from tqdm import tqdm
import json

def load_hdf5(dataset_dir, dataset_name, cam_key=None):
    dataset_path = os.path.join(dataset_dir, dataset_name)
    if not os.path.isfile(dataset_path):
        print(f'Dataset does not exist at \n{dataset_path}\n')
        exit()

    with h5py.File(dataset_path, 'r') as root:
        is_sim = root.attrs['sim']
        qpos = root['/observations/qpos'][()]
        qvel = root['/observations/qvel'][()]
        effort = root['/observations/effort'][()]
        action = root['/action'][()]
        try:
            subtask = root['/subtask'][()]
        except Exception as e:
            print(f'Error loading subtask: {e} in {dataset_path}/{dataset_name}')
            subtask = None
        image_dict = dict()
        if cam_key is not None:
            image_dict[cam_key] = root[f'/observations/images/{cam_key}'][()]
        else:
            for cam_name in root[f'/observations/images/'].keys():
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][()]

    return qpos, qvel, effort, action, subtask, image_dict

def add_text_to_image(image, text, position=(20,50), font_size=30, frame_id=0, font_color=(255,0,0)):
    image = Image.fromarray(image.astype('uint8'))
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf", font_size)
    except Exception as e:
        font = ImageFont.load_default()

    draw.text(position, f"Subtask: {text}", font=font, fill=font_color)

    draw.text((position[0], position[1]+50), f"Frame id: {frame_id}", font=font, fill=(0,255,0))
    return image

def save_images_as_video(images, save_path, subtask_labels):
    diff = subtask_labels[1:] - subtask_labels[:-1]
    diff_idxs = np.argwhere(diff > 0)
    # frames = np.split(images, images.shape[0])
    frames = []
    substep = 0
    diff_idxs = [[-1, 0]] + diff_idxs.tolist() + [[len(subtask_labels)-1, 0]]
    for i in range(len(diff_idxs)-1):
        start = diff_idxs[i][0] + 1
        end = diff_idxs[i+1][0] + 1

        for j in range(start, end):
            img = add_text_to_image(images[j], subtask_labels[j], frame_id=j)
            frames.append(img)

        substep += 1
    imageio.mimsave(save_path, frames, fps=25)
    print(f'Saved images at: {save_path}')
    return diff_idxs

def visualize_episode(data_dir, task, save_dir, target_episode=-1, has_subtask=True):
    name = f'episode_{target_episode}.hdf5'
    data_path = os.path.join(data_dir, task)
    qpos, qvel, effort, action, subtask_labels, image_dict = load_hdf5(data_path, name, cam_key='cam_high')
    if not has_subtask:
        # subtask_labels = list(range(qpos.shape[0]))
        subtask_labels = np.zeros_like(qpos)
    if 10 in subtask_labels:
        index = 0
        for i in range(len(subtask_labels)):
            if subtask_labels[i] == 10:
                index += 1
            if subtask_labels[i] == 100:
                subtask_labels[i] = 100
            else:
                subtask_labels[i] = index
    cam_high_imgs = image_dict['cam_high']

    video_save_path = os.path.join(save_dir, f'episode_{target_episode}.mp4')
    diff_idxs = save_images_as_video(cam_high_imgs, save_path=video_save_path, subtask_labels=subtask_labels)

def main(data_dir, task, skip_label, save_dir):

    #save_dir = args['save_dir']
    data_path = os.path.join(data_dir, task)
    save_video_dir = os.path.join(save_dir, 'videos2', task)
    os.makedirs(save_video_dir, exist_ok=True)
    save_json_dir = os.path.join(save_dir, 'subtask_labels', task)
    os.makedirs(save_json_dir, exist_ok=True)
    episodes_path = os.listdir(data_path)
    episodes_path = [e for e in episodes_path if e.endswith('.hdf5')]
    episodes_path = sorted(episodes_path, key=lambda x: int(x.split('.')[0].split('_')[-1]))

    recover_data_record = []

    for episode in tqdm(episodes_path):
        if not episode.endswith('.hdf5'):
            continue

        qpos, qvel, effort, action, subtask_labels, image_dict = load_hdf5(data_path, episode, cam_key='cam_high')

        if max(subtask_labels) <= 5:
            print(f'No recover label for {episode}')
            continue
        recover_data_record.append(os.path.join(data_path, episode))

        cam_high_imgs = image_dict['cam_high']

        video_save_path = os.path.join(save_video_dir, f'{episode.split(".")[0]}.mp4')
        diff_idxs = save_images_as_video(cam_high_imgs, save_path=video_save_path, subtask_labels=subtask_labels)

        if skip_label:
            continue

        recover_id = input(f"Please input the top-left corner id when recover happens in {episode}: ")
        data = {'ep_path': os.path.join(data_path,episode), 'recover_id': recover_id, 'diff_idxs': diff_idxs}
        with open(os.path.join(save_json_dir, f'{episode.split(".")[0]}.json'), 'w') as f:
            json.dump(data, f, indent=4)

    with open(os.path.join(save_json_dir, 'recover_episodes.txt'), 'w') as f:
        for each in recover_data_record:
            f.write(each + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_dir', default="/home/jovyan/tzb/wjj/data/aloha_bimanual/aloha_4views/7z_1_15_16_data_extract/", type=str, help='Dataset dir.', required=False)
    parser.add_argument('--save_dir', default="/home/jovyan/tzb/wjj/data/aloha_bimanual/aloha_4views/1_14_data_move_add_folding_shirt/move_data/", type=str, help='Save dir.', required=False)
    parser.add_argument('--target_episode', default=10, type=int, help='', required=False)

    parser.add_argument('--skip_label', default=True, type=bool, help='True means only record the trajs path which needs recover and do not label immediately.', required=False)
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    tasks = [
        # 'fold_shirt_lxy1213',
        # 'fold_shirt_wjj1213_meeting_room',
        # 'fold_shirt_zzy1213',
        # 'fold_shirt_zmj1213',
        # 'fold_shirt_zmj1212',
        # 'folding_shirt', # local debug
        # "pour_coffee_zhaopeiting_1224",
        # "get_papercup_yichen_1223"
        # "pick_up_coke_in_refrigerator_yichen_1223"
        # 'get_papercup_and_pour_coke_yichen_1224',
        # 'fold_shirt_wjj1213_meeting_room',
        # 'pick_cup_and_pour_water_wjj_weiqing_coffee'
        # "fold_tshirts_zzy_1209",
        # "fold_tshirts_129",
        # "fold_t_shirt_easy_version",
        # "fold_shirt_zmj1212",
        # "fold_t_shirt_easy_version_office"
        # "folding_blue_tshirt_yichen_0102",
        # "folding_random_yichen_0109",
        # "folding_random_yichen_0109",
        #'folding_random_tshirt_yichen_0104',
        # "folding_basket_two_tshirt_yichen_0109",
        # "folding_basket_second_tshirt_yichen_0114",
        "weiqing_folding_basket_first_tshirt_pink_wjj_0115"
    ]
    has_subtask = True
    for task in tasks:
        save_video_dir = os.path.join(args.dataset_dir, task, 'video')
        os.makedirs(save_video_dir, exist_ok=True)
        if args.target_episode!=-1:
            visualize_episode(data_dir=args.dataset_dir, task=task, save_dir=save_video_dir, target_episode=args.target_episode)
        else:
            episodes = os.listdir(os.path.join(args.dataset_dir, task))
            episodes = [e for e in episodes if e.endswith('.hdf5')]
            episodes = sorted(episodes, key=lambda x: int(x.split('.')[0].split('_')[-1]))
            with open(os.path.join(save_video_dir, 'recover_episodes.txt'), 'w') as f:
                for episode in episodes:
                    episode = episode.strip()
                    f.write(episode + ' \n')

            for episode in tqdm(episodes):
                epi = int(episode.split('.')[0].split('_')[-1])
                visualize_episode(data_dir=args.dataset_dir, task=task, save_dir=save_video_dir,
                                  target_episode=epi, has_subtask=has_subtask)

        # main(data_dir=args.dataset_dir, task=task, save_dir=args.save_dir, skip_label=args.skip_label)

