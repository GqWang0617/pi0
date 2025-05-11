import os
import h5py
from PIL import Image, ImageDraw, ImageFont
import sys
from tqdm import tqdm
sys.path.append('/gpfs/private/tzb/wjj/projects/dvla_mh_qwen2_vla')
from aloha_scripts.constants import DATA_DIR, TASK_CONFIGS
def parse_data(ep_p):
    with h5py.File(ep_p, 'r') as f:
        for key in f.keys():
            if key == 'substep_reasonings':
                print(f[key][()])
            if not isinstance(f[key], h5py.Group):
                print(key, f[key].shape)
            else:
                print(key, f[key]['qpos'].shape)
                for k in f[key]['images'].keys():
                    print(k, f[key]['images'][k].shape)
                    Image.fromarray(f[key]['images'][k][0]).show()

def visualize_episodes(ep_p, step=1):
    with h5py.File(ep_p, 'r') as f:
        try:
            targets = f['substep_reasonings'][:]
        except Exception as e:
            targets = [f['reasoning'][0]] * 1000
        # print(targets)
        imgs = []
        lang = f['language_raw'][0].decode('utf-8')
        font = ImageFont.truetype('/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf', size=12)
        # os.makedirs(os.path.join(save_path, file_name.split('.')[0]), exist_ok=True)
        fills = [(255, 0, 0), (0, 0, 255)]
        save_path = os.path.join(os.path.dirname(ep_p), 'example')
        os.makedirs(save_path, exist_ok=True)
        for i, img in enumerate(f['observations']['images']['cam_high'][0::step]):
            # image_pil = Image.open(os.path.join(path, f"{file_name.split('.')[0]}.jpg"))
            image_pil = Image.fromarray(img)
            draw = ImageDraw.Draw(image_pil)
            # print(targets[i])
            draw.text((120, 160), targets[i], font=font, fill=fills[0] if 'left' in targets[i].decode('utf-8') else fills[1])
            draw.text((20, 20), lang, font=font,
                      fill=(255,255,255))
            imgs.append(image_pil)

            image_pil.save(os.path.join(save_path, f"{i}.jpg"))

for task_p in tqdm(TASK_CONFIGS['11_8_aloha_bimanual_bin_picking_paper_cup']['dataset_dir']):
    episodes = os.listdir(task_p)
    episodes = [episode for episode in episodes if episode.endswith('.hdf5')]
    ep_p = os.path.join(task_p, episodes[0])
    visualize_episodes(ep_p)