import sys

sys.path.append('/home/jovyan/tzb/wjj/projects/dvla_mh_qwen2_vla')
from utils import *
from aloha_scripts.reasonings_constants import TASK_REASONINGS, TASK_INSTRUCTIONS


def main_add_language_to_direct_path(data_path, tasks_reasoning, task_question): # add language to a direct data path
    episodes = os.listdir(data_path)
    for ep in tqdm(episodes):
        ep_p = os.path.join(data_path, ep)
        print(ep_p)
        if os.path.isdir(ep_p):
            continue
        # raw_lang = reason
        add_reasoning_language_to_hdf5(ep_p, tasks_reasoning)
        print("reasoning..................................")
        show_hdf5(ep_p, attr='reasoning')

        q = task_question
        add_language_to_hdf5(ep_p, q)
        show_hdf5(ep_p)
        print("################New Task#####################")

def main_add_reasoning_language_by_constants(name='11_8_aloha_bimanual_bin_picking_paper_cup'): # add language to the tasks in aloha_scripts.constants.TASK_CONFIGS

    target_tasks = TASK_CONFIGS[name]['dataset_dir']

    for t, reason in TASK_REASONINGS.items():
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>Processing:{t} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        print("=" * 40, t, "=" * 40)
        for each in target_tasks:
            if t in each:
                t_p = each
                break
        # print(t, reason, t_p)
        # continue
        episodes = os.listdir(t_p)
        for ep in tqdm(episodes):
            ep_p = os.path.join(t_p, ep)
            print(ep_p)
            if not ep_p.endswith(".hdf5"):
                continue
            # raw_lang = reason
            try:
                add_reasoning_language_to_hdf5(ep_p, reason)
            except Exception as e:
                print(e)
                print(ep_p)
                exit(0)
            print("reasoning..................................")
            show_hdf5(ep_p, attr='reasoning')
            if t in TASK_INSTRUCTIONS.keys():
                q = TASK_INSTRUCTIONS[t]
                add_language_to_hdf5(ep_p, q)
                show_hdf5(ep_p)
                print("################New Task#####################")

def main_add_language_to_hdf5_droid38K(root='/gpfs/private/tzb/wjj/data/droid_with_distilbert_lang_three_view'): # instruction language and reasonings are stored in pickle
    droid_path = f'{root}/droid_with_distilbert_lang_three_view_succ_t0001_s-0-0'
    qa_path = f'{root}/results_all/'
    import pickle
    pkls = os.listdir(qa_path)

    for pkl in tqdm(pkls):
        if not pkl.endswith('.pkl'):
            continue
        with open(qa_path + pkl, 'rb') as f:
            try:
                qa = pickle.load(f)
            except Exception as e:
                print(e, pkl)

            reason = qa['answer']
            question = qa['question']
        # continue
        ep_p = os.path.join(droid_path, pkl.replace('_result.pkl', '.hdf5'))
            # raw_lang = reason
        add_reasoning_language_to_hdf5(ep_p, reason)
        print("reasoning..................................")
        show_hdf5(ep_p, attr='reasoning')
        add_language_to_hdf5(ep_p, question)
        show_hdf5(ep_p)
        print("################New Task#####################")


def main_add_distilbert_embedding_to_hdf5(task_name = "11_8_aloha_bimanual_bin_picking_paper_cup"):
    from transformers import AutoTokenizer, AutoModel
    import torch

    tasks = TASK_CONFIGS[task_name]['dataset_dir']

    tokenizer = AutoTokenizer.from_pretrained('/gpfs/private/tzb/ljm/model_param/distilbert-base-uncased')
    model = AutoModel.from_pretrained("/gpfs/private/tzb/ljm/model_param/distilbert-base-uncased",
                                      torch_dtype=torch.float16)
    model.to('cuda')

    for t in tqdm(tasks):
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>processing {t}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        eps = os.listdir(os.path.join(t))
        eps = [e for e in eps if e.endswith('.hdf5')]

        encoded_lang = None
        for ep_path in tqdm(eps):
            ep_path = os.path.join(t, ep_path)
            if encoded_lang is None:
                with h5py.File(ep_path, 'r') as hdf5_file:
                    raw_lang = hdf5_file['language_raw'][0].decode('utf-8')
                encoded_input = tokenizer(raw_lang, return_tensors='pt').to('cuda')
                outputs = model(**encoded_input)
                encoded_lang = outputs.last_hidden_state.sum(1).squeeze().unsqueeze(0)
                print(f'encoded_lang size: {encoded_lang.size()}')
            add_distilbert_embedding_to_hdf5(ep_path, encoded_lang)

# main_add_reasoning_language_by_constants(name="4_cameras_folding_shirt")
# data_path = '/home/jovyan/tzb/wjj/data/aloha_bimanual/aloha_4views'
data_path = '/home/jovyan/tzb/wjj/data/aloha_bimanual/aloha_4views/1_24_7z_extract'
tasks = [
    # 'clean_table_lxy_1222_pick_place_water_left_arm', #In: Organize the bottles on the table. Re: Pick up the bottles and place into the tray.
    # 'clean_table_ljm_1222_pick_place_water_coca-cola_Genki_forest_jasmine_tea' #In: Organize the bottles on the table. Re: Pick up the bottles and place into the tray.
    # 'pour_coffee_zhaopeiting_1224', #I am thirsty, give me a cup of coffee.  Ok, pour coffee into the cup.
    # 'get_papercup_yichen_1223' # Give me a paper cup. # Pick up the paper cup and place it on table.
    # "pick_up_coke_in_refrigerator_yichen_1223", #Pick up the coke in refrigerator. Pick up the coke in refrigerator.
    # 'storage_bottle_green_tea_oolong_mineral_water_ljm_weiqing_1225_right_hand', #In: Organize the bottles on the table. Re: Pick up the bottles and place into the tray.
    # 'storage_bottle_green_tea_oolong_mineral_water_lxy_weiqing_1225', #In: Organize the bottles on the table. Re: Pick up the bottles and place into the tray.
    # 'pour_rice_yichen_0102', # Pour the water into the pot.
    # 'pick_paper_ball_from_bike' # Pick up the paper ball and place to rubbish bin.
    # 'unloading_dryer_wjj_0119', #"Unload the dryer."  "Pick out the cloth and place to laundry basket."
    # 'unloading_dryer_yichen_0119',
    # 'unloading_dryer_yichen_0120'
    "truncate_push_basket_to_left_1_24"
]
task_question = "Push the basket to left."
tasks_reasoning = "Push the basket to left."
for t in tasks:
    episode_path = os.path.join(data_path, t)
    episodes = os.listdir(episode_path)

    main_add_language_to_direct_path(episode_path, tasks_reasoning, task_question)