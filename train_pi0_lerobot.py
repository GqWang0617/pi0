import gc
import pickle

import os
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"

os.environ['DEVICE'] = "cuda"
os.environ["WANDB_DISABLED"] = "true"


from data_utils.data_collator import PIOCollator
from typing import Dict, Optional, Sequence, List
from policy_heads import *
from transformers import AutoTokenizer
from lerobot.common.policies.factory import make_policy, make_policy_config

from aloha_scripts.constants import TASK_CONFIGS
from qwen2_vla.utils.robot_data_processor import Qwen2VLAProcess
from transformers import AutoConfig, AutoProcessor
import transformers
# from qwen2_vla import QWen2VLATrainer
from pi0.pi0_lerobot_trainer import PI0LeRobotTrainer
from data_utils.lerobot_dataset import load_data, set_seed
from qwen2_vla import model_load_utils as ml_utils
import torch
from dataclasses import dataclass, field, asdict

local_rank = None
from aloha_scripts.utils import *
#  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>parameters<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
@dataclass
class ActionHeadArguments:
    policy_head_type: str = field(default="dit_diffusion_policy") # unet_diffusion_policy
    policy_head_size: str = field(default="DiT_B") # DiT_L, DiT_XL, DiT_B, DiT_S
    state_dim: int = 7
    action_dim: int = 10


@dataclass
class ModelArguments:
    diffusion_strategy: str = field(default="flow") # diffusion or flow
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    pretrain_vlm_path: Optional[str] = field(default="/data/rl/HDD/data/weights/paligemma_3b/paligemma/pixel_224/paligemma-3b-pt-224")

    version: Optional[str] = field(default="v0")
    model_pretrain: Optional[str] = field(default="")  # pretrained model weights path
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    from_scratch: bool = field(default=False)

    concat: str = field(default="None")
    policy_class: str = field(default="droid_diffusion")

    # with_external_vit: bool = field(default=False)
    with_llm_head: bool = field(default=False)
    with_text_fcs: bool = field(default=False)
    only_using_input_embeddings: bool = field(default=False)  # using only input embeddings
    using_film: bool = field(default=False) # fusion modules
    using_xattn: bool = field(default=False) # fusion modules

    using_channel_cat: bool = field(default=False)
    using_all_reasoning_hidden: bool = field(default=False)
    ground_truth_reasoning: bool = field(default=False)

    Using_EMA_Pretrain_DiT: bool = field(default=False)

    load_pretrain_dit: bool = field(default=False) # loading pretrained dit weights
    pretrain_dit_path: Optional[str] = field(default=None) # path to pretrained dit weights

    freeze_policy_head: bool = field(default=False)
    is_tinyvla: bool = field(default=False)
    # vla_model_type: Optional[str] = field(default='qwen2_vla')

@dataclass
class DataArguments:

    lazy_preprocess: bool = False
    episode_first: bool = True  # batchsampler will samples episode index first and then samples timesteps
    select_seg_token_mask: bool = False
    use_reasoning: bool = False
    is_multimodal: bool = False
    image_aspect_ratio: str = 'square'
    task_name: str = field(default="stack_cube_2024_6_2")
    skip_mirrored_data: bool = field(default=False)
    chunk_size: int = field(default=16)
    delta_control: bool = field(default=False)
    image_size_stable: str = "480"  # default 270 x 480 and pretrain may be 180 x 320
    image_size_wrist: str = "56" # specify the image size of wrist camera
    history_images_length: int = 1
    home_lerobot: str = '/data/rl/HDD/data/data/aloha_data/lerobot'

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    using_ema: bool = field(default=False) # whether to use ema update whole module

    local_debug: bool = field(default=False)

    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.98)
    adam_epsilon: float = field(default=1e-7)
    remove_unused_columns: bool = field(default=False)

    flash_attn: bool = field(default=False)

    freeze_vision_tower: bool = field(default=False)
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    resume_from_checkpoint: bool = field(default=False)
    llm_loss_weight: float = field(default=1.0)

    seed: int = field(default=0)

    # logger
    logging_dir: str = field(default='./logs')  # TensorBoard日志的保存目录
    logging_strategy: str = field(default='steps')  # 设置为`steps`表示每几步记录一次日志
    logging_steps: int = field(default=10)

    save_steps: int = field(default=10)  # 每隔多少步保存一次模型
    num_train_epochs: int = field(default=3)
    max_steps: int = field(default=5000)

    # validate
    do_eval: bool = field(default=False)
    evaluation_strategy: str = field(default="no")
    eval_steps: int = field(default=200)
    per_device_eval_batch_size: int = field(default=32)

    load_pretrain: bool = False

    dataloader_pin_memory: bool = False
    # lora
    lora_enable: bool = True
    lora_module: str = "vit"
    lora_task_type: str = 'CAUSAL_LM'
    lora_r: int = 64
    lora_alpha: int = 256
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    non_lora_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)

    model_max_length: int = field(
        default=2048,
        metadata={
            "help":
                "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )


#  <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<parameters>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def parse_param():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, ActionHeadArguments))
    model_args, data_args, training_args, action_head_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    # todo
    # config.vision_config['image_size_wrist'] = model_args.image_size_wrist

    # config.concat = model_args.concat
    if model_args.is_tinyvla:
        rank0_print(f"{RED} This is TinyVLA, Please Check Both Using_film and Using_xattn equals False:Using_film {model_args.using_film}|Using_xattn {model_args.using_xattn} {RESET}")
        time.sleep(1)
    return model_args, data_args, training_args, action_head_args, None
def train_bc(train_dataset=None, val_dataset=None, model=None, config=None, sampler_params=None, tokenizer=None, processor=None):

    set_seed(config['training_args'].seed)
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if config['training_args'].bf16 else torch.float32))
    if config['data_args'].history_images_length > 2:
        rank0_print(f"{RED} Using History and Turn to Video mode.{RESET}")
        video = True
    else:
        video = False

    # data_collator = Qwen2VLADataCollatorForSupervisedDataset(multimodal_processor=processor, computed_type=compute_dtype, tokenizer=tokenizer, video=video)
    import time
    # print("data loader test............")
    # from torch.utils.data import DataLoader
    # data_loader = DataLoader(train_dataset, batch_size=config['training_args'].per_device_train_batch_size, shuffle=True)
    # for batch in data_loader:
    #     # batch = batch.to('cuda')
    #     # batch = {k:v.to('cuda') for k,v in batch.items()}
    #     for k,v in batch.items():
    #         print(k, v.dtype)
    #     # model(**batch)
    #     # time.sleep(1)
    #     del batch
    #     gc.collect()
    # exit(0)
    model.config.use_cache = True
    model.config.save_pretrained(config['training_args'].output_dir)

    data_module = dict(train_dataset=train_dataset,
                       data_collator=PIOCollator,
                       # eval_dataset=val_dataset
                       )
    trainer = PI0LeRobotTrainer(model=model,
                                 tokenizer=tokenizer,
                                 args=config['training_args'],
                                 sampler_params=sampler_params,
                                 **data_module)

    trainer.train(resume_from_checkpoint=config['training_args'].resume_from_checkpoint)

    trainer.save_state()

    model.config.use_cache = True


    if config['training_args'].local_rank == 0 or config['training_args'].local_rank == -1:
        model.config.save_pretrained(config['training_args'].output_dir)
        # model.save_pretrained(config['training_args'].output_dir, state_dict=state_dict)




def main(all_config=None, model_config=None):
    set_seed(1)
    # command line parameters
    # get task parameters
    task_config = TASK_CONFIGS[all_config['data_args'].task_name]
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']


    all_config['camera_names'] = camera_names
    all_config['episode_len'] = episode_len

    pi0_config = make_policy_config('pi0')
    setattr(pi0_config, "diffusion_strategy", all_config['model_args'].diffusion_strategy)

    rank0_print(f"You are using {RED} {all_config['model_args'].diffusion_strategy} {RESET}")

    if all_config['training_args'].load_pretrain:
        pi0_config.pretrained_path=all_config['model_args'].model_name_or_path
    else:
        pi0_config.pretrained_path=None
    # pi0_config.image_features = None

    train_dataset, val_dataset, stats = load_data(camera_names,
                                                  all_config['data_args'].chunk_size,
                                                    config=all_config,
                                                    rank0_print=rank0_print,
                                                    policy_class=all_config['action_head_args'].policy_head_type)

    model = make_policy(
        cfg=pi0_config,
        device='cuda',
        ds_meta=train_dataset.datasets[0]._dataset.meta,
    )
    model.language_tokenizer = AutoTokenizer.from_pretrained(all_config['model_args'].pretrain_vlm_path)

    model.requires_grad_(True)

    model.model.action_out_proj = model.model.action_out_proj.to(torch.float32)
    model.model.action_in_proj = model.model.action_in_proj.to(torch.float32)
    model.model.state_proj = model.model.state_proj.to(torch.float32)

    for k,v in model.named_parameters():
        if v.requires_grad:
            rank0_print(f'{k}: {v.requires_grad}')

    rank0_print(f"{RED} Training PI0_lerobot {RESET}")

    stats_path = os.path.join(all_config['training_args'].output_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    best_ckpt_info = train_bc(train_dataset=train_dataset, model=model, val_dataset=val_dataset, config=all_config)
    # save dataset stats
    stats_path = os.path.join(all_config['training_args'].output_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)


if __name__ == '__main__':
    model_args, data_args, training_args, action_head_args, model_config = parse_param()
    config = {
        'model_args':model_args,
        'data_args':data_args,
        'training_args':training_args,
        'action_head_args':action_head_args,
    }

    config_dict = {k:asdict(v) if not isinstance(v, dict) else v for k,v in config.items()}

    ckpt = os.path.join(config['training_args'].output_dir, f"checkpoint-{config['training_args'].save_steps}")
    if os.path.exists(ckpt):
        config['training_args'].resume_from_checkpoint = True
        rank0_print(f"{RED}Resuming Training............{RESET}")
    main(all_config=config, model_config=model_config)
    pass


