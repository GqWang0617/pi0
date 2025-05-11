#!/bin/bash
LLM=qwen2_vl   #qwen2_vl  paligemma
LLM_MODEL_SIZE=2B #3B
# LLM_MODEL_SIZE=2_8B
# lora only vit and tune adapter
ACTION_HEAD=dit_diffusion_policy  #act #unet_diffusion_policy dit_diffusion_policy

#echo '5h'
#sleep 5h
ROOT=/home/wgq
DIT_ROOT=/home/share # /home/share || /gpfs/share/share

PRETRAIN=${ROOT}/wgq/model_param/multi_head2/${ACTION_HEAD}_results/checkpoint_all/${LLM}_${LLM_MODEL_SIZE}_pure/vanilla_aloha_${LLM}_vla_pt_f_vit/qwen2_vl_all_data_1200_align_frozen_dit_lora_substep_chunk_50/checkpoint-40000 # with substeps DIT
DIT_PRETRAIN=/home/share/model_param/scaledp/zzy/resnet50_with_film_subreason/use_constant_0_mobile_franka_reasoning_w_vl_data_selected_DiT-L_320_240_128_1e-4_numsteps_25000_sub_0_use_task_norm_0_2025_01_19_17_14_06/policy_step_25000_2025-01-20_04-35-32.ckpt

if [ "${LLM}" == "paligemma" ]; then
  echo "Using PaliGemma"
  mnop=${ROOT}/model_param/PaliGemma/paligemma/pixel_224/vla-paligemma-3b-pt-224
else
  mnop=${ROOT}/model_param/Qwen2-VL-${LLM_MODEL_SIZE}-Instruct
fi

OUTPUT=${ROOT}/model_param/multi_head2/${ACTION_HEAD}_results/checkpoint_all/${LLM}_${LLM_MODEL_SIZE}/vanilla_aloha_${LLM}_vla_pt_f_vit/${LLM}_3_cameras_franka_2000_trajs_stage_1
if [ -d "$OUTPUT" ]; then
   echo 'output exists'
else
   echo '!!output not exists!!'
   mkdir -p $OUTPUT
fi

mkdir -p $OUTPUT/src
cp -r ./aloha_scripts $OUTPUT/src/
cp -r ./scripts $OUTPUT/
cp -r ./data_utils $OUTPUT/src/
cp -r ./qwen2_vla $OUTPUT/src/
cp -r ./policy_heads $OUTPUT/src/

# tinyvla set "use_reasoning with_llm_head load_pretrain using_film" false
# paligemma flash_attn False

deepspeed --master_port 29604 --num_gpus=8 --num_nodes=1 ./train_vla.py \
  --deepspeed scripts/zero2.json \
  --use_reasoning True \
  --lora_enable False \
  --action_dim 10 \
  --state_dim 7 \
  --flash_attn True \
  --Using_EMA_Pretrain_DiT True \
  --chunk_size 16 \
  --lora_module "vit llm" \
  --load_pretrain False \
  --history_images_length 1 \
  --model_pretrain $PRETRAIN \
  --load_pretrain_dit True \
  --pretrain_dit_path $DIT_PRETRAIN \
  --ground_truth_reasoning False \
  --using_all_reasoning_hidden False \
  --using_film True \
  --using_ema False \
  --policy_head_type $ACTION_HEAD \
  --policy_head_size "DiT_L" \
  --with_llm_head True \
  --image_size_stable "(320,240)" \
  --image_size_wrist "(320,240)" \
  --lora_r 64 \
  --lora_alpha 256 \
  --episode_first False \
  --task_name "mobile_franka_reasoning_w_vl_data_selected" \
  --model_name_or_path $mnop \
  --version v0 \
  --tune_mm_mlp_adapter True \
  --freeze_vision_tower False \
  --freeze_backbone False \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --image_aspect_ratio pad \
  --group_by_modality_length False \
  --bf16 True \
  --output_dir $OUTPUT \
  --max_steps 60000 \
  --per_device_train_batch_size 12 \
  --gradient_accumulation_steps 1 \
  --save_strategy "steps" \
  --save_steps 10000 \
  --save_total_limit 50 \
  --learning_rate 2e-5 \
  --weight_decay 0. \
  --warmup_ratio 0.01 \
  --lr_scheduler_type "constant" \
  --logging_steps 50 \
  --tf32 True \
  --model_max_length 2048 \
  --gradient_checkpointing True \
  --dataloader_num_workers 8 \
  --lazy_preprocess True \
  --policy_class $ACTION_HEAD \
  --concat "token_cat" \
  --report_to tensorboard \
  --logging_dir $OUTPUT/log | tee $OUTPUT/log.log


for dir in "$OUTPUT"/*/ ; do
    # 检查文件夹名称是否包含'checkpoint'
    if [[ "$(basename "$dir")" == *"checkpoint"* ]]; then
        cp ${mnop}/preprocessor_config.json $dir
        cp ${mnop}/chat_template.json $dir
        # cp $OUTPUT/non_lora_trainables.bin $dir
    fi
done
echo $OUTPUT
