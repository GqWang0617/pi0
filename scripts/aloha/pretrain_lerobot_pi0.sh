#!/bin/bash

mnop=/home/wgq/model_param/pi0_lerobot/
vlm_pretrain=/home/wgq/model_param/PaliGemma/paligemma/pixel_224/vla-paligemma-3b-pt-224/

TASKNAME=aloha_all #aloha_all

OUTPUT=/home/wgq/train_results/pi0_lerobot_results/pretrain_${TASKNAME}_all_data_diffusion_cosine

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
cp -r ./lerobot $OUTPUT/src/
cp -r ./pi0 $OUTPUT/src/

deepspeed --master_port 29604 --num_gpus=8 --num_nodes=1 ./train_pi0_lerobot.py \
  --deepspeed scripts/zero2.json \
  --action_dim 14 \
  --diffusion_strategy "diffusion" \
  --use_reasoning False \
  --state_dim 14 \
  --pretrain_vlm_path $vlm_pretrain \
  --home_lerobot "/home/wgq/lerobot_data/aloha" \
  --flash_attn False \
  --chunk_size 50 \
  --task_name $TASKNAME \
  --model_name_or_path $mnop \
  --bf16 True \
  --output_dir $OUTPUT \
  --max_steps 80000 \
  --image_size_stable "(320,240)" \
  --image_size_wrist "(320,240)" \
  --per_device_train_batch_size 12 \
  --gradient_accumulation_steps 1 \
  --save_strategy "steps" \
  --save_steps 20000 \
  --save_total_limit 50 \
  --learning_rate 2e-5 \
  --weight_decay 0. \
  --warmup_ratio 0.01 \
  --lr_scheduler_type "cosine" \
  --logging_steps 50 \
  --tf32 True \
  --model_max_length 2048 \
  --gradient_checkpointing True \
  --dataloader_num_workers 8 \
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

mv ./60030.log $OUTPUT
echo $OUTPUT
