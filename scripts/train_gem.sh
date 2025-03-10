#!/bin/bash

# distributed training configurations
GPUS_PER_NODE=8
NNODES=1
NODE_RANK=0
MASTER_ADDR="127.0.0.1"
MASTER_PORT="1234"
WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))

# your huggingface configurations
export HF_HOME=""

LLM_VERSION=""
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
DATA_VERSION=""
BASE_RUN_NAME="GEM-${LLM_VERSION_CLEAN}-${DATA_VERSION}-finetune"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

version=llava_v1

data_path=""
image_folder=""
ecg_folder=""
ecg_tower=""

num_epochs=1
GRAD_ACC_STEP=2
BATCH_PER_GPU=16
TOTAL_BATCH_SIZE=$(($WORLD_SIZE * $BATCH_PER_GPU))

torchrun \
    --nproc_per_node $GPUS_PER_NODE \
    --master_addr $MASTER_ADDR \
    --node_rank $NODE_RANK \
    --master_port $MASTER_PORT \
    --nnodes $NNODES \
    ...your_path_to/train_mem.py \ 
    --deepspeed ...your_path_to/zero2.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${version} \
    --data_path ${data_path} \
    --ecg_folder ${ecg_folder} \
    --ecg_tower ${ecg_tower} \
    --open_clip_config coca_ViT-B-32 \
    --image_folder $image_folder \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio ori \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir ./checkpoints/${BASE_RUN_NAME} \
    --num_train_epochs ${num_epochs} \
    --per_device_train_batch_size $BATCH_PER_GPU \
    --per_device_eval_batch_size $BATCH_PER_GPU \
    --gradient_accumulation_steps $GRAD_ACC_STEP \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 0.2 \
    --save_total_limit 5 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 64 \
    --lazy_preprocess True \
    --report_to wandb