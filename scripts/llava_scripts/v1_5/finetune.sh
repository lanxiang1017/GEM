#!/bin/bash
export HF_HOME="/home/jupyter/.cache/huggingface"

# llm versions: Qwen/Qwen2.5-7B-Instruct, lmsys/vicuna-13b-v1.5, meta-llama/Llama-3.1-8B-Instruct
LLM_VERSION="Qwen/Qwen2-7B-Instruct" 
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
PROMPT_VERSION="qwen_2"

# data args
data_path="/home/jupyter/data/ecg_jsons/ECGInstruct_ts.json"
image_folder="/home/jupyter/data/ecg_images"
ecg_folder="/home/jupyter/data/ecg_timeseries"
ecg_tower="/home/jupyter/LLaVA/ecg_coca/open_clip/checkpoint/cpt_wfep_epoch_20.pt"

BASE_RUN_NAME="llava-${LLM_VERSION_CLEAN}-full-finetune"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

deepspeed --master_port 29504 llava/train/train_mem.py \
    --deepspeed ./scripts/zero2_pulse.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path ${data_path} \
    --ecg_folder ${ecg_folder} \
    --ecg_tower ${ecg_tower} \
    --open_clip_config coca_ViT-B-32 \
    --image_folder ${image_folder} \
    --ecg_projector_type mlp2x_gelu \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio ori \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir ./checkpoints/${BASE_RUN_NAME} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 0.05 \
    --save_total_limit 20 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 20 \
    --lazy_preprocess True \
    --report_to wandb
