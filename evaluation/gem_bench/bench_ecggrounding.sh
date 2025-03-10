#!/bin/bash

while getopts m:h option
do
 case "${option}"
 in
  m) model_name=${OPTARG};;        # Model name
  h) echo "Usage: $0 -m model_name"
     exit 0;;
 esac
done

if [[ -z "$model_name" ]]; then
    echo "Error: Missing required parameters. Use -h for help."
    exit 1
fi

# test data chunks
splits=("chunk_0" "chunk_1" "chunk_2" "chunk_3" "chunk_4" "chunk_5" "chunk_6" "chunk_7")

SAVE_DIR=../../eval_outputs
CKPT_DIR=

model_path=${CKPT_DIR}/${model_name}

# one chunk per GPU
for i in "${!splits[@]}"; do
    split=${splits[$i]}
    save_dir=${SAVE_DIR}/${model_name}/ecg-grounding-test

    if [ ! -d "$save_dir" ]; then
        mkdir -p "$save_dir"
    fi

    gpu_id=$((i % 8))

    echo "Running on GPU $gpu_id with question file: $split"

    CUDA_VISIBLE_DEVICES=$gpu_id python ../../llava/eval/model_ecg_resume.py \
        --model-path "$model_path" \
        --image-folder "" \
        --question-file "...path/$split.json" \
        --answers-file "${save_dir}/${split}-step-final.jsonl" \
        --conv-mode "llava_v1" \
        --ecg-folder "" \
        --ecg_tower "" \
        --open_clip_config "coca_ViT-B-32" &  
done

wait
echo "All evaluations are completed."