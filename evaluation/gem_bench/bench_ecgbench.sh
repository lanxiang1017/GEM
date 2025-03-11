#!/bin/bash

# Parse command-line arguments for split, model_name
while getopts m:d:h option
do
 case "${option}"
 in
  m) model_name=${OPTARG};;        # Model name
  d) split=${OPTARG};;             # Dataset split
  h) echo "Usage: $0 -s start_checkpoint -e end_checkpoint -i interval -m model_name -d split -f is_final"
     exit 0;;
 esac
done

# Ensure all required parameters are provided
if [[ -z "$model_name" || -z "$split" ]]; then
    echo "Error: Missing required parameters. Use -h for help."
    exit 1
fi

SAVE_DIR=../../eval_outputs
CKPT_DIR=

model_path=${CKPT_DIR}/${model_name}

# Set directories and files
save_dir=${SAVE_DIR}/${model_name}/${split}
if [ ! -d "$save_dir" ]; then
    mkdir -p "$save_dir"
fi

CUDA_VISIBLE_DEVICES=0 python ../../llava/eval/model_ecg_resume.py \
    --model-path "$model_path" \
    --image-folder "" \
    --question-file "...path_to/${split}.json" \
    --answers-file "${save_dir}/step-final.jsonl" \
    --conv-mode "llava_v1" \
    --ecg-folder "" \
    --ecg_tower "" \
    --open_clip_config "coca_ViT-B-32" 