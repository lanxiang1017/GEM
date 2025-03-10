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

SAVE_DIR=/path/to/eval_outputs
CKPT_DIR=/path/to/saved/ckpts

model_path=${CKPT_DIR}/${model_name}

# Set directories and files
save_dir=${SAVE_DIR}/${model_name}/${split}
if [ ! -d "$save_dir" ]; then
    mkdir -p "$save_dir"
fi

python ../llava/eval/model_ecg_arena.py \
    --model-path "$model_path" \
    --image-folder "../../data/ECGBench/images" \
    --question-file "../../data/ECGBench/${split}.json" \
    --answers-file "${save_dir}/step-final.jsonl" \
    --conv-mode "llava_v1"