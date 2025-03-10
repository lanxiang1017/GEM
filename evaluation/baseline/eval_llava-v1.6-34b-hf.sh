#!/bin/bash
set -x

cd /path/to/evaluation/baseline
export PYTHONPATH=$(pwd)

SAVE_DIR=/path/to/eval_outputs/llava-v1.6-34b-hf

python infer/infer.py --config config/config.yaml --split ecgqa-test --mode none --model_name llava-v1.6-34b-hf --output_dir $SAVE_DIR --batch_size 1
sleep 5
python infer/infer.py --config config/config.yaml --split code15-test --mode none --model_name llava-v1.6-34b-hf --output_dir $SAVE_DIR --batch_size 1
sleep 5
python infer/infer.py --config config/config.yaml --split ptb-test-report --mode none --model_name llava-v1.6-34b-hf --output_dir $SAVE_DIR --batch_size 1
sleep 5
python infer/infer.py --config config/config.yaml --split ptb-valid --mode none --model_name llava-v1.6-34b-hf --output_dir $SAVE_DIR --batch_size 1
sleep 5
python infer/infer.py --config config/config.yaml --split ptb-test --mode none --model_name llava-v1.6-34b-hf --output_dir $SAVE_DIR --batch_size 1
sleep 5
python infer/infer.py --config config/config.yaml --split cpsc-test --mode none --model_name llava-v1.6-34b-hf --output_dir $SAVE_DIR --batch_size 1
sleep 5
python infer/infer.py --config config/config.yaml --split g12-test-no-cot --mode none --model_name llava-v1.6-34b-hf --output_dir $SAVE_DIR --batch_size 1
sleep 5
python infer/infer.py --config config/config.yaml --split csn-test-no-cot --mode none --model_name llava-v1.6-34b-hf --output_dir $SAVE_DIR --batch_size 1
sleep 5
python infer/infer.py --config config/config.yaml --split mmmu-ecg --mode none --model_name llava-v1.6-34b-hf --output_dir $SAVE_DIR --batch_size 1
sleep 5
python infer/infer.py --config config/config.yaml --split ecgqa-test --mode none --model_name llava-v1.6-34b-hf --output_dir $SAVE_DIR --batch_size 1
sleep 5
python infer/infer.py --config config/config.yaml --split arena --mode none --model_name llava-v1.6-34b-hf --output_dir $SAVE_DIR --batch_size 1
sleep 5
