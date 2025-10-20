#!/usr/bin/env bash
set -e
set -o xtrace

BASE_MODEL='meta-llama/Meta-Llama-3-8B'
MODEL='LLaMA3-8B'
ADAPTER='Stella'
LORA_R=${1:-32}
LORA_ALPHA=${2:-64}
LR=${3:-0.0005}
CUTOFF=${4:-256}
WEIGHT_DECAY=${5:-0}
RETRACTION=${6:-"polar"}
OPTIMIZER=${7:-"adamw"}
LR_SCHEDULE=${8:-"linear"}
INIT=${9:-"rando"}
POSTFIX=${10:-""}
WORK_DIR=${11:-"work_dirs/${MODEL,,}_commonsense_${ADAPTER,,}_r${LORA_R}_a${LORA_ALPHA}_lr${LR}_${RETRACTION}_${OPTIMIZER}_${LR_SCHEDULE}_${INIT}${POSTFIX}"}

export TOKENIZERS_PARALLELISM=true

mkdir -p $WORK_DIR

# debugpy --listen 5678 --wait-for-client \
python \
tools/finetune.py \
    --base_model "$BASE_MODEL" \
    --data_path 'dataset/commonsense_170k.json' \
    --output_dir $WORK_DIR \
    --batch_size 16 --micro_batch_size 8 --num_epochs 3 \
    --learning_rate $LR --cutoff_len $CUTOFF --val_set_size 120 \
    --weight_decay $WEIGHT_DECAY \
    --eval_step 200 --save_step 200 --adapter_name $ADAPTER \
    --target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]' \
    --lora_r $LORA_R --lora_alpha $LORA_ALPHA \
    --lr_scheduler_type $LR_SCHEDULE --stella_init $INIT \
    --bf16 True --fp16 False \
    --stella_retraction $RETRACTION --optimizer $OPTIMIZER \
    2>&1 | tee -a $WORK_DIR/finetune.txt


python tools/evaluate_commonsense.py \
    --model "$MODEL" \
    --adapter $ADAPTER \
    --dataset boolq \
    --base_model "$BASE_MODEL" \
    --batch_size 1 \
    --lora_weights $WORK_DIR \
    --output_dir $WORK_DIR \
    2>&1 | tee -a $WORK_DIR/boolq.txt

python tools/evaluate_commonsense.py \
    --model "$MODEL" \
    --adapter $ADAPTER \
    --dataset piqa \
    --base_model "$BASE_MODEL" \
    --batch_size 1 \
    --lora_weights $WORK_DIR \
    --output_dir $WORK_DIR \
    2>&1 | tee -a $WORK_DIR/piqa.txt

python tools/evaluate_commonsense.py \
    --model "$MODEL" \
    --adapter $ADAPTER \
    --dataset social_i_qa \
    --base_model "$BASE_MODEL" \
    --batch_size 1 \
    --lora_weights $WORK_DIR \
    --output_dir $WORK_DIR \
    2>&1 | tee -a $WORK_DIR/social_i_qa.txt

python tools/evaluate_commonsense.py \
    --model "$MODEL" \
    --adapter $ADAPTER \
    --dataset hellaswag \
    --base_model "$BASE_MODEL" \
    --batch_size 1 \
    --lora_weights $WORK_DIR \
    --output_dir $WORK_DIR \
    2>&1 | tee -a $WORK_DIR/hellaswag.txt

python tools/evaluate_commonsense.py \
    --model "$MODEL" \
    --adapter $ADAPTER \
    --dataset winogrande \
    --base_model "$BASE_MODEL" \
    --batch_size 1 \
    --lora_weights $WORK_DIR \
    --output_dir $WORK_DIR \
    2>&1 | tee -a $WORK_DIR/winogrande.txt

python tools/evaluate_commonsense.py \
    --model "$MODEL" \
    --adapter $ADAPTER \
    --dataset ARC-Challenge \
    --base_model "$BASE_MODEL" \
    --batch_size 1 \
    --lora_weights $WORK_DIR \
    --output_dir $WORK_DIR \
    2>&1 | tee -a $WORK_DIR/arc-challenge.txt

python tools/evaluate_commonsense.py \
    --model "$MODEL" \
    --adapter $ADAPTER \
    --dataset ARC-Easy \
    --base_model "$BASE_MODEL" \
    --batch_size 1 \
    --lora_weights $WORK_DIR \
    --output_dir $WORK_DIR \
    2>&1 | tee -a $WORK_DIR/arc-easy.txt

python tools/evaluate_commonsense.py \
    --model "$MODEL" \
    --adapter $ADAPTER \
    --dataset openbookqa \
    --base_model "$BASE_MODEL" \
    --batch_size 1 \
    --lora_weights $WORK_DIR \
    --output_dir $WORK_DIR \
    2>&1 | tee -a $WORK_DIR/openbookqa.txt
