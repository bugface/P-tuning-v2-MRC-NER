#!/usr/bin/env bash
# -*- coding: utf-8 -*-


FILE=conll03_cased_large
REPO_PATH=/home/alexgre/projects/2023_projects/mrc_ner_ptuning/mrc4ner
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"

ROOT=/home/alexgre/projects/2023_projects/mrc_ner_ptuning/mrc4ner

DATA_DIR=${ROOT}/datasets/conll03/

BERT_DIR=/home/alexgre/projects/transformer_pretrained_models/bert-large-uncased
OUTPUT_BASE=${ROOT}/expr_bert_conll03

BATCH=4
GRAD_ACC=1
BERT_DROPOUT=0.1
MRC_DROPOUT=0.3
LR=2e-5
LR_MINI=5e-6
LR_SCHEDULER=linear
SPAN_WEIGHT=0.1
WARMUP=0
# prefix_len + MAX_LEN = 512
prefix_len=32
MAX_LEN=480
MAX_NORM=1.0
MAX_EPOCH=5
INTER_HIDDEN=2048
WEIGHT_DECAY=0.01
OPTIM=adamw #adamw
VAL_CHECK=0.2
PREC=16
SPAN_CAND=pred_and_gold


OUTPUT_DIR=${OUTPUT_BASE}/conll03
mkdir -p ${OUTPUT_DIR}


# python ${REPO_PATH}/train/mrc_ner_trainer.py \
# --model_type bert \
# --data_dir ${DATA_DIR} \
# --bert_config_dir ${BERT_DIR} \
# --max_length ${MAX_LEN} \
# --batch_size ${BATCH} \
# --gpus="1" \
# --precision=${PREC} \
# --progress_bar_refresh_rate 1 \
# --lr ${LR} \
# --val_check_interval ${VAL_CHECK} \
# --accumulate_grad_batches ${GRAD_ACC} \
# --default_root_dir ${OUTPUT_DIR} \
# --mrc_dropout ${MRC_DROPOUT} \
# --bert_dropout ${BERT_DROPOUT} \
# --max_epochs ${MAX_EPOCH} \
# --span_loss_candidates ${SPAN_CAND} \
# --weight_span ${SPAN_WEIGHT} \
# --warmup_steps ${WARMUP} \
# --distributed_backend=ddp \
# --max_length ${MAX_LEN} \
# --gradient_clip_val ${MAX_NORM} \
# --weight_decay ${WEIGHT_DECAY} \
# --optimizer ${OPTIM} \
# --lr_scheduler ${LR_SCHEDULER} \
# --classifier_intermediate_hidden_size ${INTER_HIDDEN} \
# --flat \
# --freeze 0 \
# --lr_mini ${LR_MINI}


# ptuning
python ${REPO_PATH}/train/mrc_ner_trainer.py \
--model_type pbert \
--data_dir ${DATA_DIR} \
--bert_config_dir ${BERT_DIR} \
--max_length ${MAX_LEN} \
--batch_size ${BATCH} \
--gpus="1" \
--precision=${PREC} \
--progress_bar_refresh_rate 1 \
--lr ${LR} \
--val_check_interval ${VAL_CHECK} \
--accumulate_grad_batches ${GRAD_ACC} \
--default_root_dir ${OUTPUT_DIR} \
--mrc_dropout ${MRC_DROPOUT} \
--bert_dropout ${BERT_DROPOUT} \
--max_epochs ${MAX_EPOCH} \
--span_loss_candidates ${SPAN_CAND} \
--weight_span ${SPAN_WEIGHT} \
--warmup_steps ${WARMUP} \
--distributed_backend=ddp \
--workers=4 \
--max_length ${MAX_LEN} \
--gradient_clip_val ${MAX_NORM} \
--weight_decay ${WEIGHT_DECAY} \
--optimizer ${OPTIM} \
--lr_scheduler ${LR_SCHEDULER} \
--classifier_intermediate_hidden_size ${INTER_HIDDEN} \
--flat \
--freeze 0 \
--prefix_len ${prefix_len} \
--total_category 4 \
--lr_mini ${LR_MINI}

