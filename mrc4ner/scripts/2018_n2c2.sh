#!/usr/bin/env bash
# -*- coding: utf-8 -*-

REPO_PATH=/home/alexgre/projects/2023_projects/mrc_ner_ptuning/mrc4ner
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"
# ROOT=/home/alexgre/projects/2023_projects/mrc_ner_ptuning/mrc4ner

DATA_DIR=/data/datasets/cheng/mrc-for-ner-medical/2018_n2c2/data/mrc_entity

# {'ADE','Dosage','Drug','Duration','Form','Frequency','Reason','Route','Strength'}
CATEGORY_NUM=9

## bert
# MODEL_TYPE=pbert
# BERT_DIR=/home/alexgre/projects/transformer_pretrained_models/mimiciii-bert-large-uncased_5e_128b
# BERT_DIR=/home/alexgre/projects/transformer_pretrained_models/mimiciii_bert-base-uncased_10e_128b

## megatron
MODEL_TYPE=pmegatron
BERT_DIR=/home/alexgre/projects/transformer_pretrained_models/gatortron-syn-345m_deid_vocab

##roberta


ROOT=/home/alexgre/projects/2023_projects/mrc_ner_ptuning/expr
FILE=expr_2018n2c2_megatron_syn_unfreeze_pl16
# FILE=expr_2018n2c2_bert_mimic_base_unfreeze
# DATA_DIR=${ROOT}/datasets/conll03/
OUTPUT_BASE=${ROOT}/$FILE


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
prefix_len=16
MAX_LEN=496
MAX_NORM=1.0
MAX_EPOCH=30
INTER_HIDDEN=2048
WEIGHT_DECAY=0.01
OPTIM=adamw #adamw
VAL_CHECK=0.2
PREC=16
SPAN_CAND=pred_and_gold

OUTPUT_DIR=${OUTPUT_BASE}
mkdir -p ${OUTPUT_DIR}

# export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export CUDA_VISIBLE_DEVICES=0,1,2,3,4

# ptuning
python ${REPO_PATH}/train/mrc_ner_trainer.py \
--model_type ${MODEL_TYPE} \
--data_dir ${DATA_DIR} \
--bert_config_dir ${BERT_DIR} \
--max_length ${MAX_LEN} \
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
--max_length ${MAX_LEN} \
--gradient_clip_val ${MAX_NORM} \
--weight_decay ${WEIGHT_DECAY} \
--optimizer ${OPTIM} \
--lr_scheduler ${LR_SCHEDULER} \
--classifier_intermediate_hidden_size ${INTER_HIDDEN} \
--workers=4 \
--flat \
--freeze 0 \
--prefix_len ${prefix_len} \
--total_category $CATEGORY_NUM \
--batch_size ${BATCH} \
--gpus=1 \
--lr_mini ${LR_MINI}

#--auto_select_gpus True \
#--workers=4 \