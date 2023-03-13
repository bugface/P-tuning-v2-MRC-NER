#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# file: nested_inference.sh
#

REPO_PATH=/home/alexgre/projects/2023_projects/mrc_ner_ptuning/mrc4ner
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"


DATA_DIR=/data/datasets/cheng/mrc-for-ner-medical/2018_n2c2/data/mrc_entity

ROOT=/home/alexgre/projects/2023_projects/mrc_ner_ptuning/expr

FILE=expr_2018n2c2_bert_mimic_base_unfreeze

BERT_DIR=/home/alexgre/projects/transformer_pretrained_models/mimiciii_bert-base-uncased_10e_128b

MAX_LEN=480

OUTPUT_BASE=${ROOT}/$FILE

DATA_SIGN=${OUTPUT_BASE}/label2idx.json

predict_output=${OUTPUT_BASE}/pred_${FILE}.json

MODEL_CKPT=${OUTPUT_BASE}/epoch=25.ckpt

HPARAMS_FILE=${OUTPUT_BASE}/lightning_logs/version_0/hparams.yaml

python3 ${REPO_PATH}/inference/mrc_ner_inference.py \
--data_dir ${DATA_DIR} \
--bert_dir ${BERT_DIR} \
--max_length ${MAX_LEN} \
--model_ckpt ${MODEL_CKPT} \
--hparams_file ${HPARAMS_FILE} \
--output_fn ${predict_output} \
--dataset_sign ${DATA_SIGN}