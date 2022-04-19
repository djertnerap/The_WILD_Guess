#!/bin/bash

BASE_DIR="/home/wcallag_gmail_com"
BASE_NAME="erm_frac_1"
ALG="_doro"
PROJECT="doro"

TRAIN_NAME="train_${BASE_NAME}${ALG}"
EVAL_NAME="eval_${BASE_NAME}${ALG}"

LOG_DIR="${BASE_DIR}/Development/exp_logs"
DATA_DIR="${BASE_DIR}/Data"

TRAIN=true

if [ ${TRAIN} == true ];
then
    if [ -d "${LOG_DIR}/${TRAIN_NAME}" ];
    then
        echo "Running Training..."
        python run_expt.py \
        -d fmow \
        --algorithm ERM \
        --root_dir /home/wcallag_gmail_com/Data \
        --loader_kwargs "num_workers=8" \
        --loader_kwargs pin_memory=True \
        --frac 1 \
        --use_wandb=true \
        --log_dir="${LOG_DIR}/${TRAIN_NAME}" \
        --n_epochs=50 \
        --erm_weights="${LOG_DIR}/${TRAIN_NAME}/class_weights.pth" \
        --wandb_kwargs project=${PROJECT} entity=the-wild-guess
    fi
else
    if [ -d "${LOG_DIR}/${EVAL_NAME}" ];
        then
        echo "Running Eval..."
        python run_expt.py \
        -d fmow \
        --algorithm ERM \
        --root_dir /home/wcallag_gmail_com/Data \
        --loader_kwargs "num_workers=8" \
        --loader_kwargs pin_memory=True \
        --frac 1 \
        --use_wandb=true \
        --log_dir="${LOG_DIR}/${EVAL_NAME}" \
        --wandb_kwargs project=${PROJECT} entity=the-wild-guess \
        --eval_only
    fi
fi