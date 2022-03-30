#!/bin/bash

BASE_DIR="/home/wcallag_gmail_com"
FRAC="1"
BASE_NAME="erm_frac_${FRAC}"
ALG="_doro_30_epoch"
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
        --algorithm "doro" \
        --doro_alg "cvar_doro" \
        --alpha 0.25 \
        --eps 0.01 \
        --root_dir /home/wcallag_gmail_com/Data \
        --loader_kwargs "num_workers=8" \
        --loader_kwargs pin_memory=True \
        --frac 1 \
        --use_wandb=true \
        --log_dir="${LOG_DIR}/${TRAIN_NAME}" \
        --n_epochs=30 \
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