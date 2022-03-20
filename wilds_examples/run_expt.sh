#!/bin/bash

BASE_NAME="erm_frac_1"
TRAIN_NAME="train_${BASE_NAME}"
EVAL_NAME="eval_${BASE_NAME}"
BASE_DIR="/home/wcallag_gmail_com/Development/exp_logs"
TRAIN=false

if [ ${TRAIN} == true ];
then
    if [ -d "${BASE_DIR}/${TRAIN_NAME}" ] && ! [ "$(ls -A ${BASE_DIR}/${TRAIN_NAME})" ];
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
        --log_dir="${BASE_DIR}/${TRAIN_NAME}" \
        --n_epochs=50 \
        --wandb_kwargs project=bbse entity=the-wild-guess
    fi
else
    if [ -d "${BASE_DIR}/${EVAL_NAME}" ];
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
        --log_dir="${BASE_DIR}/${EVAL_NAME}" \
        --wandb_kwargs project=bbse entity=the-wild-guess \
        --eval_only
    fi
fi