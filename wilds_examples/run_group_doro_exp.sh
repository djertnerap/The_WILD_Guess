#!/bin/bash

BASE_DIR="/home/wcallag_gmail_com"
FRAC=1
EPOCHS=30
ALPHA=0.25
EPS=0.005

BASE_NAME="erm_frac_${FRAC}"
ALG="_groupDORO_${EPOCHS}_epoch_alpha${ALPHA}_eps${EPS}"
PROJECT="groupDORO"

TRAIN_NAME="train_${BASE_NAME}${ALG}"
EVAL_NAME="eval_${BASE_NAME}${ALG}"

LOG_DIR="${BASE_DIR}/Development/exp_logs"
DATA_DIR="${BASE_DIR}/Data"

TRAIN=true

if [ ${TRAIN} == true ];
then
    echo "Checking if ${LOG_DIR}/${TRAIN_NAME} exists..."
    if [ -d "${LOG_DIR}/${TRAIN_NAME}" ];
    then
        echo "Running Training..."
        python run_expt.py \
        -d fmow \
        --algorithm "groupDORO" \
        --doro_alg "cvar_doro" \
        --alpha=${ALPHA} \
        --eps=${EPS} \
        --root_dir /home/wcallag_gmail_com/Data \
        --loader_kwargs "num_workers=8" \
        --loader_kwargs pin_memory=True \
        --frac=${FRAC} \
        --use_wandb=true \
        --log_dir="${LOG_DIR}/${TRAIN_NAME}" \
        --n_epochs=${EPOCHS} \
        --groupby_fields="region" \
        --wandb_kwargs project=${PROJECT} entity=the-wild-guess
    fi
fi