#!/bin/bash

BASE_DIR="/home/wcallag_gmail_com"
FRAC=1
EPOCHS=30
ALPHA=0.25
EPS=0.005
PROJECT="doro"

BASE_NAME="erm_frac_${FRAC}"
ALG="_${PROJECT}_${EPOCHS}_epoch_alpha${ALPHA}_eps${EPS}"
SUFFIX=""

TRAIN_NAME="train_${BASE_NAME}${ALG}${SUFFIX}"
EVAL_NAME="eval_${BASE_NAME}${ALG}${SUFFIX}"

LOG_DIR="${BASE_DIR}/Development/exp_logs"
DATA_DIR="${BASE_DIR}/Data"

TRAIN=true

if [ ${TRAIN} == true ];
then
    echo "Checking if ${LOG_DIR}/${TRAIN_NAME} exists..."
    if [ -d "${LOG_DIR}/${TRAIN_NAME}" ];
    then
        echo "Running Training..."
        python run_will_expt.py \
        -d fmow \
        --algorithm=${PROJECT} \
        --doro_alg "cvar_doro" \
        --alpha=${ALPHA}\
        --eps=${EPS} \
        --root_dir /home/wcallag_gmail_com/Data \
        --loader_kwargs "num_workers=8" \
        --loader_kwargs pin_memory=True \
        --frac=${FRAC} \
        --use_wandb=true \
        --log_dir="${LOG_DIR}/${TRAIN_NAME}" \
        --n_epochs=${EPOCHS} \
        --wandb_kwargs project=${PROJECT} entity=the-wild-guess
    else
        echo "${LOG_DIR}/${TRAIN_NAME} does not exist. Create this directory before running experiment"
    fi
fi
