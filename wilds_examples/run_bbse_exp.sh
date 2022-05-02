#!/bin/bash

BASE_DIR="/home/wcallag_gmail_com"
FRAC=1 # Fraction of data to use (between 0 and 1)
EPOCHS=30 # Number of epochs to run

# Used to create a directory for logging experiments
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
    echo "Checking if ${LOG_DIR}/${TRAIN_NAME} exists..."
    if [ -d "${LOG_DIR}/${TRAIN_NAME}" ];
    then
        echo "Running Training..."
        python run_will_expt.py \
        -d fmow \
        --algorithm ERM \
        --root_dir "${BASE_DIR}/Data" \ 
        --loader_kwargs "num_workers=8" \
        --loader_kwargs pin_memory=True \
        --frac=${FRAC} \
        --use_wandb=true \
        --log_dir="${LOG_DIR}/${TRAIN_NAME}" \
        --n_epochs=50 \
        --erm_weights="${LOG_DIR}/${TRAIN_NAME}/class_weights.pth" \
        --wandb_kwargs project=${PROJECT} entity=the-wild-guess
    else
        echo "${LOG_DIR}/${TRAIN_NAME} does not exist. Create this directory before running experiment"
    fi
else
    echo "Checking if ${LOG_DIR}/${EVAL_NAME} exists..."
    if [ -d "${LOG_DIR}/${EVAL_NAME}" ];
    then
        echo "Running Eval..."
        python run_will_expt.py \
        -d fmow \
        --algorithm ERM \
        --root_dir "${BASE_DIR}/Data" \
        --loader_kwargs "num_workers=8" \
        --loader_kwargs pin_memory=True \
        --frac=${FRAC} \
        --use_wandb=true \
        --log_dir="${LOG_DIR}/${EVAL_NAME}" \
        --wandb_kwargs project=${PROJECT} entity=the-wild-guess \
        --eval_only
    else
        echo "${LOG_DIR}/${EVAL_NAME} does not exist. Create this directory before running experiment."
    fi
fi