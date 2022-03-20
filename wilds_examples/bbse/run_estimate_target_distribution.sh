#!/bin/bash

BASE_DIR="/home/wcallag_gmail_com/Development/exp_logs"
EXP_NAME="eval_erm_frac_1"
EXP_DIR="${BASE_DIR}/${EXP_NAME}"

OUTPUT_NAME="train_weighted_erm_frac_1"
OUTPUT_PATH="${BASE_DIR}/${OUTPUT_NAME}/class_weights.pth"


python estimate_target_distribution.py \
--yval "${EXP_DIR}/fmow_split_id_val_seed_0_TRUE_epoch_30_pred.csv" \
--ytest "${EXP_DIR}/fmow_split_test_seed_0_TRUE_epoch_30_pred.csv" \
--ypred_source "${EXP_DIR}/fmow_split_id_val_seed_0_epoch_30_pred.csv" \
--ypred_target "${EXP_DIR}/fmow_split_test_seed_0_epoch_30_pred.csv" \
--output "${OUTPUT_PATH}"