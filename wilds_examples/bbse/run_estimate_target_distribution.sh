#!/bin/bash

BASE_DIR="/home/wcallag_gmail_com/Development/exp_logs/eval_erm_frac_1"


python estimate_target_distribution.py \
--yval "${BASE_DIR}/fmow_split_id_val_seed_0_TRUE_epoch_30_pred.csv" \
--ytest "${BASE_DIR}/fmow_split_test_seed_0_TRUE_epoch_30_pred.csv" \
--ypred_source "${BASE_DIR}/fmow_split_id_val_seed_0_epoch_30_pred.csv" \
--ypred_target "${BASE_DIR}/fmow_split_test_seed_0_epoch_30_pred.csv" \
--output "/tmp/training_weights.pth"