import os
import argparse
import sys

import torch
import numpy as np
import pandas as pd
from scipy import stats

import labelshift as ls

def read(path:str) -> np.array:
    df = pd.read_csv(path, header=None)
    result = np.array(df[0].tolist())
    return result


def main():
    # Argparser
    parser = argparse.ArgumentParser()

    parser.add_argument('--yval', required=True, type=str, help='In-Domain Validation Labels')
    parser.add_argument('--ytest', required=True, type=str, help='Out-of-Domain Test Labels')
    parser.add_argument('--ypred_source', required=True, type=str, help='In-Domain Validation Predictions')
    parser.add_argument('--ypred_target', required=True, type=str, help='Out-of-Domain Test Predictions')
    parser.add_argument('--output', required=True, type=str, help='Path to save output weights')

    config = parser.parse_args()

    # Load files
    yval = read(config.yval)
    ytest = read(config.ytest)
    ypred_s = read(config.ypred_source)
    ypred_t = read(config.ypred_target)

    num_classes = 62 # Need to verify + replace this magic number.
 
    # Estimate target distribution
    wt = ls.estimate_labelshift_ratio(yval, ypred_s, ypred_t, num_classes)

    # Output additional stats
    Py_true = ls.calculate_marginal(ytest,num_classes)
    Py_base = ls.calculate_marginal(yval,num_classes)
    wt_true = Py_true/Py_base
    print("||wt - wt_true||^2  = " + repr(np.sum((wt-wt_true)**2)/np.linalg.norm(wt_true)**2))
    print("KL(Py_est|| Py_true) = " + repr(stats.entropy(wt,Py_base)))

    # Save to output
    torch.save(wt, config.output)

if __name__ == "__main__":
    main()