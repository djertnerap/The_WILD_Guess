import argparse
import os
from posixpath import split
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import parse_bool

plt.rcParams["figure.figsize"] = (8,6)

######################################################
## Script to create figures out the training logs 
## and extract Key Performance Indicators to benchmark
## Distribution Shift improvement methods.
######################################################

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', default='./logs', 
                        help='Optional argument to specify the path where logs can be found.')
    parser.add_argument('--show', type=parse_bool, const=True, nargs='?', 
                        help='If true, show all figures sequentially.')
    args = parser.parse_args()

    metadata_df = pd.DataFrame()
    algo_df = pd.DataFrame()
    eval_df = pd.DataFrame()
    split_list = []

    file_names = os.listdir(args.log_dir)
    for file in file_names:             #create metadata, algo and eval concatenated dataframes from the files in the log directory
        if file.startswith('split'):
            metadata_df = pd.concat([metadata_df,pd.read_csv(args.log_dir + "/" + file,encoding="utf-8", header=0)],ignore_index=True)
        elif file.endswith('algo.csv'):
            split = re.sub("_algo|.csv",'',file)
            algo_df = pd.concat([algo_df,pd.read_csv(args.log_dir + "/" + file,encoding="utf-8", header=0).assign(split=split)],ignore_index=True)
        elif file.endswith('eval.csv'):
            split = re.sub("_eval|.csv",'',file)
            split_list.append(split)
            eval_df = pd.concat([eval_df,pd.read_csv(args.log_dir + "/" + file,encoding="utf-8", header=0).assign(split=split)],ignore_index=True)
    
    if 'id_test' in split_list:         #replace test & val split datasets (used in training algo) by their real name (i.e. [id/ood]_[test/val])
        test_replace = 'ood_test'
    else:
        test_replace = 'id_test'
    
    if 'id_val' in split_list:
        val_replace = 'ood_val'
    else:
        val_replace = 'id_val'
    
    algo_df['split'] = algo_df['split'].replace(['test', 'val'],[test_replace, val_replace])
    eval_df['split'] = eval_df['split'].replace(['test', 'val'],[test_replace, val_replace])

    if len(metadata_df) != 0:       # create meatadata graphs only if the files are present
        metadata_df.year = metadata_df.year + 2002 
        metadata_df = metadata_df[metadata_df['region'] != 'Other']       
        sns.histplot(x="year", hue="usage", data=metadata_df, bins=16, element="step")
        #plt.legend(["OOD Val", "Train", "OOD Test", "ID Val", "ID Test"], loc='upper right')
        plt.title('Dataset split distribution according to years')
        plt.savefig(args.log_dir + "/dist_split_ds.png")
        if args.show: plt.show() 

        metadata_df.groupby(['region','usage'],as_index=False).size() \
        .pivot_table(index='usage',columns='region', values = 'size',aggfunc=np.sum).fillna(0) \
        .div(metadata_df.groupby(['usage']).size(), axis=0).mul(100) \
        .plot(kind = 'barh', stacked = True)
        plt.title('Dataset split distribution (%) according to regions')
        plt.legend(loc="lower center", bbox_to_anchor =(0.5,-0.16), ncol=5)
        plt.xlabel("Split distribution (%)")
        plt.ylabel("Split Dataset")
        plt.savefig(args.log_dir + "/dist_region_ds.png")
        if args.show: plt.show()

        sns.catplot(x="category", col="region", data=metadata_df, kind="count", height=4, aspect=1).set(xticklabels=[])
        plt.subplots_adjust(top=0.8)
        plt.suptitle('Categories distribution per region')
        plt.savefig(args.log_dir + "/dist_cat_region.png")
        if args.show: plt.show()  
    
    #create algo training graphs
    sns.lineplot(x="epoch", y='loss_avg', hue="split", data=algo_df)
    plt.title('Model Loss with regards to training epoch per dataset split')
    plt.savefig(args.log_dir + "/global_loss_ds.png")
    if args.show: plt.show()  

    sns.lineplot(x="epoch", y='acc_avg', hue="split", data=algo_df)
    plt.title('Model Accuracy with regards to training epoch per dataset split')
    plt.savefig(args.log_dir + "/global_acc_ds.png",)
    if args.show: plt.show() 

    #Sample distribution per group
    group_idx = [re.sub('loss_group:','',idx) for idx in algo_df.keys() if 'loss_group' in idx]
    temp_df =  pd.melt(algo_df,["batch","split"],['count_group:' + idx for idx in group_idx])
    temp_df = temp_df[temp_df['split']=='train']
    sns.lineplot(x="batch", y='value', hue="variable", data=temp_df)
    plt.title('Sample count per training group vs batch number')
    plt.savefig(args.log_dir + "/group_train_count.png")
    if args.show: plt.show()

    #Validation Loss per group
    temp_df =  pd.melt(algo_df,["epoch","split"],['loss_group:' + idx for idx in group_idx])
    temp_df = temp_df[temp_df['split']=='ood_val']
    sns.lineplot(x="epoch", y='value', hue="variable", data=temp_df)
    plt.title('Validation Loss per training group vs training epoch ')
    plt.savefig(args.log_dir + "/group_val_loss.png")
    if args.show: plt.show()

    #Test accuracy per group
    temp_df =  pd.melt(algo_df,["epoch","split"],['acc_group:' + idx for idx in group_idx])
    temp_df = temp_df[temp_df['split']=='ood_test']
    sns.lineplot(x="epoch", y='value', hue="variable", data=temp_df)
    plt.title('Test Accuracy per training group vs training epoch ')
    plt.savefig(args.log_dir + "/group_test_acc.png")
    if args.show: plt.show()

    #Key Performance Indicators extraction 

    best_epoch = algo_df['epoch'].loc[algo_df[algo_df['split']==val_replace].loss_avg.idxmin()]     #find best epoch at val loss minima

    for split_name in eval_df.split.unique():      #find best epoch accuracies for each dataset evaluation
        globals()['avg_' + split_name + '_acc'] = eval_df[(eval_df['split']==split_name)&(eval_df['epoch']==best_epoch)].acc_avg.values[0]
        globals()['avg_rg_' + split_name + '_acc'] = eval_df[(eval_df['split']==split_name)&(eval_df['epoch']==best_epoch)] \
                                                    [['acc_region:Asia','acc_region:Europe','acc_region:Africa','acc_region:Americas','acc_region:Oceania']].values.mean()
        globals()['wrst_rg_' + split_name + '_acc'] = eval_df[(eval_df['split']==split_name)&(eval_df['epoch']==best_epoch)].acc_worst_region.values[0]
        
    kpi_extract = str(best_epoch) + '\t' + \
    str(avg_train_acc) + '\t' + \
    str(avg_id_val_acc) + '\t' + \
    str(avg_ood_val_acc) + '\t' + \
    str(avg_id_test_acc) + '\t' + \
    str(avg_ood_test_acc) + '\t' + \
    str(avg_rg_train_acc) + '\t' + \
    str(avg_rg_id_val_acc) + '\t' + \
    str(avg_rg_ood_val_acc) + '\t' + \
    str(avg_rg_id_test_acc) + '\t' + \
    str(avg_rg_ood_test_acc) + '\t' + \
    str(wrst_rg_train_acc) + '\t' + \
    str(wrst_rg_id_val_acc) + '\t' + \
    str(wrst_rg_ood_val_acc) + '\t' + \
    str(wrst_rg_id_test_acc) + '\t' + \
    str(wrst_rg_ood_test_acc)

    f = open(args.log_dir + "/kpi_extract.txt","w+")
    f.write(kpi_extract)
    f.close()

if __name__=='__main__':
    main()   