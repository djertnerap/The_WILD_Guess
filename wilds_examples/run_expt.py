import os
import time
import re
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import sys
from collections import defaultdict

try:
    import wandb
except Exception as e:
    pass

import wilds
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from wilds.common.grouper import CombinatorialGrouper
from wilds.datasets.unlabeled.wilds_unlabeled_dataset import WILDSPseudolabeledSubset

from utils import set_seed, Logger, BatchLogger, log_config, ParseKwargs, load, initialize_wandb, log_group_data, parse_bool, get_model_prefix, move_to
from train import train, evaluate, infer_predictions
from algorithms.initializer import initialize_algorithm, infer_d_out
from transforms import initialize_transform
from models.initializer import initialize_model
from configs.utils import populate_defaults
import configs.supported as supported

import torch.multiprocessing
from torch.utils.data.sampler import WeightedRandomSampler, SubsetRandomSampler         #for bootstrap exploration

# Necessary for large images of GlobalWheat
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def main():
    ''' Arg defaults are filled in according to examples/configs/ '''
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('-d', '--dataset', choices=wilds.supported_datasets, required=True)
    parser.add_argument('--algorithm', required=True, choices=supported.algorithms)
    parser.add_argument('--root_dir', required=True,
                        help='The directory where [dataset]/data can be found (or should be downloaded to, if it does not exist).')

    # Dataset
    parser.add_argument('--split_scheme', help='Identifies how the train/val/test split is constructed. Choices are dataset-specific.')
    parser.add_argument('--dataset_kwargs', nargs='*', action=ParseKwargs, default={},
                        help='keyword arguments for dataset initialization passed as key1=value1 key2=value2')
    parser.add_argument('--download', default=False, type=parse_bool, const=True, nargs='?',
                        help='If true, tries to download the dataset if it does not exist in root_dir.')
    parser.add_argument('--frac', type=float, default=1.0,
                        help='Convenience parameter that scales all dataset splits down to the specified fraction, for development purposes. Note that this also scales the test set down, so the reported numbers are not comparable with the full test set.')
    parser.add_argument('--version', default=None, type=str, help='WILDS labeled dataset version number.')

    # Unlabeled Dataset
    parser.add_argument('--unlabeled_split', default=None, type=str, choices=wilds.unlabeled_splits,  help='Unlabeled split to use. Some datasets only have some splits available.')
    parser.add_argument('--unlabeled_version', default=None, type=str, help='WILDS unlabeled dataset version number.')
    parser.add_argument('--use_unlabeled_y', default=False, type=parse_bool, const=True, nargs='?', 
                        help='If true, unlabeled loaders will also the true labels for the unlabeled data. This is only available for some datasets. Used for "fully-labeled ERM experiments" in the paper. Correct functionality relies on CrossEntropyLoss using ignore_index=-100.')

    # Loaders
    parser.add_argument('--loader_kwargs', nargs='*', action=ParseKwargs, default={})
    parser.add_argument('--unlabeled_loader_kwargs', nargs='*', action=ParseKwargs, default={})
    parser.add_argument('--train_loader', choices=['standard', 'group'])
    parser.add_argument('--uniform_over_groups', type=parse_bool, const=True, nargs='?', help='If true, sample examples such that batches are uniform over groups.')
    parser.add_argument('--distinct_groups', type=parse_bool, const=True, nargs='?', help='If true, enforce groups sampled per batch are distinct.')
    parser.add_argument('--n_groups_per_batch', type=int)
    parser.add_argument('--unlabeled_n_groups_per_batch', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--unlabeled_batch_size', type=int)
    parser.add_argument('--eval_loader', choices=['standard'], default='standard')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Number of batches to process before stepping optimizer and schedulers. If > 1, we simulate having a larger effective batch size (though batchnorm behaves differently).')

    # Model
    parser.add_argument('--model', choices=supported.models)
    parser.add_argument('--model_kwargs', nargs='*', action=ParseKwargs, default={},
                        help='keyword arguments for model initialization passed as key1=value1 key2=value2')
    parser.add_argument('--noisystudent_add_dropout', type=parse_bool, const=True, nargs='?', help='If true, adds a dropout layer to the student model of NoisyStudent.')
    parser.add_argument('--noisystudent_dropout_rate', type=float)
    parser.add_argument('--pretrained_model_path', default=None, type=str, help='Specify a path to pretrained model weights')
    parser.add_argument('--load_featurizer_only', default=False, type=parse_bool, const=True, nargs='?', help='If true, only loads the featurizer weights and not the classifier weights.')

    # NoisyStudent-specific loading
    parser.add_argument('--teacher_model_path', type=str, help='Path to NoisyStudent teacher model weights. If this is defined, pseudolabels will first be computed for unlabeled data before anything else runs.')

    # Transforms
    parser.add_argument('--transform', choices=supported.transforms)
    parser.add_argument('--additional_train_transform', choices=supported.additional_transforms, help='Optional data augmentations to layer on top of the default transforms.')
    parser.add_argument('--target_resolution', nargs='+', type=int, help='The input resolution that images will be resized to before being passed into the model. For example, use --target_resolution 224 224 for a standard ResNet.')
    parser.add_argument('--resize_scale', type=float)
    parser.add_argument('--max_token_length', type=int)
    parser.add_argument('--randaugment_n', type=int, help='Number of RandAugment transformations to apply.')

    # Objective
    parser.add_argument('--loss_function', choices=supported.losses)
    parser.add_argument('--loss_kwargs', nargs='*', action=ParseKwargs, default={},
                        help='keyword arguments for loss initialization passed as key1=value1 key2=value2')

    # Algorithm
    parser.add_argument('--groupby_fields', nargs='+')
    parser.add_argument('--group_dro_step_size', type=float)
    parser.add_argument('--coral_penalty_weight', type=float)
    parser.add_argument('--dann_penalty_weight', type=float)
    parser.add_argument('--dann_classifier_lr', type=float)
    parser.add_argument('--dann_featurizer_lr', type=float)
    parser.add_argument('--dann_discriminator_lr', type=float)
    parser.add_argument('--afn_penalty_weight', type=float)
    parser.add_argument('--safn_delta_r', type=float)
    parser.add_argument('--hafn_r', type=float)
    parser.add_argument('--use_hafn', default=False, type=parse_bool, const=True, nargs='?')
    parser.add_argument('--irm_lambda', type=float)
    parser.add_argument('--irm_penalty_anneal_iters', type=int)
    parser.add_argument('--self_training_lambda', type=float)
    parser.add_argument('--self_training_threshold', type=float)
    parser.add_argument('--pseudolabel_T2', type=float, help='Percentage of total iterations at which to end linear scheduling and hold lambda at the max value')
    parser.add_argument('--soft_pseudolabels', default=False, type=parse_bool, const=True, nargs='?')
    parser.add_argument('--algo_log_metric')
    parser.add_argument('--process_pseudolabels_function', choices=supported.process_pseudolabels_functions)

    # Model selection
    parser.add_argument('--val_metric')
    parser.add_argument('--val_metric_decreasing', type=parse_bool, const=True, nargs='?')

    # Optimization
    parser.add_argument('--n_epochs', type=int)
    parser.add_argument('--optimizer', choices=supported.optimizers)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--max_grad_norm', type=float)
    parser.add_argument('--optimizer_kwargs', nargs='*', action=ParseKwargs, default={},
                        help='keyword arguments for optimizer initialization passed as key1=value1 key2=value2')

    # Scheduler
    parser.add_argument('--scheduler', choices=supported.schedulers)
    parser.add_argument('--scheduler_kwargs', nargs='*', action=ParseKwargs, default={},
                        help='keyword arguments for scheduler initialization passed as key1=value1 key2=value2')
    parser.add_argument('--scheduler_metric_split', choices=['train', 'val'], default='val')
    parser.add_argument('--scheduler_metric_name')

    # Evaluation
    parser.add_argument('--process_outputs_function', choices = supported.process_outputs_functions)
    parser.add_argument('--evaluate_all_splits', type=parse_bool, const=True, nargs='?', default=True)
    parser.add_argument('--eval_splits', nargs='+', default=[])
    parser.add_argument('--eval_only', type=parse_bool, const=True, nargs='?', default=False)
    parser.add_argument('--eval_epoch', default=None, type=int, help='If eval_only is set, then eval_epoch allows you to specify evaluating at a particular epoch. By default, it evaluates the best epoch by validation performance.')

    # Misc
    parser.add_argument('--device', type=int, nargs='+', default=[0])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--log_dir', default='./logs')
    parser.add_argument('--log_every', default=50, type=int)
    parser.add_argument('--save_step', type=int)
    parser.add_argument('--save_best', type=parse_bool, const=True, nargs='?', default=True)
    parser.add_argument('--save_last', type=parse_bool, const=True, nargs='?', default=True)
    parser.add_argument('--save_pred', type=parse_bool, const=True, nargs='?', default=True)
    parser.add_argument('--no_group_logging', type=parse_bool, const=True, nargs='?')
    parser.add_argument('--progress_bar', type=parse_bool, const=True, nargs='?', default=False)
    parser.add_argument('--resume', type=parse_bool, const=True, nargs='?', default=False, help='Whether to resume from the most recent saved model in the current log_dir.')
    parser.add_argument('--correct_label_shift', type=str, const=supported.label_shift_estimation_data_sets[0],
                        nargs='?', default=None, choices=supported.label_shift_estimation_data_sets,
                        help='Whether to also print results with label shift correction through Expectation Maximization with Bias-Corrected Temperature Scaling')
    parser.add_argument('--label_shift_estimation_grouping', type=str, nargs='+', default=None,
                        choices=supported.label_shift_estimation_groupings)

    # Weighted ERM
    parser.add_argument('--erm_weights', default=None, type=str, help='Weights to use for BBSE')
    
    #Bagging
    parser.add_argument('--bagging', type=parse_bool, const=True, nargs='?', help='If true, use the Bagging method to train multiple predictors.', default=False)
    parser.add_argument('--bagging_size', default=1, type=int, help='Number of predictors for the Bagging method.')
    parser.add_argument('--bagging_seeds', type=int, nargs='+', default=[0])

    # Weights & Biases
    parser.add_argument('--use_wandb', type=parse_bool, const=True, nargs='?', default=False)
    parser.add_argument('--wandb_api_key_path', type=str,
                        help="Path to Weights & Biases API Key. If use_wandb is set to True and this argument is not specified, user will be prompted to authenticate.")
    parser.add_argument('--wandb_kwargs', nargs='*', action=ParseKwargs, default={},
                        help='keyword arguments for wandb.init() passed as key1=value1 key2=value2')

    config = parser.parse_args()
    config = populate_defaults(config)

    # Set device
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if len(config.device) > device_count:
            raise ValueError(f"Specified {len(config.device)} devices, but only {device_count} devices found.")

        config.use_data_parallel = len(config.device) > 1
        device_str = ",".join(map(str, config.device))
        os.environ["CUDA_VISIBLE_DEVICES"] = device_str
        config.device = torch.device(f"cuda:{config.device[0]}")
    else:
        config.use_data_parallel = False
        config.device = torch.device("cpu")

    assert config.bagging_size == len(config.bagging_seeds)             #--bagging_size needs to be the same as the number of seeds.
    assert ((config.bagging == True) & (len(config.bagging_seeds)>1)) | (len(config.bagging_seeds)==1)  #--bagging needs to be set to True to be used.

    main_log_dir = config.log_dir

    for bag_seed in config.bagging_seeds:           #Bagging loop
        print('baggin',  config.bagging)
        print('eval',  config.eval_only)
        if config.bagging & (not config.eval_only):          #Bagging adaptation if used
            config.seed = bag_seed
            config.log_dir = main_log_dir + "/bag" + str(bag_seed) + "/"

        # Initialize logs
        if os.path.exists(config.log_dir) and config.resume:
            resume=True
            mode='a'
        elif os.path.exists(config.log_dir) and config.eval_only:
            resume=False
            mode='a'
        else:
            resume=False
            mode='w'

        if not os.path.exists(config.log_dir):
            os.makedirs(config.log_dir)
        logger = Logger(os.path.join(config.log_dir, 'log.txt'), mode)
        
        if config.bagging:          #Bagging iteration in log
            logger.write(f'Bagging with seed {bag_seed}.')

        # Record config
        log_config(config, logger)

        # Set random seed
        set_seed(config.seed)

        # Data
        full_dataset = wilds.get_dataset(
            dataset=config.dataset,
            version=config.version,
            root_dir=config.root_dir,
            download=config.download,
            split_scheme=config.split_scheme,
            **config.dataset_kwargs)

        # Transforms & data augmentations for labeled dataset
        # To modify data augmentation, modify the following code block.
        # If you want to use transforms that modify both `x` and `y`,
        # set `do_transform_y` to True when initializing the `WILDSSubset` below.
        train_transform = initialize_transform(
            transform_name=config.transform,
            config=config,
            dataset=full_dataset,
            additional_transform_name=config.additional_train_transform,
            is_training=True)
        eval_transform = initialize_transform(
            transform_name=config.transform,
            config=config,
            dataset=full_dataset,
            is_training=False)

        # Configure unlabeled datasets
        unlabeled_dataset = None
        if config.unlabeled_split is not None:
            split = config.unlabeled_split
            full_unlabeled_dataset = wilds.get_dataset(
                dataset=config.dataset,
                version=config.unlabeled_version,
                root_dir=config.root_dir,
                download=config.download,
                unlabeled=True,
                **config.dataset_kwargs
            )
            train_grouper = CombinatorialGrouper(
                dataset=[full_dataset, full_unlabeled_dataset],
                groupby_fields=config.groupby_fields
            )

            # Transforms & data augmentations for unlabeled dataset
            if config.algorithm == "FixMatch":
                # For FixMatch, we need our loader to return batches in the form ((x_weak, x_strong), m)
                # We do this by initializing a special transform function
                unlabeled_train_transform = initialize_transform(
                    config.transform, config, full_dataset, is_training=True, additional_transform_name="fixmatch"
                )
            else:
                # Otherwise, use the same data augmentations as the labeled data.
                unlabeled_train_transform = train_transform

            if config.algorithm == "NoisyStudent":
                # For Noisy Student, we need to first generate pseudolabels using the teacher
                # and then prep the unlabeled dataset to return these pseudolabels in __getitem__
                print("Inferring teacher pseudolabels for Noisy Student")
                assert config.teacher_model_path is not None
                if not config.teacher_model_path.endswith(".pth"):
                    # Use the best model
                    config.teacher_model_path = os.path.join(
                        config.teacher_model_path,  f"{config.dataset}_seed:{config.seed}_epoch:best_model.pth"
                    )

                d_out = infer_d_out(full_dataset, config)
                teacher_model = initialize_model(config, d_out).to(config.device)
                load(teacher_model, config.teacher_model_path, device=config.device)
                # Infer teacher outputs on weakly augmented unlabeled examples in sequential order
                weak_transform = initialize_transform(
                    transform_name=config.transform,
                    config=config,
                    dataset=full_dataset,
                    is_training=True,
                    additional_transform_name="weak"
                )
                unlabeled_split_dataset = full_unlabeled_dataset.get_subset(split, transform=weak_transform, frac=config.frac)
                sequential_loader = get_eval_loader(
                    loader=config.eval_loader,
                    dataset=unlabeled_split_dataset,
                    grouper=train_grouper,
                    batch_size=config.unlabeled_batch_size,
                    **config.unlabeled_loader_kwargs
                )
                teacher_outputs = infer_predictions(teacher_model, sequential_loader, config)
                teacher_outputs = move_to(teacher_outputs, torch.device("cpu"))
                unlabeled_split_dataset = WILDSPseudolabeledSubset(
                    reference_subset=unlabeled_split_dataset,
                    pseudolabels=teacher_outputs,
                    transform=unlabeled_train_transform,
                    collate=full_dataset.collate,
                )
                teacher_model = teacher_model.to(torch.device("cpu"))
                del teacher_model
            else:
                unlabeled_split_dataset = full_unlabeled_dataset.get_subset(
                    split, 
                    transform=unlabeled_train_transform, 
                    frac=config.frac, 
                    load_y=config.use_unlabeled_y
                )

            unlabeled_dataset = {
                'split': split,
                'name': full_unlabeled_dataset.split_names[split],
                'dataset': unlabeled_split_dataset
            }
            unlabeled_dataset['loader'] = get_train_loader(
                loader=config.train_loader,
                dataset=unlabeled_dataset['dataset'],
                batch_size=config.unlabeled_batch_size,
                uniform_over_groups=config.uniform_over_groups,
                grouper=train_grouper,
                distinct_groups=config.distinct_groups,
                n_groups_per_batch=config.unlabeled_n_groups_per_batch,
                **config.unlabeled_loader_kwargs
            )
        else:
            train_grouper = CombinatorialGrouper(
                dataset=full_dataset,
                groupby_fields=config.groupby_fields
            )

        # Configure labeled torch datasets (WILDS dataset splits)
        datasets = defaultdict(dict)
        for split in full_dataset.split_dict.keys():
            if split == 'train':
                transform = train_transform
                verbose = True
            elif split == 'val':
                transform = eval_transform
                verbose = True
            else:
                transform = eval_transform
                verbose = False
            # Get subset
            datasets[split]['dataset'] = full_dataset.get_subset(
                split,
                frac=config.frac,
                transform=transform)

            merged_df = pd.merge(datasets[split]['dataset'].dataset.metadata.iloc[datasets[split]['dataset'].indices], 
                                pd.DataFrame({"usage": datasets[split]['dataset'].dataset._split_array}), 
                                left_index=True, right_index=True)                  #add split information as a new column with metadata
            usage_text = list(datasets[split]['dataset']._split_names.values())
            merged_df['usage'] = merged_df['usage'].apply(lambda x:usage_text[int(x)] )     #convert split information id into text labels
            merged_df['region'] = merged_df['region'].apply(lambda x:datasets[split]['dataset'].metadata_map['region'][x])
            merged_df.to_csv(config.log_dir + "/split_" + split + "_metadata.csv")             #export metadata per training set

            if split == 'train':
                datasets[split]['loader'] = get_train_loader(
                    loader=config.train_loader,
                    dataset=datasets[split]['dataset'],
                    batch_size=config.batch_size,
                    uniform_over_groups=config.uniform_over_groups,
                    grouper=train_grouper,
                    distinct_groups=config.distinct_groups,
                    n_groups_per_batch=config.n_groups_per_batch,
                    **config.loader_kwargs)
            else:
                datasets[split]['loader'] = get_eval_loader(
                    loader=config.eval_loader,
                    dataset=datasets[split]['dataset'],
                    grouper=train_grouper,
                    batch_size=config.batch_size,
                    **config.loader_kwargs)

            # Set fields
            datasets[split]['split'] = split
            datasets[split]['name'] = full_dataset.split_names[split]
            datasets[split]['verbose'] = verbose

            # Loggers
            datasets[split]['eval_logger'] = BatchLogger(
                os.path.join(config.log_dir, f'{split}_eval.csv'), mode=mode, use_wandb=config.use_wandb
            )
            datasets[split]['algo_logger'] = BatchLogger(
                os.path.join(config.log_dir, f'{split}_algo.csv'), mode=mode, use_wandb=config.use_wandb
            )

        if config.use_wandb:
            initialize_wandb(config)

        # Logging dataset info
        # Show class breakdown if feasible
        if config.no_group_logging and full_dataset.is_classification and full_dataset.y_size==1 and full_dataset.n_classes <= 10:
            log_grouper = CombinatorialGrouper(
                dataset=full_dataset,
                groupby_fields=['y'])
        elif config.no_group_logging:
            log_grouper = None
        else:
            log_grouper = train_grouper
        log_group_data(datasets, log_grouper, logger)
        if unlabeled_dataset is not None:
            log_group_data({"unlabeled": unlabeled_dataset}, log_grouper, logger)

        # Initialize algorithm & load pretrained weights if provided
        algorithm = initialize_algorithm(
            config=config,
            datasets=datasets,
            train_grouper=train_grouper,
            unlabeled_dataset=unlabeled_dataset,
        )

        model_prefix = get_model_prefix(datasets['train'], config)
        if not config.eval_only:
            # Resume from most recent model in log_dir
            resume_success = False
            if resume:
                save_path = model_prefix + 'epoch:last_model.pth'
                if not os.path.exists(save_path):
                    epochs = [
                        int(file.split('epoch:')[1].split('_')[0])
                        for file in os.listdir(config.log_dir) if file.endswith('.pth')]
                    if len(epochs) > 0:
                        latest_epoch = max(epochs)
                        save_path = model_prefix + f'epoch:{latest_epoch}_model.pth'
                try:
                    prev_epoch, best_val_metric = load(algorithm, save_path, device=config.device)
                    epoch_offset = prev_epoch + 1
                    logger.write(f'Resuming from epoch {epoch_offset} with best val metric {best_val_metric}')
                    resume_success = True
                except FileNotFoundError:
                    pass
            if resume_success == False:
                epoch_offset=0
                best_val_metric=None

            # Log effective batch size
            if config.gradient_accumulation_steps > 1:
                logger.write(
                    (f'\nUsing gradient_accumulation_steps {config.gradient_accumulation_steps} means that')
                    + (f' the effective labeled batch size is {config.batch_size * config.gradient_accumulation_steps}')
                    + (f' and the effective unlabeled batch size is {config.unlabeled_batch_size * config.gradient_accumulation_steps}' 
                        if unlabeled_dataset and config.unlabeled_batch_size else '')
                    + ('. Updates behave as if torch loaders have drop_last=False\n')
                )

            train(
                algorithm=algorithm,
                datasets=datasets,
                general_logger=logger,
                config=config,
                epoch_offset=epoch_offset,
                best_val_metric=best_val_metric,
                unlabeled_dataset=unlabeled_dataset,
            )
        elif not config.bagging:
            if config.eval_epoch is None:
                eval_model_path = model_prefix + 'epoch:best_model.pth'
            else:
                eval_model_path = model_prefix + f'epoch:{config.eval_epoch}_model.pth'

            eval_model_path = eval_model_path.replace(':', '_')

            best_epoch, best_val_metric = load(algorithm, eval_model_path, device=config.device)
            if config.eval_epoch is None:
                epoch = best_epoch
            else:
                epoch = config.eval_epoch
            if epoch == best_epoch:
                is_best = True
            evaluate(
                algorithm=algorithm,
                datasets=datasets,
                epoch=epoch,
                general_logger=logger,
                config=config,
                is_best=is_best)
        else:           #Bagging evaluation special case
            #assert config.frac == 1         #always use the full dataset to evaluate the joint Bagging model
            print('Evaluating Bagging')
            bag_pred = {}
            split_pred = {}
            split_acc = {}
            result_line = pd.DataFrame()
            if config.eval_epoch is None:
                model_name = 'last_model.pth'
            else:
                model_name = f'{config.eval_epoch}_model.pth'
            # Get Bagged models
            directories = next(os.walk(config.log_dir))[1]
            print(directories)
            for folder in directories: 
                print(folder)
                if folder.startswith('bag'):
                    bag_pred.update({folder:{}})
                    file_names = os.listdir(config.log_dir + "/" + folder)
                    for file in file_names:
                        if file.endswith(model_name):
                            print(file)
                            bag_seed = re.sub('bag','',folder)
                            eval_model_path = config.log_dir + "/" + folder + "/" + file
                            #bag_models.update({folder: eval_model_path})
                            load(algorithm, eval_model_path, device=config.device)
                            config.seed = bag_seed          #setting seed parameter for log file names
                            full_preds = evaluate(
                                            algorithm=algorithm,
                                            datasets=datasets,
                                            epoch='Last',
                                            general_logger=logger,
                                            config=config,
                                            is_best=False)
                            for split, (epoch_y_pred, epoch_y_true, epoch_metadata) in full_preds.items():
                                df = pd.DataFrame({'pred cat': epoch_y_pred, 'true cat': epoch_y_true}) 
                                df[['region','year','true true cat','split']] = epoch_metadata.detach().numpy()      #.str.split(',', expand=True).astype(int)
                                bag_pred[folder].update({split: df})
                                split_pred.update({split:pd.DataFrame(0,index=[i for i in range(0,len(epoch_y_pred))], columns=[i for i in range(0,62)])})
                                bag_pred[folder][split].to_csv(config.log_dir + "/" + folder +'_' + split +'.csv', index=False, header=bag_pred[folder][split].columns)
                                #df = pd.DataFrame(epoch_y_true.numpy())
                                #df.to_csv(config.log_dir + "/" + folder +'_' + split +'_full_y_true.csv', index=False, header=False)
                                #df = pd.DataFrame(epoch_metadata.numpy())
                                #df.to_csv(config.log_dir + "/" + folder +'_' + split +'_full_y_metadata.csv', index=False, header=False)
            for bag, df_dict in bag_pred.items():
                for split, df in df_dict.items():
                    split_pred[split] += pd.get_dummies(list(df['pred cat'])+[i for i in range(0,62)]).iloc[:-62]
            for split, df in split_pred.items():
                split_acc.update({split: pd.concat([bag_pred[next(iter(bag_pred))][split]['region'],(df.idxmax(axis=1) == bag_pred[next(iter(bag_pred))][split]['true cat']).rename('correct')],axis=1)})
                split_acc[split].to_csv(config.log_dir + "/" + split +'_acc.csv', index=False, header=split_acc[split].columns)
                pivott = split_acc[split].drop(split_acc[split].index[split_acc[split]['region'] == 5]).pivot_table(index=['region'],columns=['correct'],aggfunc=len)  #dropping 'Other' region for worst region analysis     
                print(split,split_acc[split]['correct'].sum()/len(split_acc[split]))
                result_line['Average ' + split + ' Accuracy'] = [split_acc[split]['correct'].sum()/len(split_acc[split])]
                result_line['Average Per Region ' + split + ' Accuracy'] = [pivott.div(pivott.sum(axis=1),axis=0)[True].mean()]
                result_line['Worst Region ' + split + ' Accuracy'] = [pivott.div(pivott.sum(axis=1),axis=0)[True].min()]
            result_line.to_csv(config.log_dir + "/result_summary.csv", index=False, header=result_line.columns)

        if config.use_wandb:
            wandb.finish()
        for split in datasets:
            datasets[split]['eval_logger'].close()
            datasets[split]['algo_logger'].close()
    sys.stdout.close()
    logger.close()

if __name__ == '__main__':
    main()
