# The_WILD_Guess


## Interesting informations

To train the model:
- Go to the wilds folder
- Execute: `./run_expt.py -d fmow --algorithm ERM  --root_dir ./data --download` 


To add data loading workers, add this argument to the `run_expt.py`: `--loader_kwargs "num_workers=8"`
(8 because I have an 8-core CPU)

To speed up the data transfer between host and GPU: `--loader_kwargs pin_memory=True`

The `wilds/configs/datasets.py` file  contains default  training config for each dataset.

The `wilds/models/initializer` is the place where the model is created/initialized (`def initialize_model(config, d_out, is_featurizer=False)`)

## Result sharing convention & analytics
### Logs & results sharing

The logs files will be shared on the Google Drive in the `IFT6759 - The WILD Guess Team\Logs` folder. \
Naming convention: <first_name>\_\<model>\_\<method>\_<partial/full data>\_exp<#>         _ex: `Nathan_ERM_Baseline_full_exp1`_ \
Link: https://drive.google.com/drive/folders/1-GfVHWnTdhvYA4-LM7mLIWidzdqZBQUE?usp=sharing

The one line result in `kpi_extract.txt` extracted with the `run_analysis.py` script described below needs to be copied in the Excel tracker file along with the model training command line. \
Link: https://docs.google.com/spreadsheets/d/1Z0vVkII57D0G3OWWFtW8muDY7glydptM/edit?usp=sharing&ouid=109019793128097247425&rtpof=true&sd=true
  
### Log visualisation \& results extract with homemade script

The `run_analyse.py` script can be used to extract results \& make useful plots from the logs coming from the `run_expt.py` script. The script will namely:
1. Plot the data split distributions.
2. Plot the Loss \& Accuracy curves.
3. Extract an Excel pre-formatted one line result summary for the model's "Best Epoch" (based on validation loss minima).

How to use the script:
1. Run `python run_analyse.py --log_dir <logs> --show --eval_only`
  where `<logs>` is the path to the directory where the logs can be found, where `--show` is a boolean argument to make figure pop-ups appear sequentially (omitting it will stop the pop-ups) and where `--eval_only` only performs the KPI extract based on the evaluation logs (w/o training logs).
2. All the figures & text file will be saved in the `<logs>` directory.
3. Copy the `kpi_extract.txt` content in the above shared Excel tracker file, adding also the command line used for the model training with `run_expt.py` for tracking/reproducibility purposes.

### Log visualisation with Weights \& Biases (wand) package

1. `pip install wandb`
2. `wandb login`
3. `--use_wandb=True --wandb_kwargs project="wilds" entity="the-wild-guess"`

Get online to view results... (procedure TBD)

## Dataset split exploration
### Bootstrapping for evenly distributed splits

### Training & ID validation in the 2013-2015 range

### Bagging

## Model evaluation with label shift correction

### Expectation Minimization + Bias Corrected Temperature Scaling

Since this method requires probability distributions as predictions, an additional softmax is applied if any prediction doesn't sum to 1. 
To activate both while training & evaluating, add the argument `--correct_label_shift`. 
You can also specify the split to use for label distribution evaluation out of `train`, `id_val` & `val`. 

Example to evaluate best `convnet` model from `logs` folder:
```commandline
python wilds_examples/run_expt.py -d fmow --algorithm ERM --root_dir ./data --download --model convnet
--frac 0.01 --loader_kwargs "num_workers=8" --loader_kwargs pin_memory=True 
--correct_label_shift id_val --log_dir ./logs --eval_only
```

#### Grouping

In order to estimate label shift per grouping in the test sets, use the argument:

```commandline
--label_shift_estimation_grouping region year
```

You can group by either region, year or both depending on which are present in the argument.

### Label Shift Correction w/ Black Box Predictors
Note: The following method requires training two different models. The first model (baseline) can be trained using the standard ERM approach. After the baseline model is trained, we need to estimate the target label distribution on the test set. This can be done by running the following:

`wilds_examples/bbse/run_estimate_target_distribution.sh`

ensuring to update the following arguments to point to:

`yval`: The true labels for the in-domain validation set

`ytest`: The true labels for the OOD test set

`ypred_source`: The predictions for the in-domain validation set

`ypred_target`: The predictions for the OOD test set

This will output a class weights file, which should be used to train a second model (using `run_expt.py`) with the additional argument `--erm_weights`.
