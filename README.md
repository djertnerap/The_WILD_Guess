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
1. Run `python run_analyse.py --log_dir <logs> --show`
  where `<logs>` is the path to the directory where the logs can be found and `--show` is a boolean argument to make figure pop-ups appear sequentially (omitting it will stop the pop-ups).
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

Example to evaluate best model from `logs` folder:
```commandline
python wilds_examples -d fmow --algorithm ERM --root_dir ./data --download --model convnet
--frac 0.01 --loader_kwargs "num_workers=8" --loader_kwargs pin_memory=True --correct_label_shift
--log_dir ./logs --eval_only
```
