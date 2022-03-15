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
