# The_WILD_Guess


## Interesting informations

To train the model:
- Go to the wilds folder
- Execute: `./run_expt.py -d fmow --algorithm ERM  --root_dir ./data --download` 


To add data loading workers, add this argument to the `run_expt.py`: `--loader_kwargs "num_workers=8"`
(8 because I have an 8-core CPU)

To speed up the data transfer between host and GPU: `--loader_kwargs pin_memory=True`

The `examples/configs/datasets.py` file  contains default  training config for each dataset
The `examples/models/initializer` is the place where the model is created/initialized (`def initialize_model(config, d_out, is_featurizer=False)`)

 
