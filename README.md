# The_WILD_Guess


## Interesting informations

To train the model:
- Go to the wilds folder
- Execute: `python run_expt.py -d fmow --algorithm ERM  --root_dir ./data --download`


To add data loading workers, add this argument to the `run_expt.py`: `--loader_kwargs "num_workers=8"`
(8 because I have an 8-core CPU)

To speed up the data transfer between host and GPU: `--loader_kwargs pin_memory=True`

The `wilds/configs/datasets.py` file  contains default  training config for each dataset.

The `wilds/models/initializer` is the place where the model is created/initialized (`def initialize_model(config, d_out, is_featurizer=False)`)

 
To run the Visual Transformer run:
`python run_expt.py -d fmow --model vit --algorithm ERM  --root_dir ./data --loader_kwargs pin_memory=True --loader_kwargs "num_workers=26" --model_kwargs="model_size=B_16" --model_kwargs="pretrained=True" --device=0`

To run a WandB sweep:
1. `wandb sweep sweep.yaml`
2. then start as many agents as your hardware allows:
    - `CUDA_VISIBLE_DEVICES=0 wandb agent SWEEP_ID` agent 1
    - `CUDA_VISIBLE_DEVICES=0 wandb agent SWEEP_ID` agent 2

sweep agents can be started on multiple slurm / cloud instances or hardware devices, as long as you're logged in to WandB on each machine. 
