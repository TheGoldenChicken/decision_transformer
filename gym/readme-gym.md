
# OpenAI Gym

## Installation

Experiments require MuJoCo.
Follow the instructions in the [mujoco-py repo](https://github.com/openai/mujoco-py) to install.


MuJoCo200 program and licence can be downloaded from [MuJoCo](https://www.roboti.us/download.html) and [License](https://www.roboti.us/license.html)

Then, dependencies can be installed with the following command:

```
conda env create -f conda_env.yml
```

## Downloading datasets

Datasets are stored in the `data` directory.
Install the [D4RL repo](https://github.com/rail-berkeley/d4rl), following the instructions there.

Before running *pip install -e .* edit the file setup.py: replace lines 16 and 17 with the following:*’dm_control==0.0.364896371’,*

Then, run the following script in order to download the datasets and save them in our format:

```
python download_d4rl_datasets.py
```

## Example usage

Experiments can be reproduced with the following:

```
python experiment_single_reward.py --env 'halfcheetah' --dataset 'medium_replay' --seed 46 --save_iters '8,10,12' --eval_iters '8,9,10,11,12' --max_iters 12 --device 'cuda' --log_to_wandb True
```

Or submitted to the DTU HPC by the bash scripts in gym/sh_submissions .
