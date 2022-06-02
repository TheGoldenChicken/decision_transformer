#!/bin/sh
#BSUB -J DT-cheetah
#BSUB -o DT-cheetah_%J.out
#BSUB -e DT-cheetah_%J.err
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[mem=4G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 15:00
# end of BSUB options


# load CUDA (for GPU support)

# activate the virtual environment
source $HOME/miniconda3/envs/decision-transformer-gym/bin/activate

wandb on
echo 'cad8b043f3731a2c453efd8f61915e186ac93ac3' | wandb login

python experiment.py --env 'halfcheetah' --dataset 'medium_replay' --save_iters '5,10,11,12,13,14,15' --eval_iters '5,10,15' --max_iters 15 --device 'cuda' --log_to_wandb True --split_reward True
