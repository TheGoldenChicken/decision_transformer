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

module load cuda/11.6

# activate the virtual environment
source $HOME/miniconda3/envs/decision-transformer-gym/bin/activate

wandb on
echo 'cad8b043f3731a2c453efd8f61915e186ac93ac3' | wandb login

python experiment_single_reward.py --env 'halfcheetah' --dataset 'medium_replay' --seed 42 --save_iters '8,10,12' --eval_iters '8,9,10,11,12' --max_iters 12 --device 'cuda' --log_to_wandb True
