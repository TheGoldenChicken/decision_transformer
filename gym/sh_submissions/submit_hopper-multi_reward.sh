#!/bin/sh
#BSUB -J DT-hopper_MR
#BSUB -o DT-hopper_MR_%J.out
#BSUB -e DT-hopper_MR_%J.err
#BSUB -q gpuv100
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -n 4
#BSUB -R "rusage[mem=4G]"
#BSUB -R "span[hosts=1]"
#BSUB -W 24:00
# end of BSUB options


# load CUDA (for GPU support)

# activate the virtual environment
source $HOME/miniconda3/envs/decision-transformer-gym/bin/activate

wandb on
echo 'cad8b043f3731a2c453efd8f61915e186ac93ac3' | wandb login

python experiment_multiple_rewards.py --env 'hopper' --dataset 'medium_replay' --save_iters '5,10,11,12' --eval_iters '8,9,10,11,12' --max_iters 12 --device 'cuda' --split_reward True --seed 44 --num_eval_episodes 20 --num_steps_per_iter 10000
