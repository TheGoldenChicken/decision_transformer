
# Multi reward decision making using offline reinforcement learning

Calle Ryge Carlsen, Christian Ole Nielsen, Karl Meisner-Jensen, Magnus Elgaard Bennett

Based on code originally from berkely (license MIT) [github](https://github.com/kzl/decision-transformer)

In relation to the paper *Decision Transformer: Reinforcement Learning Using Sequence modelling* - Lu et. al. 2021. Paper found at [arXiv](https://arxiv.org/abs/2106.01345)


## Overview

Using the decision transformer (DT) to perform offline reinforcement learning in Markovian Gym MuJoCo environments.

![image info](./architecture.png)

The multi-return case for the transformer has been introduced, allowing to condition on multiple return signals, as well as code to generate the multi-return data. 

several submit_environment_case.sh files are included to allow for easy training on DTU HPC. Otherwise performing experiments has been easened when using the console.

## Instructions

See /gym/readme-gym.md on initializing environment and common errors associated with this.

All code associated with the decision trannsformer is found in the /gym folder	

## License

DTU
