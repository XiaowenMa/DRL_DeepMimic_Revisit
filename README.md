## DeepMimic Revisited
This repo contains the dev scripts for a pytorch-mujoco version of original DeepMimic paper for Deep Reinforcement Learning project. The goal is to reimplement the physics-based humanoid walking task using PPO with tailored reward functions.

For midterm checkin, we've currently implemented mocap data transformation, customized humanoid model, Gym envrionment with tailored reward function(for joint position/velocity and CoM). We used PPO with GAE for optimization. The simulation was run at 500Hz and mocap data was at 30Hz; we used 500 horizon steps for training. Initial demo video after 30 training iterations can be found under demos folder(* The demo video was at 100fps).

## Current stage and Next Step
Tuning the architecture and adding replay buffer to improve learning. Propose a minor delta version to the baseline: train the agent to walk on uneven terrain.

## Demo
The following is an demo of the agent after 30 iterations of training. (Agent falling after first step...)

<!-- [Demo](https://) -->