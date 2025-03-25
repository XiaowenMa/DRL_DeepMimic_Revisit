## DeepMimic Revisited
This repo contains the dev scripts for a pytorch-mujoco version of original [DeepMimic](https://xbpeng.github.io/projects/DeepMimic/index.html) paper for Deep Reinforcement Learning project. Due to the high complexity of the original C++ project, the goal for this project is to reimplement the physics-based humanoid walking task in Mujoco using PPO with tailored reward functions plus adding a minor experimental change.

For midterm checkpoint, we've currently implemented 
- Mocap data transformation
- Customized humanoid model
- Customized Gym envrionment with tailored reward function(for joint position/velocity and CoM matching). 
- PPO with GAE for optimization. 

The simulation was run at 500Hz and mocap data was at 30Hz; we used 500 horizon steps for training. Initial demo video after 30 training iterations can be found under demos folder(The demo video was at 100fps).

We also have a paralleled approach implemented in loco-mujoco, which provides high frequency ref dataset.


## Current stage and Next Steps
Tuning the architecture/using ensemble and adding replay buffer to improve learning. Propose a minor delta version to the baseline: 
- Try different optimization such as TRPO/GAIL
- Train the agent to walk on uneven terrain

## Demo
The following is an demo of the agent after 30 iterations of training(simulated at 500Hz, video rendered at 100fps). (Agent falling after first step...)

![Demo](https://github.com/user-attachments/assets/53680baa-590d-4c71-aa56-122a984433db)
