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
<!-- The following is an demo of the agent after 1500 iterations of training(64 horizon)(simulated at 500Hz, policy updated/video recorded at 30Hz). -->
Here is a demonstration of the agent's performance after 1500 iterations of training, with a horizon of 64 steps. The simulation runs at 500Hz, while the policy is updated and the video is recorded at 30Hz.

<!-- 
![Demo](https://github.com/user-attachments/assets/53680baa-590d-4c71-aa56-122a984433db)
-->

<!-- <div align="center">
  <img src="https://github.com/user-attachments/assets/02ead9f5-4cb9-4843-b6e7-28260e5fedb2" width="700" />
</div> -->


<div align="center">
  <img src="https://github.com/user-attachments/assets/85e3b5b0-4f1a-465b-a37d-3fd987c9a9c8" width="700" />
</div>


