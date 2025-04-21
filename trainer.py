from DRL import DRL
import tqdm
import torch
import os
from torch.utils.tensorboard import SummaryWriter
import time

model_path = f"saved_models/{time.time()}/"
os.makedirs(model_path,exist_ok=True)
def save_model(drl:DRL,i):
    torch.save({
        'value_state_dict': drl.agent.value.state_dict(),
        'policy_state_dict': drl.agent.policy.state_dict(),
        'value_out_state_dict': drl.agent.value_out.state_dict(),
        'policy_out_state_dict': drl.agent.policy_out.state_dict(),
        'optimizer_value_state_dict': drl.critic_optimizer.state_dict(),  # If you're using an optimizer
        'optimizer_policy_state_dict': drl.actor_optimizer.state_dict(),  # If you're using an optimizer
        'obs_mean':drl.obs_running_mean,
        'obs_var':drl.obs_running_var
    }, f'{model_path}model_checkpoint_{i}.pth')


if __name__=="__main__":
    drl = DRL()

    writer = SummaryWriter(log_dir="runs/drl_experiment_0419_traj_skipframes_contd_ET_1000iter")

    # for i in tqdm.tqdm(range(10)):
    #     drl.rollout(i)
    #     drl.update()
    for episode in tqdm.tqdm(range(1001)):
        drl.rollout(episode)
        reward = drl.get_avg_reward()

        drl.update3()

        policy_loss,value_loss = drl.get_avg_loss() 

        writer.add_scalar('Reward/episode', reward, episode)
        writer.add_scalar('Policy_Loss/episode', policy_loss, episode)
        writer.add_scalar('Value_Loss/episode', value_loss, episode)
        writer.add_scalar('Steps', drl.avg_steps, episode)
        

        if episode%100==0:
            save_model(drl,episode)
    writer.close()
