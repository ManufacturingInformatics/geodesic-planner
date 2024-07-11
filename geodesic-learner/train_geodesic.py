import os
import glob
import time
from datetime import datetime
import torch
import numpy as np
import gymnasium as gym
from ppo import PPO
from environment import CurveEnv
import argparse
from tqdm import tqdm
import torchquad
from torchquad import set_up_backend
import warnings
import polars as pl

def train() -> None:
    
    # Argument parser for various parameters
    parser = argparse.ArgumentParser(
        description='Geodesic Policy Search via PPO'
    )
    parser.add_argument('-e', '--episodes', type=int, help='Number of episodes', default=10)
    parser.add_argument('-p', '--path', type=str, help='Path to save the model weights')
    parser.add_argument('-r', '--rep', type=int)
    
    args = parser.parse_args()
    
    # A path must must be supplied in order to save the model. 
    if args.path is None:
        print("Path cannot be None. Please include a path for saving model weights")
        exit()
    
    # Reason for this: there are some warnings due to mixmatched versions of torchquad and PyTorch. They have been suppressed for now, but this should be fixed downstream by torchquad
    warnings.filterwarnings("ignore")
    
    # Set device and torchquad backend
    print("========================================")
    # set device to cpu or cuda
    device = torch.device('cpu')
    if(torch.cuda.is_available()): 
        device = torch.device('cuda:0') 
        torch.cuda.empty_cache()
        print("Device set to : " + str(torch.cuda.get_device_name(device)))
    else:
        print("Device set to : cpu")
    print("========================================")
    
    # Configure the torchquad backend.
    set_up_backend('torch', data_type='float32')
    
    var_measure_const = np.loadtxt('../data/heat_map.csv')
    
    ##### Environment Parameters #####
    
    has_continuous_action_spaces = True
    
    env = CurveEnv([1,7], [8,4.5], device, np.rot90(var_measure_const))
    max_ep_len = env._max_episode_steps
    max_training_timesteps = int(max_ep_len*args.episodes)
    
    print_freq = int(max_ep_len*2)
    save_model_freq = int(max_ep_len*3)
    
    action_std = float(3)
    action_std_decay_rate = 0.05
    min_action_std = 0.01
    action_std_decay_freq = int(2e5)
    
    ##### PPO Hyperparameters #####
    
    update_timestep = max_ep_len*2.5
    K_epochs = 80
    eps_clip = 0.2
    gamma = 0.99
    lr_actor = 0.003
    lr_critic = 0.001
    
    random_seed = 0
    
    ##### Training #####
    
    state_dim = env.observation_space.shape[0]
    if has_continuous_action_spaces:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n
    
    agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_spaces, device, action_std)
    
    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    
    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0
    
    param_storage = np.zeros((max_training_timesteps+max_ep_len, 8))
    
    while time_step <= max_training_timesteps:
        
        state, L, _ = env.reset()
        current_ep_reward = 0
        
        for t in tqdm(range(0, max_ep_len)):
            
            action = agent.select_action([state])
            state, reward, vals, done, params, cl = env.step(t, action)
            
            # Render environment
            x, y = vals
            env.render(t, x, y, cl)
            
            agent.buffer.rewards.append(reward)
            agent.buffer.is_terminals.append(done)
            
            time_step += 1
            current_ep_reward += reward
            
            if time_step % update_timestep == 0:
                agent.update()
                
            if has_continuous_action_spaces and time_step % action_std_decay_freq == 0:
                agent.decay_action_std(action_std_decay_rate, min_action_std)
            
            # Storing the parameters for plotting later
            [ax, bx, cx, ay, by, cy] = params
            param_storage[t+(i_episode*max_ep_len), :] = [ax, bx, cx, env.dx, ay, by, cy, env.dx]
                
            if done:
                break
        
        print("Episode : {} \t Timestep : {} \t Episode Reward : {}".format(i_episode+1, time_step, current_ep_reward.detach().cpu().numpy()[0]))
        print_running_reward += current_ep_reward
        print_running_episodes += 1
        
        i_episode += 1
    
    env.close()
    # print total training time
    print("========================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("========================================")    
    
    # Save the model for loading later
    torch.save(agent.policy.actor.state_dict(), '{}/policy_{}.pt'.format(args.path, datetime.now().date()))     
    df = pl.from_numpy(param_storage, schema=['ax', 'bx', 'cx', 'dx', 'ay', 'by', 'cy', 'dy']) 
    df.write_parquet('{}/policy_params_{}.parquet'.format(args.path, args.rep))

if __name__ == '__main__':
    train()