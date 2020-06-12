import argparse
import datetime
import time

import torch
import numpy as np
from gym import spaces
import cv2

from models.sac import SAC
from models.ae_sac  import AE_SAC
from environments.donkey_car import DonkeyCar
from environments.donkey_sim import DonkeySim
from utils.functions import image_to_ascii

from config import STEER_LIMIT_LEFT, STEER_LIMIT_RIGHT, THROTTLE_MAX, THROTTLE_MIN, MAX_STEERING_DIFF, MAX_EPISODE_STEPS, \
                   COMMAND_HISTORY_LENGTH, FRAME_STACK, VAE_OUTPUT, LR_START, LR_END, ANNEAL_END_EPISODE, PARAMS, IMAGE_SIZE

parser = argparse.ArgumentParser()

parser.add_argument("--pretrained_model", help="File to save or load SAC from", default="")
parser.add_argument("--pretrained_vae", help="File to load VAE from", default="")
parser.add_argument("--existing_model", help="Continue training an existing model", default="")
parser.add_argument("--train", help="Train the model", default=True)
parser.add_argument("--max_episode_length", help="Max steps in episode", default=1000)
parser.add_argument("--episodes", help="How many episodes", default=1000, type=int)

parser.add_argument("--env_type", help="Is this DonkeyCar or DonkeySim", default="DonkeySim")
parser.add_argument("--car_name", help="Name of the car on Mqtt-server", default="Kari")
parser.add_argument("--mqtt_server", help="Name of the car on Mqtt-server", default="Kari")

parser.add_argument("--random_episodes", help="Number of random episodes at the start", default=5, type=int)
parser.add_argument("--training_steps", help="Number of gradient steps for SAC per episode", default=600, type=int)
parser.add_argument("--model", help="Algorithm to use", default="sac")

args = parser.parse_args()

models = {"ae_sac": AE_SAC}

if args.existing_model:
    agent = torch.load(args.pretrained_model)
else:
    agent = models[args.model](PARAMS)


env = None
if args.env_type == "DonkeySim":
    env = DonkeySim(args.car_name)
elif args.env_type == "DonkeyCar":
    env = DonkeySim(args.car_name)

action_space = spaces.Box(
    low=np.array([STEER_LIMIT_LEFT, THROTTLE_MIN]), 
    high=np.array([STEER_LIMIT_RIGHT, THROTTLE_MAX]), dtype=np.float32)

timestamp = datetime.datetime.today().isoformat()
model_name = "./trained_models/sac/SAC_{}.pth".format(timestamp)


with open("./records/log_sac_{}.csv".format(timestamp), "w+") as f:
    f.write("Episode;Reward;Time\n")


lr_step = (LR_START - LR_END) / (ANNEAL_END_EPISODE - args.random_episodes)
        

def enforce_limits(action, prev_steering):
     var = (THROTTLE_MAX - THROTTLE_MIN) / 2
     mu = (THROTTLE_MAX + THROTTLE_MIN) / 2
     
     steering_min = max(STEER_LIMIT_LEFT, prev_steering - MAX_STEERING_DIFF)
     steering_max = min(STEER_LIMIT_RIGHT, prev_steering + MAX_STEERING_DIFF)
     
     steering = max(steering_min, min(steering_max, action[0]))
     #print("Prev steering: {:.2f}, Steering min: {:.2f}, Steering max: {:.2f}, Action: {:.2f}, Steering: {:.2f}".format(prev_steering, steering_min, steering_max, action[0], steering))
     return [steering, action[1] * var + mu]

for e in range(args.episodes):
    
    episode_reward = 0
    step = 0
    done = 0.0
    interrupted = 0

    command_history = np.zeros(2*COMMAND_HISTORY_LENGTH)

    obs = env.reset()
    obs = cv2.resize(obs, (IMAGE_SIZE, IMAGE_SIZE)).reshape(3, IMAGE_SIZE, IMAGE_SIZE)
    action = [0, 0]

    state = np.vstack([obs for x in range(FRAME_STACK)])
    #print(state.shape)

    while step < MAX_EPISODE_STEPS:
        try:
            step += 1
            t1 = time.time_ns()

            if e < args.random_episodes:
                action = action_space.sample()
            else:
                action = agent.select_action((state, command_history))

            limited_action = enforce_limits(action, command_history[0])
            taken_action, obs, dead = env.step(limited_action)
            obs = cv2.resize(obs, (IMAGE_SIZE, IMAGE_SIZE)).reshape(3, IMAGE_SIZE, IMAGE_SIZE)
    
            done = dead or interrupted

            next_command_history = np.roll(command_history, 2)
            next_command_history[:2] = taken_action
            
            reward = 1 if not done else -10

            next_state = np.roll(state, 3)
            print(state.shape)
            next_state[:3, :, :] = obs

            agent.push_buffer([(state, command_history), action, [reward], (next_state, next_command_history), [float (not done)]])

            episode_reward += reward
            t2 = time.time_ns()

            image_to_ascii(obs, 20)

            print("Episode: {}, Step: {}, Reward: {:.2f}, Episode reward: {:.2f}, Time: {:.2f}".format(e, step, reward, episode_reward, (t2 - t1) / 1e6))
            t1 = t2

            state = next_state
            command_history = next_command_history

            if done:
                break

        except KeyboardInterrupt:
            interrupted = 1
            continue

    with open("./records/log_sac_{}.csv".format(timestamp), "a+") as f:
        f.write("{};{};{}\n".format(e, episode_reward, datetime.datetime.today().isoformat()))  

    env.step((0,0))
    time.sleep(2)
    env.step((0,0.01))

    if e == 20:
        agent.update_lr(0.0001)

    print("Traning SAC")
    if e >= args.random_episodes:
        #if e < ANNEAL_END_EPISODE:
            #agent.update_lr(LR_START - lr_step * (e - args.random_episodes))
        
        for i in range(args.training_steps):
            agent.update_parameters()

    print("Saving model")
    torch.save(agent, model_name)





            

    
        




