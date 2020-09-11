"""
Train and drive a Donkey Car using a VAE + SAC agent
"""

import argparse
import datetime
import time
import os
import random
import copy

import warnings

import torch
import numpy as np
from gym import spaces

from models.ae_sac  import AE_SAC

from environments.donkey_car import DonkeyCar
from environments.donkey_sim import DonkeySim
from environments.donkey_car_speed import DonkeyCarSpeed

from utils.functions import image_to_ascii

from config import STEER_LIMIT_LEFT, STEER_LIMIT_RIGHT, THROTTLE_MAX, THROTTLE_MIN, MAX_STEERING_DIFF, MAX_EPISODE_STEPS, RGB, \
                   COMMAND_HISTORY_LENGTH, FRAME_STACK, VAE_OUTPUT, PARAMS, IMAGE_SIZE, STEP_LENGTH

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

parser.add_argument("--random_episodes", help="Number of random episodes at the start", default=1, type=int)
parser.add_argument("--training_steps", help="Number of gradient steps for SAC per episode", default=600, type=int)
parser.add_argument("--model", help="Algorithm to use", default="ae_sac")
parser.add_argument("--record_folder", help="Folder for records", default="")

parser.add_argument("--continue_training", help="Continue training latest model", default=0, type=int)

args = parser.parse_args()

warnings.filterwarnings("ignore")

models = {"ae_sac": AE_SAC}

timestamp = datetime.datetime.today().isoformat()
model_name = "./trained_models/sac/SAC_{}.pth".format(timestamp)
record_name = "./records/{}log_sac_{}.csv".format(args.record_folder, timestamp)

if not os.path.isdir("./records/{}".format(args.record_folder)):
    os.mkdir("./records/{}".format(args.record_folder))

if args.existing_model:
    agent = torch.load(args.pretrained_model)
else:
    agent = models[args.model](PARAMS)

if args.continue_training:
    models = sorted(os.listdir("./trained_models/sac/"))
    record_name = sorted(os.listdir("./records/{}".format(args.record_folder)))[-1]
    print(record_name)
    if len(models) > 0:
        print("Loading existing model {}".format(models[-1]))
        agent = torch.load("./trained_models/sac/" + models[-1])
    else:
        print("No previous models, training a new one")


env = None
if args.env_type == "DonkeySim":
    env = DonkeySim(args.car_name)
elif args.env_type == "DonkeyCarSpeed":
    env = DonkeyCarSpeed(args.car_name)
elif args.env_type == "DonkeyCar":
    env = DonkeyCar(args.car_name)

action_space = spaces.Box(
    low=np.array([STEER_LIMIT_LEFT, THROTTLE_MIN]), 
    high=np.array([STEER_LIMIT_RIGHT, THROTTLE_MAX]), dtype=np.float32)

channels = 3 if RGB else 1

if not args.continue_training:
    if not args.env_type == "DonkeyCarSpeed":
        with open(record_name, "w+") as f:
            f.write("Episode;Steps;Reward;Time\n")
    else:
        with open(record_name, "w+") as f:
            f.write("Episode;Step;Reward;Time;Steering;Throttle;SpeedX;SpeedY;SpeedZ;PosX;PosY;PosZ\n")

def enforce_limits(action, prev_steering):

    """Scale the agent actions to environment limits"""

    var = (THROTTLE_MAX - THROTTLE_MIN) / 2
    mu = (THROTTLE_MAX + THROTTLE_MIN) / 2

    steering_min = max(STEER_LIMIT_LEFT, prev_steering - MAX_STEERING_DIFF)
    steering_max = min(STEER_LIMIT_RIGHT, prev_steering + MAX_STEERING_DIFF)

    steering = max(steering_min, min(steering_max, action[0]))

    return [steering, action[1] * var + mu]

def save_model(agent, name):
    print("Saving model")

    agent_2 = copy.deepcopy(agent)

    images = len(agent.replay_buffer.buffer)
    agent_2.replay_buffer.buffer = random.sample(agent.replay_buffer.buffer, k=min(images, 10000))

    torch.save(agent_2, model_name)

try:

    for e in range(args.episodes):
        input("Press Enter to Start")
        episode_reward = 0
        step = 0
        done = 0.0
        interrupted = 0

        # Initialize state variables

        command_history = np.zeros(3*COMMAND_HISTORY_LENGTH)

        img = env.reset()
        obs = agent.process_im(img, IMAGE_SIZE, RGB)
        action = [0, 0]

        # Frame stack

        state = np.vstack([obs for x in range(FRAME_STACK)])

        while step < MAX_EPISODE_STEPS:
            try:
                step += 1
                t1 = time.time_ns()

                # Select action

                if e < args.random_episodes:
                    action = action_space.sample()
                else:
                    action = agent.select_action((state, command_history))

                # Scale action and step environment

                limited_action = enforce_limits(action, command_history[0])
                img, dead = env.step(limited_action, STEP_LENGTH)

                obs = agent.process_im(img, IMAGE_SIZE, RGB)

                done = dead or interrupted

                # Decide reward

                base = 1

                if args.env_type == "DonkeyCarSpeed":
                    base += (env.speed - THROTTLE_MIN) / (THROTTLE_MAX - THROTTLE_MIN)

                reward = base if not done else -10 * base

                # Update next state variabels

                next_command_history = np.roll(command_history, 3)
                next_command_history[:3] = limited_action + [env.speed]

                next_state = np.roll(state, channels * FRAME_STACK)
                next_state[:channels * FRAME_STACK, :, :] = obs

                # Add step to episode buffer

                agent.push_buffer([(state, command_history), action, [reward], (next_state, next_command_history), [float(not done)]])

                # Print statistics

                episode_reward += reward
                t2 = time.time_ns()

                image_to_ascii(obs *255, 40)

                print("Episode: {}, Step: {}, Reward: {:.2f}, Episode reward: {:.2f}, Throttle: {:.2f}, Speed: {:.2f}, Time: {:.2f}".format(e, step, reward, episode_reward, action[1], env.speed, (t2 - t1) / 1e6))
                t1 = t2

                # Update state variables

                state = next_state
                command_history = next_command_history

                # Save episode statistics to file
                if args.env_type == "DonkeyCarSpeed":
                    with open(record_name, "a+") as f:
                        f.write("{};{};{};{};{};{};{};{};{};{};{};{}\n"
                                .format(e, step, reward, time.time(), *limited_action, *env.state["v"], *env.state["x"]))

                if done:
                    break

            except (KeyboardInterrupt, IndexError, NameError) as e:
                interrupted = 1
                continue

        if not args.env_type == "DonkeyCarSpeed":
            with open(record_name, "a+") as f:
                f.write("{};{};{};{}\n".format(e, step, episode_reward, datetime.datetime.today().isoformat()))


        # Stop the car

        env.reset()

        if e >= args.random_episodes:

            # Update agent

            print("Traning SAC")
            agent.update_parameters(args.training_steps)

except KeyboardInterrupt:
    save_model(agent, model_name)