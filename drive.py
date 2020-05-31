import argparse

from models.sac import SAC
from environments.donkey_car import DonkeyCar
from environments.donkey_sim import DonkeySim

from config import STEER_LIMIT_LEFT, STEER_LIMIT_RIGHT, THROTTLE_MAX, THROTTLE_MIN,, MAX_STEERING_DIFF, MAX_EPISODE_STEPS

parser = argparse.ArgumentParser()

parser.add_argument("--sac", help="File to save or load SAC from", default="")
parser.add_argument("--vae", help="File to load VAE from", default="")
parser.add_argument("--existing_model", help="Continue training an existing model", default="")
parser.add_argument("--train", help="Train the model", default=True)
parser.add_argument("--max_episode_length", help="Max steps in episode", default=1000)

parser.add_argument("--env_type", help="Is this DonkeyCar or DonkeySim", default="DonkeySim")
parser.add_argument("--car_name", help="Name of the car on Mqtt-server", default="Kari")
parser.add_argument("--mqtt_server", help="Name of the car on Mqtt-server", default="Kari")

parser.add_argument("--random_episodes", help="Number of random episodes at the start", default=5, type=int)
parser.add_argument("--training_steps", help="Number of gradient steps for SAC per episode", default=600, type=int)

args = parser.parse_args()

if args.existing_model:
    sac = torch.load(args.vae)
else:
    sac = SAC()

vae = torch.load(args.vae)

env = None
if args.env_type == "DonkeySim":
    env = DonkeySim(args.car_name)
elif args.env_type == "DonkeyCar":
    env = DonkeySim(args.car_name)

action_space = spaces.Box(
    low=np.array([STEER_LIMIT_LEFT, THROTTLE_MIN]), 
    high=np.array([STEER_LIMIT_RIGHT, THROTTLE_MAX]), dtype=np.float32)


env.reset()

for e in range(10000):

    if True:
        




