import argparse

from models.sac import SAC
from environments.donkey_car import DonkeyCar
from environments.donkey_sim import DonkeySim


parser = argparse.ArgumentParser()

parser.add_argument("--sac", help="File to save or load SAC from", default='./')
parser.add_argument("--vae", help="File to load VAE from", default='')
parser.add_argument("--train", help="Train the model", default=True)
parser.add_argument("--max_episode_length", help="Max steps in episode", default=1000)

parser.add_argument("--env_type", help="Is this DonkeyCar or DonkeySim", default="DonkeySim")
parser.add_argument("--car_name", help="Name of the car on Mqtt-server", default="Kari")
parser.add_argument("--mqtt_server", help="Name of the car on Mqtt-server", default="Kari")


args = parser.parse_args()

sac = torch.load(args.vae)
vae = torch.load(args.vae)

env = None
if args.env_type == "DonkeySim"
    env = DonkeySim(args.car_name)
