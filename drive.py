import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--sac", help="File to save or load SAC from", default='./')
parser.add_argument("--vae", help="File to save or load VAE from", default='')
parser.add_argument("--train", help="Train the model", default=True)
parser.add_argument("--max_episode_length", help="Max steps in episode", default=1000)

parser.add_argument("--env", help="Which car environment to use", default="donkey_sim")
parser.add_argument("--donkey_sim_path", help="Path to donkey simulator")
parser.add_argument("")


