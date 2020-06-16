STEER_LIMIT_LEFT = -1
STEER_LIMIT_RIGHT = 1
THROTTLE_MAX = 0.21
THROTTLE_MIN = 0.2
MAX_STEERING_DIFF = 0.5

MAX_EPISODE_STEPS = 2000

COMMAND_HISTORY_LENGTH = 5 
FRAME_STACK = 3
VAE_OUTPUT = 10

LR_START = 0.0001
LR_END = 0.0001
ANNEAL_END_EPISODE = 50

IMAGE_SIZE = 40
RGB = True

PARAMS = {

    "sac": {
        "linear_output": VAE_OUTPUT + COMMAND_HISTORY_LENGTH * 2,
        "lr": LR_START,
        "target_entropy": -2,
        "batch_size": 128,
        "hidden_size": 100
        },
    "ae": {
        "framestack": FRAME_STACK,
        "output": VAE_OUTPUT,
        "linear_input": 100,
        "image_size": IMAGE_SIZE,
        "lr": LR_END,
        "image_channels": 3 if RGB else 1,
        "encoder_type": "vae"

    }

}

