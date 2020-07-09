STEER_LIMIT_LEFT = -1
STEER_LIMIT_RIGHT = 1
THROTTLE_MAX = 0.251
THROTTLE_MIN = 0.25
MAX_STEERING_DIFF = 0.3
STEP_LENGTH = 0.05

MAX_EPISODE_STEPS = 1000

COMMAND_HISTORY_LENGTH = 5 
FRAME_STACK = 1
VAE_OUTPUT = 20

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
        "hidden_size": 64,
        "encoder_update_frequency": 1,
        "pretrained_ae": "",
        #"pretrained_ae": "./trained_models/vae/vae_1.pth",
        "image_folder": "./data/sim_images/",
        "im_size": IMAGE_SIZE,
        "n_images": 10000,
        "epochs": 1000
        },
    "ae": {
        "framestack": FRAME_STACK,
        "output": VAE_OUTPUT,
        "linear_input": 100,
        "image_size": IMAGE_SIZE,
        "lr": LR_START / 10,
        "image_channels": 3 if RGB else 1,
        "encoder_type": "vae",
        "batch_size": 64,
        "l2_regularization": True
    },
    "pretrained_ae":{
    }

}

