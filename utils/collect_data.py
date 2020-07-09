"""
Script for collecting images to pretrain encoder
"""

import datetime
import time
import argparse
import os

from PIL import Image

from donkeycar.gym import remote_controller

parser = argparse.ArgumentParser()

parser.add_argument("--image_folder", help="Folder to save images")
parser.add_argument("--car_name", help="Name of the car on mqtt-server", default="Kari")
parser.add_argument("--images", help="Number of images", default=10000, type=int)
parser.add_argument("--step_length", help="Time between images in (s)", default=0.1)

args = parser.parse_args()

ts = datetime.datetime.now().isoformat()

data_path = "../data/"
folder = data_path + args.image_folder if args.image_folder else data_path + "run_{}/".format(ts)

if not os.path.isdir(folder):
    os.mkdir(folder)

controller = remote_controller.DonkeyRemoteContoller(donkey_name=args.car_name, mqtt_broker="mqtt.eclipse.org")

time.sleep(1)

for i in range(args.images):
    try:
        start_time = time.time_ns()
        filename = "{}_cam-image_array_.jpg".format(i + 1)

        array = controller.observe()
        image = Image.fromarray(array)
        image.save(folder + filename)

        time.sleep(args.step_length)
        print("Step: {}, Step time: {}ms".format(i + 1, (time.time_ns() - start_time)/1e6))
    except KeyboardInterrupt:
        continue
