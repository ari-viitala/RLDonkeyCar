import argparse
import datetime
import os
import random

import torch
import numpy as np
import matplotlib.pyplot as plt

from models.vae import VAE



timestamp = datetime.datetime.today().isoformat()

parser = argparse.ArgumentParser()

parser.add_argument("--image_folder", help="Folder of the training images", default="data/images")
parser.add_argument("--existing_model", help="Load an existing model", default="")
parser.add_argument("--train", help="Train the model", default=1, type=int)
parser.add_argument("--type", help="Model type, eiher vae or ae", default="vae")
parser.add_argument("--model_file", help="File for saving the model", default="")
parser.add_argument("--epochs", help="VAE training episodes", default=100, type=int)
parser.add_argument("--n_images", help="Number of images to use", default=10000, type=int)
parser.add_argument("--test_size", help="Number of test images", default=5, type=int)

parser.add_argument("--visualize_results", help="Plot vae results in the end", default=True)

args = parser.parse_args()

if args.existing_model:
    print("Loading model from: {}".format(args.existing_model))
    vae = torch.load(args.existing_model)
else:
    vae = VAE(linear_input=1000, linear_output=32, batch_size=64, image_channels=3, lr=0.001, encoder_type=args.type)

if not args.model_file:
    args.model_file = "./trained_models/vae/{}_{}.pth".format(args.type, timestamp)

if not args.train:
    args.epochs = 0
    args.n_images = 5

files = [args.image_folder + x for x in os.listdir(args.image_folder) if "cam" in x]
files = random.sample(files, min(len(files), args.n_images))
images = []

print("Loading {} images".format(len(files)))

for i, file in enumerate(files):
    os.system('clear')
    print("Loading image {}/{}".format(i, len(files)))
    images.append(plt.imread(file, format="jpeg"))


random.shuffle(images)
test_images = images[:args.test_size]

for i in images[args.test_size:]:
    vae.add_image(i)

for i in test_images:
    vae.add_test_image(i / 255)

best_loss = float("inf")
best_epoch = 0

for i in range(args.epochs):
    train_loss, recon_loss, test_loss = vae.update()
    
    if best_loss > test_loss:
        best_loss = test_loss
        best_epoch = i

    os.system('clear')
    print("Training episode {}/{}".format(i, args.epochs))
    print("Training loss: {:.2f}, Reconstruction loss: {:.2f}, Test loss: {}".format(train_loss, recon_loss, test_loss))
    print("Best test loss {:.2f} at epoch {}".format(best_loss, best_epoch))


vae.images = []

print("Saving model to: {}".format(args.model_file))

if args.train:
    torch.save(vae, args.model_file)

if args.visualize_results:

    plt.figure(1, (8, 20))
    for i, im in enumerate(random.sample(test_images, min(len(test_images), 5))):
        plt.subplot(len(test_images), 2, i * 2 + 1)
        plt.imshow(im)
        
        plt.subplot(len(test_images), 2, i * 2 + 2)
        embedding = vae.embed(im)
        reconstruction = vae.decode(embedding[np.newaxis, np.newaxis, :]).reshape(40, 40, 3)
        
        plt.imshow(reconstruction)
        
    plt.savefig("vae_results.png")
    plt.show()




