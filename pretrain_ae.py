import argparse
import datetime
import os
import random

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from models.vae import VAE
from models.ae import AE 

from config import PARAMS, IMAGE_SIZE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

ae = AE(PARAMS["ae"])

def process_image(im):
    return np.rollaxis(cv2.resize(im, (IMAGE_SIZE, IMAGE_SIZE)), 2, 0)
    
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


test_images = images[:args.test_size]

batch_size = PARAMS["ae"]["batch_size"]

train_loader = torch.utils.data.DataLoader([torch.FloatTensor(process_image(i)).to(device) for i in images], shuffle=True, batch_size=batch_size)
#test_loader = torch.utils.data.DataLoader([torch.Tensor(i).to(device) for i in test_images], shuffle=True, batch_size=batch_size)

best_loss = float("inf")
best_epoch = 0

for i in range(args.epochs):
    cum_loss = 0
    for b, inputs in enumerate(train_loader):
        ae.optimizer.zero_grad()

        loss = ae.loss(inputs)
        loss.backward()

        ae.optimizer.step()

        cum_loss += loss.item()

    
    print("Epoch: {}, Encoder loss: {}".format(i, cum_loss))

    #print("Training episode {}/{}".format(i, args.epochs))
    #print("Training loss: {:.2f}, Reconstruction loss: {:.2f}, Test loss: {}".format(train_loss, recon_loss, test_loss))
    #print("Best test loss {:.2f} at epoch {}".format(best_loss, best_epoch))


print("Saving model to: {}".format(args.model_file))

if args.train:
    torch.save(ae, args.model_file)

if args.visualize_results:

    plt.figure(1, (8, 20))
    for i, im in enumerate(random.sample(test_images, min(len(test_images), 5))):
        plt.subplot(len(test_images), 2, i * 2 + 1)
        plt.imshow(im)
        
        plt.subplot(len(test_images), 2, i * 2 + 2)
        embedding = ae.embed(im)
        reconstruction = vae.decode(embedding[np.newaxis, np.newaxis, :]).reshape(40, 40, 3)
        
        plt.imshow(reconstruction)
        
    plt.savefig("vae_results.png")
    plt.show()




