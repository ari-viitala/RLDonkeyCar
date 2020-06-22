import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2

import random

from .modules import Encoder, Decoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AE:
    
    def __init__(self, parameters = {}):

        params = {
            "framestack": 2,
            "output": 32,
            "linear_input": 500,
            "image_size": 40,
            "lr": 0.001,
            "image_channels": 3,
            "encoder_type": "vae"
        }

        for p in parameters.keys():
            params[p] = parameters[p]

        self.framestack = params["framestack"]
        self.image_size = params["image_size"]
        self.linear_output = params["output"]
        self.linear_input = params["linear_input"]
        self.lr = params["lr"]
        self.image_channels = params["image_channels"]
        self.type = params["encoder_type"]
        self.l2_regularization = params["l2_regularization"]
     
        self.encoder = Encoder(self.image_size, self.linear_input, self.linear_output, self.image_channels * self.framestack).to(device)
        self.decoder = Decoder(self.image_size, self.linear_input, self.linear_output, self.image_channels * self.framestack).to(device)

        self.encoder_target = Encoder(self.image_size, self.linear_input, self.linear_output, self.image_channels * self.framestack).to(device)
        
        for target_param, param in zip(self.encoder_target.parameters(), self.encoder.parameters()):
            target_param.data.copy_(param.data)

        parameters = list(self.encoder.parameters()) + list(self.decoder.parameters())
        
        self.optimizer = torch.optim.Adam(parameters, lr=self.lr)
        
        self.recon_func = nn.BCELoss()
        self.recon_func.size_average = False

        self.criterion = nn.KLDivLoss()
        self.recon_loss = nn.MSELoss()

        self.tau = 0.005

   
    def calculate_vae_loss(self, true, pred, mu, log_sigma):
        #print(true.shape)
        #print(pred.shape)
        recon = F.binary_cross_entropy(pred, true, size_average=False)

        KLD = -0.5 * torch.mean(1 + log_sigma - mu.pow(2) - log_sigma.exp())

        return recon, 3*KLD
    
    def sample_z(self, mu, log_sigma):
        std = log_sigma.mul(0.5).exp_()
        eps = torch.randn(*mu.size()).to(device)
        return mu + std * eps


    def loss(self, ims):
        if self.type == "vae":
            return self.vae_loss(ims)
        elif self.type == "ae":
            return self.ae_loss(ims)


    def vae_loss(self, ims):

        mu, log_sigma = self.encoder(ims)
  
        target = ims.clone() 
        pred = self.decoder(self.sample_z(mu, log_sigma))
        
        recon, kl = self.calculate_vae_loss(target, pred, mu, log_sigma)
        loss = recon + kl
            
        return loss

    def ae_loss(self, ims):

        encoder_output, _ = self.encoder(ims)
        reconstruction = self.decoder(encoder_output)

        reconstruction_loss = self.recon_loss(ims, reconstruction)

        loss = reconstruction_loss
        
        #print(encoder_output.sum(0).shape)

        if self.l2_regularization:
            l2_loss = 0.5 * encoder_output.pow(2).sum(1).mean()
            loss += l2_loss
        
        return loss

    def process_image(self, im):

        if self.image_channels == 1:
            gs_im = np.dot(im[...,:3], [0.299, 0.587, 0.114]) / 255
            return cv2.resize(gs_im, (self.image_size, self.image_size))[np.newaxis, np.newaxis, :]
        else:
            return im.reshape(1, 3 * self.framestack, self.image_size, self.image_size)
            
            
    def embed(self, image):
        
        im = torch.Tensor(self.process_image(image)).to(device)
        mu, log_sigma = self.encoder.forward(im)
        return mu
        #return mu.detach().cpu().numpy().squeeze()
    
    def decode(self, embedding):
        return self.decoder(torch.FloatTensor(embedding).to(device)).detach().cpu().numpy()

    def update_encoder_target(self):
        for target_param, param in zip(self.encoder_target.parameters(), self.encoder.parameters()):
            target_param.data.copy_((1.0-self.tau)*target_param.data + self.tau*param.data)


        


