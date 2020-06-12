import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, init_w=3e-3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        self.net[-1].weight.data.uniform_(-init_w, init_w)
        self.net[-1].bias.data.uniform_(-init_w, init_w)

    def forward(self, x):
        return self.net(x)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
class UnFlatten(nn.Module):
    def __init__(self, channels, size):
        super().__init__()
        self.channels = channels
        self.size = size
    
    def forward(self, x):
        return x.view(x.size(0), self.channels, self.size, self.size)


class Encoder(nn.Module):
    def __init__(self, image_size, linear_input, linear_output, image_channel, image_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=image_channels, out_channels=16, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            Flatten())
        self.mu = nn.Sequential(
            nn.Linear(int(image_size**2/4)*32, linear_input),
            nn.ReLU(),
            nn.Linear(linear_input, linear_output)
        )
        self.ls = nn.Sequential(
            nn.Linear(int(image_size**2/4)*32, linear_input),
            nn.ReLU(),
            nn.Linear(linear_input, linear_output)
        )

    def forward(self, x):
        x = self.net(x)
        mu = self.mu(x)
        log_sigma = self.ls(x)

        return mu, log_sigma

class Decoder(nn.Module):
    def __init__(self, image_size, linear_input, linear_output, image_channels=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(linear_output, linear_input),
            nn.ReLU(),
            nn.Linear(linear_input, 32*int(image_size**2/4)),
            UnFlatten(32, int(image_size/2)),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=image_channels, kernel_size=3, padding=1, stride=2, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


