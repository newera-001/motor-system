import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self,params_size,z_size):
        super(Encoder, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(params_size, 512),
            nn.ReLU(),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128, z_size),
        )

    def forward(self,params):
        return self.fc(params)

    def optimizer(self, lr=1e-3, parameters = None):
        return torch.optim.Adam(lr=lr,params=parameters)
