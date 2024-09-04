import h5py
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Callable, Union, Any, TypeVar, Tuple
Tensor = TypeVar('torch.tensor')


class CA(nn.Module):
    def __init__(self, inc):
        super(CA, self).__init__()
        self.sq = nn.AdaptiveAvgPool2d(1)
        self.ex = nn.Sequential(
            nn.Conv2d(inc, inc, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.sq(x)
        y = self.ex(y)
        return x * y


class CAR(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CAR, self).__init__()
        self.conv = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        self.ca = CA(out_ch)

    def forward(self, x):
        y = self.conv(x)
        y = self.ca(y)
        y = y + x
        return y

class Model_2(nn.Module):
    def __init__(self):
        super(Model_2, self).__init__()

        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv1 = nn.Conv2d(256, 256, kernel_size=2, stride=1, padding=1)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=2, stride=1, padding=1)
        self.conv3 = nn.Conv2d(512, 768, kernel_size=2, stride=1, padding=1)
        self.conv4 = nn.Conv2d(768, 512, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=0)

        ca1 = [
            CAR(512,512) for _ in range(1)
        ]

        self.ca_1 = nn.Sequential(*ca1)

        ca2 = [
            CAR(768,768) for _ in range(2)
        ]

        self.ca_2 = nn.Sequential(*ca2)


        ca3 = [
            CAR(512,512) for _ in range(2)
        ]

        self.ca_3 = nn.Sequential(*ca3)

        self.backbone = nn.Sequential(
            self.conv1,


            self.relu,
            self.pool,

            self.conv2,
            self.ca_1,
            self.relu,
            self.pool,

            self.conv3,
            self.ca_2,
            self.relu,
            self.pool,

        self.conv4,
        self.ca_3,
        self.relu,
        self.conv5,
        self.relu,

        )

        self.Flatten = nn.Flatten()

        # self.lstm = nn.LSTM(input_size=256, hidden_size=256, batch_first=True)
        # self.fc_1 = nn.Linear(512 * 9 * 5, 2)

    def forward(self, x, data_format='channels_last'):
        x = self.backbone(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        # x, _ = self.lstm(x)

        # x = self.Flatten(x)
        #
        # out = self.fc_1(x)

        return x

class ModelVAE(nn.Module):
    def __init__(self,
                 in_channels: int = 2,
                 latent_dim: int = 512,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(ModelVAE, self).__init__()

        self.latent_dim = latent_dim

        self.encoder = Model_2()
        self.fc_mu = nn.Linear(256 * 9 * 5, latent_dim)
        # self.fc_var = nn.Linear(512 * 9 * 5, latent_dim)


        # # Build Decoder
        # modules = []
        #
        self.decoder_input = nn.Linear(latent_dim, 256 * 9 * 5)
        # self.de_lstm = nn.LSTM(input_size=512, hidden_size=768, batch_first=True)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=0)
        self.de_backbone = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            self.relu,


            nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2),
            self.relu,
            self.pool,

            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            self.relu,
            self.pool,

            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            self.relu,
            self.pool,
        )
        self.final_layer = nn.Sequential(
                            nn.Conv2d(256, out_channels= 256,
                                      kernel_size= 3, padding= 1)
                            )

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder.backbone(input)
        # result = result.reshape(result.shape[0], result.shape[1], -1)
        # result = result.permute(0, 2, 1)
        # result, _ = self.encoder.lstm(result)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        # log_var = self.fc_var(result)
        mu=self.relu(mu)
        return mu

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = self.decoder_input(z)



        result = result.view(-1, 45, 256)
        # result, _ = self.de_lstm(result)
        result = result.permute(0, 2, 1)
        result = result.reshape(result.shape[0], result.shape[1], 9, 5)
        result = self.de_backbone(result)
        result = self.final_layer(result)
        return result

    # def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
    #     """
    #     Reparameterization trick to sample from N(mu, var) from
    #     N(0,1).
    #     :param mu: (Tensor) Mean of the latent Gaussian [B x D]
    #     :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
    #     :return: (Tensor) [B x D]
    #     """
    #     std = torch.exp(0.5 * logvar)
    #     eps = torch.randn_like(std)
    #     return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu = self.encode(input)
        # z = self.reparameterize(mu, log_var)
        output=self.decode(mu)
        # return  [self.decode(z), input, mu, log_var]
        return  output
