import numpy as np

from dip_models import *
from dip_utils.denoising_utils import *

import torch
import torch.optim

class defense_model():

    def __init__(self, img):
        self.loss = torch.nn.MSELoss().type(torch.cuda.FloatTensor)
#        self.net = get_net(3, 'skip', 'reflection',
#                skip_n33d=128, 
#                skip_n33u=128, 
#                skip_n11=4, 
#                num_scales=5,
#                upsample_mode='bilinear').type(torch.cuda.FloatTensor)
        self.net = skip(
                3, 3, 
                num_channels_down = [8, 16, 32, 64], 
                num_channels_up   = [8, 16, 32, 64],
                num_channels_skip = [0, 0, 0, 4], 
                upsample_mode='bilinear',
                need_sigmoid=True, need_bias=True, pad='reflection', act_fun='LeakyReLU')

        self.net = self.net.type(torch.cuda.FloatTensor)

        self.mse = torch.nn.MSELoss().type(torch.cuda.FloatTensor)
        
        self.img_noisy_torch = torch.unsqueeze(img, 0).type(torch.cuda.FloatTensor)
        
        self.reg_noise_std = 1./30
        self.sigma = 25/255.
        print(img.shape[1], img.shape[0])
        self.net_input = get_noise(3, 'noise', (32,32)).type(torch.cuda.FloatTensor).detach()
        self.net_input_saved = self.net_input.detach().clone()
        self.noise = self.net_input.detach().clone()
        self.exp_weight=0.99
        print(self.net_input.size())
        
        self.out_avg = None
        self.optimizer = torch.optim.Adam(get_params('net', self.net, self.net_input), lr=0.01)


    def forward(self):
      self.optimizer.zero_grad()
      net_input = self.net_input_saved + (self.noise.normal_() * self.reg_noise_std)
      
      out = self.net(net_input)
      
      # Smoothing
      if self.out_avg is None:
         self.out_avg = out.detach()
      else:
          self.out_avg = self.out_avg * self.exp_weight + out.detach() * (1 - self.exp_weight)

      total_loss = self.mse(out, self.img_noisy_torch)
      total_loss.backward()
      self.optimizer.step()
      return out.type(torch.FloatTensor)