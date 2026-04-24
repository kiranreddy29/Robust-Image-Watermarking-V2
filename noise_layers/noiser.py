import numpy as np
import torch.nn as nn
import random

class Noiser(nn.Module):
    def __init__(self, noise_layers: list, composed=False):
        super(Noiser, self).__init__()
        self.noise_layers = nn.ModuleList(noise_layers)
        self.composed = composed

    def forward(self, encoded_and_cover):
        if not self.noise_layers:
            return encoded_and_cover
            
        if self.composed:
            num_attacks = random.randint(1, min(3, len(self.noise_layers)))
            layers_to_apply = random.sample(list(self.noise_layers), num_attacks)
            
            output = encoded_and_cover
            for layer in layers_to_apply:
                output = layer(output)
            return output
        else:
            layer = random.choice(self.noise_layers)
            return layer(encoded_and_cover)