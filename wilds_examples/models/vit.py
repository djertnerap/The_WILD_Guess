import torch
import torch.nn as nn
import pytorch_pretrained_vit
from pytorch_pretrained_vit.configs import PRETRAINED_MODELS

class Identity(torch.nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class ViT(nn.Module):
    """
    Visual Transformer Implementation
    Original Implementation: [Luke Melas-Kyriazi](https://github.com/lukemelas/PyTorch-Pretrained-ViT)
    Pre-trainted Weights: [Google, ImageNet-21k](https://github.com/google-research/vision_transformer)
    Adapted for the Wilds project.
    """
    def __init__(self, model_size='B_16', pretrained=True, featurize=False, d_out=62):
        super().__init__()
        self.network = pytorch_pretrained_vit.ViT(model_size, pretrained=pretrained)
        if featurize:
            self.network.fc = Identity()
            self.d_out = PRETRAINED_MODELS[model_size]['config']['representation_size']

    def forward(self, x):
        x = self.network(x)
        return x
