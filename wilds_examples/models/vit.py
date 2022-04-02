from pyexpat import model
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_pretrained_vit
from pytorch_pretrained_vit.configs import PRETRAINED_MODELS
from transformers import ViTFeatureExtractor, ViTForImageClassification

from transformers import ViTForImageClassification

class Identity(torch.nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class ViT(nn.Module):
    def __init__(self, model_size='B_16', pretrained=True, featurize=False, d_out=62):
        super().__init__()

        print('Model size: ', model_size)
        print('Pretrained: ', pretrained)

        self.network = pytorch_pretrained_vit.ViT(
            model_size, pretrained=pretrained,
        )

        if featurize:
            self.network.fc = Identity()
            self.d_out = PRETRAINED_MODELS[model_size]['config']['representation_size']


    def forward(self, x):
        x = self.network(x)
        return x
