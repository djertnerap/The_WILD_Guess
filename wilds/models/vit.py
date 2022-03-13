import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTFeatureExtractor, ViTForImageClassification
from pytorch_pretrained_vit import ViT

from transformers import ViTForImageClassification

class Identity(torch.nn.Module):
    """An identity layer"""
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class ViT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.network = ViT(
            'B_16', pretrained=True
        )
        self.n_outputs = num_classes

    def forward(self, x):
        out = self.network(x)
        return out


# class ViT(nn.Module):
#     def __init__(self,num_classes=1024):
#         super().__init__()

#         self.transformer_feature_extractor = ViTFeatureExtractor.from_pretrained('facebook/deit-base-patch16-224')
#         network = ViTForImageClassification.from_pretrained(
#             'facebook/deit-base-patch16-224', 
#             num_labels=num_classes,
#         )

#         self.network = network
    
#     def forward(self, xb):
#         print(xb.shape)
#         inputs = self.transformer_feature_extractor(xb.cpu(), return_tensors="pt")

#         print(inputs.shape)

#         outputs = self.network(**inputs)

#         print(outputs.shape)

#         return outputs

# if __name__ == "__main__":
#     vit = ViT(10)

#     print(vit.eval())