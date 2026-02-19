import torch.nn as nn


class TransformModule(nn.Module):
    """
    Torch module to wrap transforms.Compose into a module.
    """
    def __init__(self, transform):
        super(TransformModule, self).__init__()
        self.transform = transform

    def forward(self, x):
        return self.transform(x)