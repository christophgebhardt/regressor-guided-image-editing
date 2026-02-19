import torch
import torch.nn as nn
from torch import Tensor


class GuidanceClassifier(nn.Module):
    """
    Base class for guidance classifier
    """

    def __init__(self, device: str):
        """
        Constructor
        :param device: device on which loss is computed (cpu or cuda:x)
        """
        super(GuidanceClassifier, self).__init__()
        self.device = torch.device(device)
        self.model = None

    def forward(self, latents: Tensor, t: float, prompt_embeds: Tensor = None) -> Tensor:
        """
        forward propagation of loss
        :param latents: tensor with noisy latents
        :param t: timestep
        :param prompt_embeds: text prompt embeddings
        :return: tensor with loss
        """
        pass

    def get_loss(self, latents: Tensor, label: Tensor, t: float, prompts: list) -> (Tensor, Tensor):
        """
        Predicts the metric based on which the loss is computed for a batch of images
        :param latents: tensor with noisy latents
        :param label: tensor with labels
        :param t: timestep
        :param prompts: list with positive and negative prompts
        :return:
        """
        pass

    def predict_score(self, latents: Tensor, t: Tensor, prompts: list) -> Tensor:
        """
        Predicts the metric based on which the loss is computed for a batch of images
        :param latents: tensor with noisy latents
        :param t: timestep
        :param prompts: list with positive and negative prompts
        :return:
        """
        pass
