import torch
import torch.nn as nn
from torch import Tensor


class EmotionImageLoss(nn.Module):
    """
    Base class for additional generator losses
    """

    def __init__(self, device: torch.device, weight: float, is_minimized: bool = True):
        """
        Constructor
        :param device: device on which loss is computed (cpu or cuda:x)
        :param weight: factor with which the calculated loss is multiplied
        :param is_minimized: flag which indicates if loss is minimized or maximized (default=True)
        """
        super(EmotionImageLoss, self).__init__()
        self.model = None
        self.device = device
        self.is_minimized = is_minimized
        self.weight = weight
        self.fake_loss_metric = None
        self.real_loss_metric = None

    def forward(self, fake_imgs: Tensor, real_imgs: Tensor = None, condition: Tensor = None) -> Tensor:
        """
        forward propagation of loss
        :param fake_imgs: tensor with generated images
        :param real_imgs: tensor with real images
        :param condition: values with which the respective generated image was conditioned.
        If None, unconditioned version of loss is computed.
        :return: tensor with loss
        """
        pass

    def predict_loss_metric(self, imgs: Tensor) -> Tensor:
        """
        Predicts the metric based on which the loss is computed for a batch of images
        :param imgs: tensor with images
        :return: tensor returning the metic
        """
        return self.model(imgs)

    def get_random_condition_tensor(self, batch_size):
        """
        Returns a random condition tensor for the specific loss
        :param batch_size: batch size of tensor to generate
        :return: random condition tensor
        """
        pass

