import torch
import math
from torch import Tensor
from baselines.losses.CompoundEmotionVector import emotions_type, CompoundEmotionVector
from baselines.losses.EmotionImageLoss import EmotionImageLoss
from baselines.models.EmotionPredictionModel import load_model_eval


class CompoundEmotionLoss(EmotionImageLoss):
    """
    Base class for losses that are computed based on the predictions of the module described in "A Circular-Structured
    Representation for Visual Emotion Distribution Learning".
    """

    def __init__(self, path_to_model: str, device: torch.device, weight: float,
                 is_minimized: bool = True, loss: str = "compound", is_input_range_0_1: bool = True,
                 activation_function: object = torch.nn.Softmax(dim=1), input_size=480, crop_size=448):
        """
        Constructor
        :param path_to_model: path to the model which is used to predict emotions
        :param device: device on which loss is computed (cpu or cuda:x)
        :param weight: factor with which the calculated loss is multiplied
        :param is_minimized: flag which indicates if loss is minimized or maximized (default=True)
        :param loss: specifies type of loss (default = 'compound'). Options are 'intensity', 'theta', 'compound'.
        :param activation_function: Activation function to use on model outputs (default = Softmax)
        :param is_input_range_0_1: flag that specifies if input range of images is 0 to 1 (default = True).
        :param input_size: Size images of the dataset get resized to (default = None)
        :param crop_size: Size of center crop performed with images (default = None)
        """
        super(CompoundEmotionLoss, self).__init__(device, weight, is_minimized)
        self.model = load_model_eval(
            path_to_model, 8, normalize=is_input_range_0_1, activation_function=activation_function,
            input_size=input_size, crop_size=crop_size, is_ten_crop=False).to(device)
        self.emotions_type = emotions_type.to(device)
        self.loss = loss
        if loss == "intensity":
            self.get_error = self.get_error_intensity
        elif loss == "theta":
            self.get_error = self.get_error_theta
        else:
            self.get_error = self.get_error_compound

    def predict_compound_vectors_from_imgs(self, imgs: Tensor) -> CompoundEmotionVector:
        """
        Initializes the instances of compound emotion vectors from generated images and target emotions
        :param imgs: tensor with generated images
        :return: instance of CompoundEmotionVector for predictions
        """
        with torch.no_grad():
            return CompoundEmotionVector.init_compound_emotions_vector(self.model(imgs), self.emotions_type)

    def forward(self, fake_imgs: Tensor, real_imgs: Tensor = None, condition: Tensor = None) -> Tensor:
        """
        forward propagation of loss
        :param fake_imgs: tensor with generated images
        :param real_imgs: predicted intensity, theta or both of real images.
        :param condition: values with which the respective generated image was conditioned.
        If None, unconditioned version of loss is computed.
        :return: tensor with loss
        """
        predictions_fake = self.predict_compound_vectors_from_imgs(fake_imgs)
        predictions_real = self.predict_compound_vectors_from_imgs(real_imgs) if real_imgs is not None else None
        return torch.mean(self.weight * self.get_error(predictions_fake, predictions_real, condition))

    def get_error_intensity(self, predictions, source, condition):
        """
        Returns the intensity error
        :param predictions: instance of CompoundEmotionVector for predictions
        :param source: instance of CompoundEmotionVector for real images.
        :param condition: values with which the respective generated image was conditioned.
        If None, unconditioned version of loss is computed.
        :return:
        """
        self.fake_loss_metric = predictions.intensity
        if source is not None:
            self.real_loss_metric = source.intensity

        if condition is None:
            # low intensity: 0, high intensity: 1
            condition = 1 if self.is_minimized else 0

        target_intensity = torch.ones(predictions.theta.size(0)).to(self.device) * (1 - condition)

        error = target_intensity - predictions.intensity
        return error * error

    def get_error_theta(self, predictions, source, condition):
        """
        Returns the theta error
        :param predictions: instance of CompoundEmotionVector for predictions
        :param source: instance of CompoundEmotionVector for real images.
        :param condition: values with which the respective generated image was conditioned.
        If None, unconditioned version of loss is computed.
        :return:
        """
        self.fake_loss_metric = predictions.theta
        if source is not None:
            self.real_loss_metric = source.theta

        if condition is None:
            # negative theta: 2 * math.pi, positive theta: math.pi
            condition = 0 if self.is_minimized else 1

        target_theta = math.pi * torch.ones(predictions.theta.size(0)).to(self.device) * (2 - condition)

        error = target_theta - predictions.theta
        return error * error

    def get_error_compound(self, predictions, source, condition):
        """
        Returns the intensity and theta error
        :param predictions: instance of CompoundEmotionVector for predictions
        :param source: instance of CompoundEmotionVector for real images.
        :param condition: values with which the respective generated image was conditioned.
        If None, unconditioned version of loss is computed.
        :return:
        """
        error_int = self.get_error_intensity(predictions, source, condition)
        error_theta = self.get_error_theta(predictions, source, condition)
        # order is important as loss metrics are assigned in other functions too!
        self.fake_loss_metric = torch.cat(
            (predictions.theta.reshape(4, 1), predictions.intensity.reshape(4, 1)), dim=1)
        if source is not None:
            self.real_loss_metric = torch.cat(
                (source.theta.reshape(4, 1), source.intensity.reshape(4, 1)), dim=1)
        return error_int + error_theta

    def predict_loss_metric(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Predicts the metric based on which the loss is computed for a batch of images
        :param imgs: tensor with images
        :return: tensor returning the metric
        """
        predictions = self.predict_compound_vectors_from_imgs(imgs)
        if self.loss == "intensity":
            return predictions.intensity
        elif self.loss == "theta":
            return predictions.theta
        else:
            return torch.cat((predictions.theta.reshape(imgs.size(0), 1),
                              predictions.intensity.reshape(imgs.size(0), 1)), dim=1)

    def get_random_condition_tensor(self, batch_size):
        """
        Returns a random target tensor for the specific loss
        :param batch_size: batch size of tensor to generate
        :return: random target tensor
        """
        if self.loss == "compound":
            random_condition_vals = torch.randint(0, 2, (2 * batch_size,)).reshape(batch_size, 2)
        else:
            random_condition_vals = torch.randint(0, 2, (batch_size,))

        return random_condition_vals.to(self.device)

