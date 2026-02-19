import torch
from torch import Tensor
from baselines.losses.EmotionImageLoss import EmotionImageLoss
from baselines.models.EmotionPredictionModel import load_model_eval
from baselines.models.EmoNet import load_model_eval as load_model_eval_emo_net


class ValenceArousalLoss(EmotionImageLoss):
    """
    intensity image loss computed on the intensity property of the CompoundEmotionVector.
    Sets intensity of images to be zero, meaning they should not cause emotions.
    """

    def __init__(self, path_to_model: str, device: torch.device, weight: float, is_minimized: bool = True,
                 loss: str = "va", is_input_range_0_1=True, input_size=480, crop_size=448, requires_grad=False):
        """
        Constructor
        :param path_to_model: path to the model which is used to predict emotions
        :param device: device on which loss is computed (cpu or cuda:x)
        :param weight: factor with which the calculated loss is multiplied
        :param is_minimized: flag which indicates if loss is minimized or maximized (default=True)
        :param loss: specifies type of loss (default = 'va'). Options are 'va', 'valence', 'arousal'.
        :param is_input_range_0_1: flag that specifies if input range of images is 0 to 1 (default = True).
        If False, input range is assumed to be -1 to 1.
        :param input_size: Size images of the dataset get resized to (default = None)
        :param crop_size: Size of center crop performed with images (default = None)
        :param requires_grad: if true model parameters are set to require a gradient (default = False)
        """
        super(ValenceArousalLoss, self).__init__(device, weight, is_minimized)

        if "EmoNet" in path_to_model:
            self.model = load_model_eval_emo_net(
                path_to_model, normalize=is_input_range_0_1, requires_grad=requires_grad).to(device)
        else:
            num_classes = 4
            activ_func = torch.nn.Sigmoid()
            if "no_sigmoid" in path_to_model:
                activ_func = None
            if "mse" in path_to_model:
                num_classes = 2
                activ_func = None
            if "arousal_nll" in path_to_model:
                num_classes = 2

            self.model = load_model_eval(
                path_to_model, num_classes, normalize=is_input_range_0_1, activation_function=activ_func,
                input_size=input_size, crop_size=crop_size, is_ten_crop=True, requires_grad=requires_grad).to(device)

        if loss == "valence":
            self.get_error = self.get_valence_error
            self.output_ixs = [0]  # access valence mean 0, arousal mean 1, valence std 2 and arousal std 3
        elif loss == "arousal":
            self.get_error = self.get_arousal_error
            self.output_ixs = [1]
        else:
            self.get_error = self.get_valence_arousal_error
            self.output_ixs = [0, 1]

    def forward(self, fake_imgs: Tensor, real_imgs: Tensor = None, target: Tensor = None) -> Tensor:
        """
        forward propagation of loss
        :param fake_imgs: tensor with generated images
        :param real_imgs: tensor with real images
        :param target: target value of valence and our arousal the image should elicit.
        If None, untargeted version of loss is computed.
        :return: tensor with loss
        """
        # with torch.no_grad():
        self.fake_loss_metric = self.model(fake_imgs)[:, self.output_ixs]
        if real_imgs is not None:
            self.real_loss_metric = self.model(real_imgs)[:, self.output_ixs]
        loss = torch.mean(self.weight * self.get_error(self.fake_loss_metric, target))
        return loss

    def get_valence_error(self, predicted, target):
        """
        Returns the error for valence
        :param predicted: predicted valence
        :param target: target valence the image should elicit. If None, untargeted version of loss is computed.
        :return: error
        """
        if target is None:
            if self.is_minimized:
                # low valence:
                # sets the target for predictions over 0.5 to 0 and for under 0.5 to 1
                # target = (predicted < 0.5).int() * torch.ones(predicted.size(0)).to(self.device)
                target = 0.5 * torch.ones(predicted.size(0)).to(self.device)
                # target = torch.zeros(predicted.size(0)).to(self.device)
            else:
                # high valence:
                target = torch.ones(predicted.size(0)).to(self.device)

        error = target - predicted
        return error * error

    def get_arousal_error(self, predicted, target):
        """
        Returns the error for arousal
        :param predicted: predicted arousal
        :param target: target arousal the image should elicit. If None, untargeted version of loss is computed.
        :return: error
        """
        if target is None:
            if self.is_minimized:
                # low arousal:
                target = torch.zeros(predicted.size(0)).to(self.device)
            else:
                # high arousal:
                target = torch.ones(predicted.size(0)).to(self.device)

        error = target - predicted
        return error * error

    def get_valence_arousal_error(self, predicted, target):
        """
        Returns the error for valence and arousal
        :param predicted: predicted valence and arousal
        :param target: target value of valence and our arousal the image should elicit.
        If None, untargeted version of loss is computed.
        :return:
        """
        if target is not None:
            val_error = self.get_valence_error(predicted[:, 0], target[:, 0])
            ar_error = self.get_arousal_error(predicted[:, 1], target[:, 1])
        else:
            val_error = self.get_valence_error(predicted[:, 0], None)
            ar_error = self.get_arousal_error(predicted[:, 1], None)

        return val_error + ar_error

    def predict_loss_metric(self, imgs: torch.Tensor) -> torch.Tensor:
        """
        Predicts the metric based on which the loss is computed for a batch of images
        :param imgs: tensor with images
        :return: tensor returning the metic
        """
        with torch.no_grad():
            return self.model(imgs)[:, self.output_ixs]

    def get_random_condition_tensor(self, batch_size):
        """
        Returns a random target tensor for the specific loss
        :param batch_size: batch size of tensor to generate
        :return: random target tensor
        """
        dim_space = len(self.output_ixs)
        return torch.randint(0, 2, (batch_size * dim_space,)).reshape(batch_size, dim_space).to(self.device)

    # def get_valence_or_arousal_error(self, predicted, source, condition):
    #     """
    #     Returns the error for valence or arousal
    #     :param predicted: predicted valence or arousal
    #     :param source: predicted valence or arousal of real images.
    #     :param condition: values with which the respective generated image was conditioned.
    #     If None, unconditioned version of loss is computed.
    #     :return: error
    #     """
    #     error = predicted - source
    #     if condition is None:
    #         error = error if self.is_minimized else -1 * error
    #     else:
    #         error = (1 - 2 * condition) * error
    #     return error
