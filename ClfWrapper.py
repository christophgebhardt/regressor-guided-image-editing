import sys
import torch

# setting path
sys.path.append('../emotion-adaptation')
from losses.ValenceArousalLoss import ValenceArousalLoss
from losses.CompoundEmotionLoss import CompoundEmotionLoss


class ClfWrapper:
    """
    Wrapper class for off-the-shelf classifier
    """

    def __init__(self, model_name: str, device: str,
                 model_directory: str = "/home/cgebhard/emotion-adaptation/trained_models"):
        """
        Constructor
        :param model_name: name of model
        :param device: string specifying the cuda device
        :param model_directory: directory where models are saved
        """
        self.model_name = model_name
        self.model_directory = model_directory

        if model_name == "EmoNet_valence_moments_resnet50_5_best.pth.tar":
            self.clf = ValenceArousalLoss(f"{model_directory}/{model_name}", device, 1.0, loss="valence")
        elif model_name == "va_pred_all":
            self.clf = ValenceArousalLoss(f"{model_directory}/{model_name}", device, 1.0)
        elif model_name == "emo_pred_ldl":
            self.clf = CompoundEmotionLoss(f"{model_directory}/{model_name}", device, 1.0, loss="intensity")

    def get_label(self, image: torch.Tensor) -> torch.Tensor:
        """
        forward propagation of loss
        :param image: image tensor
        :return: labels as Tensor
        """
        with torch.no_grad():
            labels = self.clf.predict_loss_metric(image)
            return labels
        #     if self.model_name == "EmoNet_valence_moments_resnet50_5_best.pth.tar":
        #         return labels
        #     elif self.model_name == "va_pred_all":
        #         return labels
        #     elif self.model_name == "emo_pred_ldl":
        #         return labels
        #
        # raise ValueError
