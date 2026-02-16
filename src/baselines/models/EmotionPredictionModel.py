import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms

from models.utilities.ReplicateAndCrop import ReplicateAndCrop
from models.utilities.MeanReplicatedCrops import MeanReplicatedCrops


def load_model_eval(path_to_model, num_classes, input_size=None, crop_size=None, normalize=False,
                    activation_function=None, is_ten_crop=False, requires_grad=False):
    """
    Initialize resnet 50 model with num_classes output layer as used in PCL paper
    :param path_to_model: path to the model which is used to predict emotions
    :param num_classes: number of neurons in output layer
    :param input_size: Size images of the dataset get resized to (default = None)
    :param crop_size: Size of center crop performed with images (default = None)
    :param normalize: Normalize to -1 and 1 (default = False)
    :param activation_function: Activation function to use on model outputs (default = None)
    :param is_ten_crop: if true each image in batch gets replicated ten times and cropped randomly (default = False)
    :param requires_grad: if true model parameters are set to require a gradient (default = False)
    :return:
    """
    model = models.resnet50()
    num_ft = model.fc.in_features
    model.fc = nn.Linear(num_ft, num_classes)
    model.load_state_dict(torch.load(path_to_model))
    if not requires_grad:
        if hasattr(model, 'parameters'):
            for p in model.parameters():
                p.requires_grad = False
    model.eval()

    modules = []
    num_replications = 10
    if input_size is not None:
        modules.append(transforms.Resize(input_size, antialias=True))
    if crop_size is not None:
        if not is_ten_crop:
            modules.append(transforms.CenterCrop(crop_size))
        else:
            modules.append(ReplicateAndCrop(crop_size, normalize, num_replications))
    if normalize and not is_ten_crop:
        modules.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    modules.append(model)

    if is_ten_crop:
        modules.append(MeanReplicatedCrops(num_replications))

    if activation_function is not None:
        modules.append(activation_function)

    return nn.Sequential(*modules)


def initialize_model(num_classes=8, feature_extract=False):
    """
    Initialize resnet 50 model with num_classes output layer as used in PCL paper
    :param num_classes: number of neurons in output layer (default = 8)
    :param feature_extract: Flag for feature extracting. When False, we finetune the whole model, when True we only
    update the reshaped layer params (default = False)
    :return:
    """
    # model_ft = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model_ft = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    _set_parameter_requires_grad(model_ft, feature_extract)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    params_to_update = _params_to_update(model_ft, feature_extract)

    return model_ft, params_to_update


def _set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False


def _params_to_update(model_ft, feature_extract):
    """
    Gather the parameters to be optimized/updated in this run. If we are fine-tuning we will be updating all parameters.
    However, if we are doing feature extract method, we will only update the parameters that we have just initialized,
    i.e. the parameters with requires_grad is True.
    :return: list with parameters to update
    """

    params_to_update = model_ft.parameters()
    # print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
                # print("\t", name)
    # else:
    #     for name, param in model_ft.named_parameters():
    #         if param.requires_grad == True:
                # print("\t", name)

    return params_to_update


def get_emo_pred_transform(input_size=480, crop_size=448):
    """
    Get the transform for the emotion prediction resnet 50 model
    :param input_size: Size images of the dataset get resized to (default = 480, from paper)
    :param crop_size: Size of center crop performed with images (default = 448, from paper)
    """
    return transforms.Compose([
        # transforms.RandomResizedCrop(input_size, antialias=True),
        transforms.Resize(input_size, antialias=True),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def get_emo_pred_random_transform(input_size=480, crop_size=448):
    """
    Get the transform for the emotion prediction resnet 50 model, utilizing random transformations
    :param input_size: Size images of the dataset get resized to (default = 480, from paper)
    :param crop_size: Size of center crop performed with images (default = 448, from paper)
    """
    return transforms.Compose([
        # transforms.RandomResizedCrop(input_size, antialias=True),
        transforms.Resize(input_size, antialias=True),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
