import torchvision
import torch
import torch.nn as nn
import math
import warnings

from baselines.models.utilities.TransformModule import TransformModule


class EmoNet(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet50(pretrained=False)
        self.model.fc = torch.nn.Linear(self.model.fc.in_features, 1)

    def forward(self, x):
        return self.model(x)

    @property
    def mean(self):
        return [0.485, 0.456, 0.406]

    @property
    def std(self):
        return [0.229, 0.224, 0.225]

    @property
    def input_size(self):
        return [3, 224, 224]


def load_model_eval(path_to_model, normalize=False, requires_grad=False):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model, input_transform, output_transform = emonet(True, path_to_model, normalize)

    if not requires_grad:
        if hasattr(model, 'parameters'):
            for p in model.parameters():
                p.requires_grad = False

    model.eval()
    return nn.Sequential(TransformModule(input_transform), model, TransformModule(output_transform))


def emonet(is_tencrop, path="./assessors/EmoNet_valence_moments_resnet50_5_best.pth.tar", normalize=False):
    model = EmoNet()
    parameters = torch.load(path, map_location='cpu')
    state_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(parameters['state_dict'].items())}
    state_dict["model.fc.weight"] = state_dict.pop("model.last_linear.weight")
    state_dict["model.fc.bias"] = state_dict.pop("model.last_linear.bias")
    model.load_state_dict(state_dict)

    if is_tencrop:
        input_transform = tencrop_image_transform(model, normalize)
        output_transform = tencrop_output_transform_emonet
    else:
        input_transform = image_transform(model)
        output_transform = lambda x: x

    return model, input_transform, output_transform


def tencrop_image_transform(model, is_normalize=False):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    normalize = torchvision.transforms.Normalize(mean=model.mean, std=model.std)

    transforms = [
        torchvision.transforms.Resize(256, antialias=True),
        torchvision.transforms.Lambda(lambda image: denorm(image)),
        torchvision.transforms.Lambda(lambda image: tencrop(image.permute(0, 2, 3, 1), cropped_size=224)),
        torchvision.transforms.Lambda(
            lambda image: torch.stack([torch.stack([normalize(x / 255) for x in crop]) for crop in image])),
        torchvision.transforms.Lambda(lambda image: change_view(image))
    ]
    if is_normalize:
        transforms.insert(0, torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    return torchvision.transforms.Compose(transforms)


def change_view(image):
    return image.view(-1, *image.shape[-3:])


def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1) * 255.0


def tencrop_output_transform_emonet(output):
    output = output.view(-1, 10).mean(1)
    # add fake arousal dimension for easy use
    output = torch.stack((output, torch.zeros(output.shape[0]).to(output.device)), dim=1)
    return output


def image_transform(model):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    normalize = torchvision.transforms.Normalize(mean=model.mean, std=model.std)
    return torchvision.transforms.Compose([
        torchvision.transforms.Lambda(
            lambda image: torch.nn.functional.interpolate(image, size=(224, 224), mode="bilinear")),
        torchvision.transforms.Lambda(lambda image: torch.stack([normalize(x / 255) for x in image])),
    ])


def tencrop(images, cropped_size=227):
    im_size = 256  # hard coded

    crops = torch.zeros(images.shape[0], 10, 3, cropped_size, cropped_size).to(images.device)
    indices = [0, im_size - cropped_size]  # image size - crop size

    for img_index in range(images.shape[0]):  # looping over the batch dimension
        img = images[img_index, :, :, :]
        curr = 0
        for i in indices:
            for j in indices:
                temp_img = img[i:i + cropped_size, j:j + cropped_size, :]
                crops[img_index, curr, :, :, :] = temp_img.permute(2, 0, 1)
                crops[img_index, curr + 5, :, :, :] = torch.flip(crops[img_index, curr, :, :, :], [2])
                curr = curr + 1
        center = int(math.floor(indices[1] / 2) + 1)
        crops[img_index, 4, :, :, :] = img[center:center + cropped_size,
                                           center:center + cropped_size, :].permute(2, 0, 1)
        crops[img_index, 9, :, :, :] = torch.flip(crops[img_index, curr, :, :, :], [2])

    return crops
