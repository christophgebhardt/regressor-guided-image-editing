#! /usr/bin/env -S uv run optimize_image_param.py

import os

import torch
import PIL.Image
import torch.nn as nn
from torchvision import transforms

from datasets.CocoCaptions import CocoCaptions

from baselines.image_transformations.image_transformations import apply_params
from baselines.losses.ValenceArousalLoss import ValenceArousalLoss
from baselines.models.Discriminator import Discriminator
from baselines.optimize_image import optimize_images, compute_clip_loss
from baselines.run_img_trans import compare_emotions
from baselines.utils import check_init_stats_adapt, print_stats, is_local, get_str_timestamp

from paths import COCO_DIR, MODELS_DIR, OUT_DIR

STATS = {}
OUTPUT_TRANSFORM = None

OUTPUT_PATH = OUT_DIR / "optimized_param"
VA_MODEL = MODELS_DIR / "va_pred_all"


def main():
    global OUTPUT_TRANSFORM
    # parameters
    is_gradient_free = False
    learning_rate = 0.05

    params = {
        # "min": {},
        "pos_01": {"alpha": 0.1},
        "pos_02": {"alpha": 0.2},
        "neg_01": {"alpha": -0.1},
        "neg_02": {"alpha": -0.1},
        "neutral": {"alpha": 0.0}
        # "max": {}
    }

    # weight_clf_list = [0.1, 0.3, 0.5, 0.7, 1.0]
    weight_clf_list = [0.15]
    weight_dis = 0.0
    weight_recon = 1.0

    show_results = False
    verbose = False
    save_orig_img = False
    output_size = 1024
    input_size = 480
    crop_size = input_size
    # crop_size = 448
    batch_size = 1
    num_steps = 300  # in most times 100 are enough sometimes it is necessary to have more step, COCO requires 300
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # parameters: end

    # inputs
    va_loss = ValenceArousalLoss(f"{VA_MODEL}", device, 1,
                                 is_minimized=True, requires_grad=not is_gradient_free)
    eval_params = {"emotion_type_labels": None}
    # va_loss = ValenceArousalLoss("trained_models/EmoNet_valence_moments_resnet50_5_best.pth.tar",
    #                              device, 1, is_input_range_0_1=False, is_minimized=True,
    #                              requires_grad=not is_gradient_free, loss="valence")
    # eval_params = {"emotion_type_labels": ['Valence']}

    data_transforms = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(crop_size),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    # TODO: CenterCrop can be removed for final image generation
    OUTPUT_TRANSFORM = transforms.Compose([
        transforms.Resize(output_size),
        transforms.CenterCrop(output_size),
        transforms.ToTensor()
    ])




    dataset_test = CocoCaptions(COCO_DIR, "val", data_transforms)
    data_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)

    net = None
    if weight_dis > 0:
        num_features = 64
        dis = Discriminator(num_features, input_size, input_size)
        state_dict = torch.load("trained_models/imagenet_w0_high_lookhere_dis")
        dis.load_state_dict(state_dict)

        net = NetWithCriterion(dis, nn.BCELoss())
        net = net.to(device)

    for weight_clf in weight_clf_list:
        output_path_i = f'{OUTPUT_PATH}_{weight_clf:<1.2f}'

        if not os.path.exists(output_path_i):
            os.makedirs(output_path_i)

        params_default = {
            "clf": va_loss, "dis": net, "weight_clf": weight_clf, "weight_dis": weight_dis, "weight_recon": weight_recon
        }
        for key in params.keys():
            params[key] = {**params[key], **params_default}

        optimize_images(data_loader, params, initialize_parametric, objective_function_parametric, device,
                        output_path_i, output_transform, eval_params, is_gradient_free, verbose,
                        num_steps=num_steps, save_orig_img=save_orig_img, learning_rate=learning_rate,
                        show_results=show_results)

        print(f"weight_clf: {weight_clf}; weight_dis: {weight_dis}; weight_recon: {weight_recon}")
        print_stats(STATS)


def init_params(trans_to_apply):
    x0 = []
    params = {}
    # for loop ensures that params are applied according to order in trans_to_apply
    for name in trans_to_apply:

        # Apply gamma
        if "gamma" == name:
            params["gamma"] = 1.0
            x0.append(params["gamma"])

        # Apply sharpening
        # defined from 0 to inf, meaningful 0 to 100
        if "sharp" == name:
            params["sharp"] = 0.0
            x0.append(params["sharp"])

        # Apply white balance
        if "wb" == name:
            params["wb"] = 0.0
            x0.append(params["wb"])

        # Apply brightness
        if "bright" == name:
            # defined from 0 to 1
            params["bright"] = 0.0
            x0.append(params["bright"])

        # Apply exposure
        if "exposure" == name:
            params["exposure"] = 0.0
            x0.append(params["exposure"])

        # Apply contrast
        if "contrast" == name:
            # from 0 to 3 gives meaningful adjustments
            params["contrast"] = 1.0
            x0.append(params["contrast"])

        # Apply saturation
        if "saturation" == name:
            # defined from 0 to inf, meaningful 0 to 10
            params["saturation"] = 1.0
            x0.append(params["saturation"])

        # Apply black and white, produces very nice images, however more restricted than saturation filter.
        if "bw" == name:
            # defined from 0 to 1
            params["bw"] = 0.0
            x0.append(params["bw"])

        # Apply hue
        if "hue" == name:
            # defined from -pi to pi
            params["hue"] = 0.0
            x0.append(params["hue"])

        # Apply blur
        if "blur" == name:
            # from 0 to 10 gives meaningful adjustments
            params["blur"] = 1e-4
            x0.append(params["blur"])

        # Apply tone curve adjustment
        if "tone" == name:
            # defined from 0 to 3
            params["tone"] = torch.ones(1, 8, 1)
            x0.extend(params["tone"].view(-1).tolist())

        # Apply color curve adjustment
        if "color" == name:
            # defined from 0 to 3
            params["color"] = torch.ones(3, 8, 1)
            x0.extend(params["color"].view(-1).tolist())

        # Apply affine transformation
        if "affine" == name:
            params["affine"] = torch.eye(2, 3)
            x0.extend(params["affine"].view(-1).tolist())

        # Apply scale transformation
        if "scale" == name:
            # 0=scale_x, 1=scale_y, 2=center_x, 3=center_y
            # params["scale"] = torch.ones(1, 2)
            params["scale"] = torch.ones(1, 4)
            params["scale"][0, 2:4] = 0.0
            x0.extend(params["scale"].view(-1).tolist())

    return params, torch.tensor(x0)


def initialize_parametric(image, obj_params):
    # all transformations, ordered as in code of look-here
    # trans_to_apply = ['gamma', 'sharp', 'wb', 'exposure', 'bright', 'contrast', 'saturation', 'bw',
    #                   'tone', 'hue', 'color', 'blur', 'affine']

    # all transformations, ordered from low to high (tone to approximate lighting conditions)
    # trans_to_apply = ['exposure', 'bright', 'gamma', 'bw', 'wb', 'hue', 'color',
    #                   'saturation', 'contrast', 'sharp', 'blur', 'tone', 'affine', 'scale']

    # look here transformations
    # trans_to_apply = ['sharp', 'exposure', 'contrast', 'tone', 'color']

    # theoretic transformations, ordered from low to high (tone and color to approximate lighting conditions)
    # trans_to_apply = ['exposure', 'saturation', 'tone', 'color', 'contrast', 'sharp', 'blur', 'affine']

    trans_to_apply = ['exposure', 'saturation', 'tone', 'color', 'contrast', 'sharp', 'blur', 'scale']

    params_trans, x0 = init_params(trans_to_apply)
    x0 = x0.to(image.device)
    obj_params["image"] = image
    obj_params["params"] = params_trans

    return x0, obj_params


def objective_function_parametric(
        x_opt, image, params, clf, weight_clf, weight_dis, weight_recon, dis=None, target=None):
    params_x = get_params_from_vector(x_opt, 1, params, image.size(2))
    outputs = apply_params(image, params_x)

    loss = weight_clf * clf(outputs[-1], target=target)

    if dis is not None and weight_dis > 0:
        loss_dis = dis(image)
        loss -= weight_dis * loss_dis
        # loss += weight_dis * torch.relu(-(loss_dis - 0.5))

    if weight_recon > 0:
        # L1 loss on black and white image
        # bw = torch.ones(1).to(image.device)
        # image_bw = apply_black_white(image, bw)
        # output_bw = apply_black_white(outputs[-1], bw)
        # loss += weight_recon * torch.nn.functional.l1_loss(output_bw, image_bw)
        #
        # Clip loss
        loss += weight_recon * compute_clip_loss(image, outputs[-1])

    return loss


def get_params_from_vector(x, batch_size, params, input_size=480):
    ix_start = 0
    for name in params.keys():
        len_param = 1 if isinstance(params[name], float) else len(params[name].view(-1))
        if len_param == 1:
            params[name] = x[ix_start]
        else:
            param_tensor = x[ix_start:ix_start + len_param]
            if "tone" == name:
                params["tone"] = param_tensor.view(batch_size, 1, 8, 1)
            elif "color" == name:
                params["color"] = param_tensor.view(batch_size, 3, 8, 1)
            elif "affine" == name:
                params["affine"] = param_tensor.view(2, 3)[None].repeat(batch_size, 1, 1)
            elif "scale" == name:
                if param_tensor.size(0) == 4:
                    # The following code is to avoid in-place operations to keep the gradient.
                    # Clamp the scale to be larger 1. (to avoid black margin) and the center to be within the image dim.
                    clamped_scale = param_tensor[0:2].clamp(min=1.0, max=torch.inf)
                    clamped_center = param_tensor[2:].clamp(min=0.0, max=input_size)
                    param_tensor = torch.cat((clamped_scale, clamped_center))
                else:
                    # Clamp the scale to be larger 1. (to avoid black margin)
                    param_tensor = param_tensor.clamp(min=1.0, max=5.0)

                params["scale"] = param_tensor.repeat(batch_size, 1)

        ix_start += len_param

    params["contrast"] = 0.0 if params["contrast"] < 0 else params["contrast"]
    return params


def output_transform(image, x_opt, obj_params, eval_params, adaptation, image_path):
    params_x = get_params_from_vector(x_opt, 1, obj_params["params"], image.size(2))
    outputs = apply_params(image, params_x)
    print(f"optimized scale: {params_x['scale'].tolist()}")

    # evaluation
    check_init_stats_adapt(STATS, adaptation)
    compare_emotions(obj_params["clf"], image, outputs[-1], eval_params["emotion_type_labels"], STATS[adaptation])

    input_image = PIL.Image.open(image_path)
    if input_image.mode != "RGB":
        input_image = input_image.convert('RGB')

    input_image_tensor = OUTPUT_TRANSFORM(input_image).unsqueeze(0).to(image.device)
    output_image_tensor = apply_params(input_image_tensor, params_x)[-1]

    return input_image_tensor, output_image_tensor


class NetWithCriterion(nn.Module):
    """
    Define a custom module that includes both the network and the criterion.
    """
    def __init__(self, dis, criterion, label=1.0):
        super(NetWithCriterion, self).__init__()
        self.network = dis
        self.criterion = criterion
        self.label = label

    def forward(self, x):
        output = self.network(x)
        label_tensor = output.clone()
        label_tensor.fill_(self.label)
        loss = self.criterion(output, label_tensor)
        return loss


if __name__ == '__main__':
    main()

