import os

import pandas as pd
from enum import Enum
from tqdm.auto import tqdm
from torchvision import transforms

from datasets.LDLDatasetPCLabeled import LDLDatasetPCLabeled
from datasets.ImageDirectoryDataset import ImageDirectoryDataset
from datasets.CocoCaptions import CocoCaptions

from image_transformations.image_transformations import *
from utils import plot_imgs_tensor, interweave_batch_tensors
from losses.CompoundEmotionLoss import CompoundEmotionLoss
from losses.ValenceArousalLoss import ValenceArousalLoss


def main():
    # parameters
    input_size = 1024
    crop_size = 1024
    batch_size = 12

    transformation_type = TransformationType.CUSTOM
    # transformation_type = TransformationType.SAME
    # transformation_type = TransformationType.MIN
    # transformation_type = TransformationType.MAX
    # transformation_type = TransformationType.RANDOM

    is_ind_func_check = False
    is_compare_emotions = False

    is_save_output = True
    is_adapt_one_batch = False

    # output_directory = "./COCO_BW"
    output_directory = "./COCO_MAN"
    is_plot_images = False
    is_debug = False

    # all transformations used in optimize_image_param
    # trans_to_apply = ['exposure', 'saturation', 'tone', 'color', 'contrast', 'sharp', 'blur', 'scale']

    trans_to_apply = ['exposure', 'saturation', 'tone', 'color', 'contrast', 'sharp', 'blur', 'scale']
    # trans_to_apply = ['bw']

    # parameters: end

    # inputs
    va_loss = ValenceArousalLoss("trained_models/va_pred_all", torch.device("cpu"), 1)

    # compound_loss = CompoundEmotionLoss("trained_models/emo_pred_ldl", torch.device("cpu"), 1)
    # inputs: end

    # TODO: CenterCrop can be removed for final image generation
    data_transforms = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(crop_size),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    # code for LDL Dataset
    # data_dir = "data/LDL"
    # labels_path = os.path.join(data_dir, "ground_truth.txt")
    # img_dir = "{}/images".format(data_dir)
    # img_labels = pd.read_csv(labels_path, sep=" ").iloc[:, :9]
    # dataset_test = LDLDatasetPCLabeled(img_labels, img_dir, data_transforms)

    # base_directory = "../GitRepos/SocialMediaExperimentalPlatform/Media/NAPS"
    # dataset_test = ImageDirectoryDataset(base_directory, data_transforms)
    dataset_test = CocoCaptions("./coco", "val", data_transforms)

    data_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=True, num_workers=0)

    for image, image_path in tqdm(data_loader):
        if isinstance(dataset_test, CocoCaptions):
            image_path = image_path[0]

        image_adapted = check_apply_params(
            image, trans_to_apply, transformation_type, is_ind_func_check, is_plot_images, is_debug)

        if is_compare_emotions:
            compare_emotions(va_loss, image, image_adapted, ['Valence', 'Arousal'])
            # print()
            # compare_emotions(compound_loss, image, image_adapted, ['Theta', 'Intensity'])

        if is_save_output:
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)
            save_output(output_directory, image_path, image_adapted)

        if is_adapt_one_batch:
            break


def check_apply_params(image, trans_to_apply, transformation_type, is_ind_func_check,
                       is_plot_images=True, is_debug=True):
    batch_size = image.size(0)
    output = None
    
    if transformation_type == TransformationType.CUSTOM:
        params = init_custom_params(trans_to_apply, batch_size)
    else:
        params = {}
        # for loop ensures that params are applied according to order in trans_to_apply
        for name in trans_to_apply:
            params[name] = None
    
    if is_ind_func_check:
        plot_imgs_tensor(image, "original")
    
    # Apply gamma
    if "gamma" in trans_to_apply:
        if transformation_type != TransformationType.CUSTOM:
            # defined from 0 to inf, meaningful 0 to 2
            if transformation_type == TransformationType.SAME:
                gamma = torch.ones(batch_size)
            elif transformation_type == TransformationType.MAX:
                gamma = 0.1 * torch.ones(batch_size)
            elif transformation_type == TransformationType.MIN:
                gamma = 2 * torch.ones(batch_size)
            else:
                gamma = 2 * torch.rand(batch_size)
            params["gamma"] = gamma
        
        if is_ind_func_check:
            # gamma_im = functional.adjust_gamma(im, gamma)
            output = apply_gamma(image, params["gamma"])
            plot_imgs_tensor(output, "gamma")

    # Apply sharpening
    if "sharp" in trans_to_apply:
        if transformation_type != TransformationType.CUSTOM:
            # defined from 0 to inf, meaningful 0 to 100
            if transformation_type == TransformationType.SAME:
                sharpness = 0 * torch.ones(batch_size).squeeze(0)
            else:
                # sharpness = 50 * torch.ones(batch_size).squeeze(0)
                sharpness = 50 * torch.rand(batch_size).squeeze(0)
            params["sharp"] = sharpness
        
        if is_ind_func_check:
            # sharp_im = functional.adjust_sharpness(im, sharpness_factor)
            output = apply_sharpening(image, params["sharp"])
            plot_imgs_tensor(output, "sharpening")

    # Apply white balance
    if "wb" in trans_to_apply:
        if transformation_type != TransformationType.CUSTOM:
            # defined from -1 to 1
            if transformation_type == TransformationType.SAME:
                white_balance = 0 * torch.ones(batch_size)
            elif transformation_type == TransformationType.MIN:
                white_balance = -1 * torch.ones(batch_size)
            elif transformation_type == TransformationType.MAX:
                white_balance = 1 * torch.ones(batch_size)
            else:
                white_balance = rand_tensor_minus_plus_1(batch_size)
            params["wb"] = white_balance
        
    if is_ind_func_check:
        output = apply_white_balance(image, params["wb"])
        plot_imgs_tensor(output, "white balance")

    # Apply brightness
    if "bright" in trans_to_apply:
        if transformation_type != TransformationType.CUSTOM:
            # defined from 0 to 1
            if transformation_type == TransformationType.SAME:
                brightness = torch.zeros(batch_size)
            elif transformation_type == TransformationType.MAX:
                brightness = torch.ones(batch_size)
            elif transformation_type == TransformationType.MIN:
                brightness = torch.zeros(batch_size)
            else:
                brightness = torch.rand(batch_size)
            params["bright"] = brightness

        if is_ind_func_check:
            # functional.adjust_brightness(im, brightness_factor)
            output = apply_brightness(image, params["bright"])
            plot_imgs_tensor(output, "bright")

    # Apply exposure
    if "exposure" in trans_to_apply:
        if transformation_type != TransformationType.CUSTOM:
            # from -2 to 2 gives meaningful adjustments
            if transformation_type == TransformationType.SAME:
                exposure = torch.zeros(batch_size)
            elif transformation_type == TransformationType.MAX:
                exposure = 2 * torch.ones(batch_size)
            elif transformation_type == TransformationType.MIN:
                exposure = -2 * torch.ones(batch_size)
            else:
                exposure = 2 * rand_tensor_minus_plus_1(batch_size)
            params["exposure"] = exposure

        if is_ind_func_check:
            output = apply_exposure(image, params["exposure"])
            plot_imgs_tensor(output, "exposure")

    # Apply contrast
    if "contrast" in trans_to_apply:
        if transformation_type != TransformationType.CUSTOM:
            # from 0 to 3 gives meaningful adjustments
            if transformation_type == TransformationType.SAME:
                contrast = torch.ones(batch_size)
            elif transformation_type == TransformationType.MAX:
                contrast = 3 * torch.ones(batch_size)
            elif transformation_type == TransformationType.MIN:
                contrast = 0.5 * torch.ones(batch_size)
            else:
                contrast = 3 * torch.rand(batch_size)
            params["contrast"] = contrast

        if is_ind_func_check:
            # functional.adjust_contrast(im, contrast_factor)
            output = apply_contrast(image, params["contrast"])
            plot_imgs_tensor(output, "contrast")

    # Apply saturation
    if "saturation" in trans_to_apply:
        if transformation_type != TransformationType.CUSTOM:
            # defined from 0 to inf, meaningful 0 to 10
            if transformation_type == TransformationType.SAME:
                saturation = torch.ones(batch_size)
            else:
                # saturation = 2 * torch.ones(batch_size)
                saturation = 10 * torch.rand(batch_size)
            params["saturation"] = saturation

        if is_ind_func_check:
            # functional.adjust_saturation(im, saturation_factor)
            output = apply_saturation(image, params["saturation"])
            plot_imgs_tensor(output, "saturation")

    # Apply black and white, produces very nice images, however more restricted than saturation filter.
    if "bw" in trans_to_apply:
        if transformation_type != TransformationType.CUSTOM:
            # defined from 0 to 1
            if transformation_type == TransformationType.SAME:
                bw = torch.zeros(batch_size)
            elif transformation_type == TransformationType.MIN:
                bw = torch.ones(batch_size)
            elif transformation_type == TransformationType.MAX:
                bw = torch.ones(batch_size)
            else:
                bw = torch.rand(batch_size)
            params["bw"] = bw

        if is_ind_func_check:
            # functional.adjust_saturation(im, saturation_factor)
            output = apply_black_white(image, params["bw"])
            plot_imgs_tensor(output, "black and white")

    # Apply hue
    if "hue" in trans_to_apply:
        if transformation_type != TransformationType.CUSTOM:
            # defined from -pi to pi
            if transformation_type == TransformationType.SAME:
                hue = 0.0 * torch.ones(batch_size)
            else:
                # hue = 0.5 * torch.ones(batch_size) * math.pi
                hue = rand_tensor_minus_plus_1(batch_size) * math.pi
            params["hue"] = hue

        if is_ind_func_check:
            output = apply_hue(image, params["hue"])
            plot_imgs_tensor(output, "hue")

    # Apply blur
    if "blur" in trans_to_apply:
        if transformation_type != TransformationType.CUSTOM:
            # from 0 to 10 gives meaningful adjustments
            if transformation_type == TransformationType.SAME:
                blur_param = 1e-10 * torch.ones(batch_size)
            else:
                # blur_param = 10 * torch.ones(batch_size)
                blur_param = 10 * torch.rand(batch_size)
            params["blur"] = blur_param

        if is_ind_func_check:
            # functional.gaussian_blur(im, kernel_size, [sigma])
            output = apply_gaussian_blur(image, params["blur"] )
            plot_imgs_tensor(output, "blur")

    # Apply tone curve adjustment
    if "tone" in trans_to_apply:
        if transformation_type != TransformationType.CUSTOM:
            # defined from 0 to 3
            if transformation_type == TransformationType.SAME:
                tone = torch.ones(batch_size, 8).view(batch_size, 1, 8, 1)
            else:
                tone = (3 * torch.rand(batch_size, 8)).view(batch_size, 1, 8, 1)
            params["tone"] = tone

        if is_ind_func_check:
            output = apply_tone_curve_adjustment(image, params["tone"])
            plot_imgs_tensor(output, "tone curve adjustment")

    # Apply color curve adjustment
    if "color" in trans_to_apply:
        if transformation_type != TransformationType.CUSTOM:
            # defined from 0 to 3
            if transformation_type == TransformationType.SAME:
                color = torch.ones(batch_size, 24).view(batch_size, 3, 8, 1)
            else:
                color = (3 * torch.rand(batch_size, 24)).view(batch_size, 3, 8, 1)
            params["color"] = color

        if is_ind_func_check:
            output = apply_color_curve_adjustment(image, params["color"])
            plot_imgs_tensor(output, "color curve adjustment")

    # Apply affine transformation
    if "affine" in trans_to_apply:
        if transformation_type != TransformationType.CUSTOM:
            matrix = torch.eye(2, 3)[None].repeat(batch_size, 1, 1)
            if not transformation_type == TransformationType.SAME:
                # translation
                matrix[:, :, 2] += (100 * rand_tensor_minus_plus_1(2 * batch_size)).reshape(batch_size, 2)
                # rotation
                matrix[:, :, :2] += (0.25 * rand_tensor_minus_plus_1((2 * batch_size, 2))).reshape(batch_size, 2, 2)
            params["affine"] = matrix

        if is_ind_func_check:
            output = kornia.geometry.transform.affine(image, params["affine"], padding_mode='border')
            plot_imgs_tensor(output, "affine")

    # Apply scale transformation
    if "scale" in trans_to_apply:
        if transformation_type != TransformationType.CUSTOM:
            scale_factor = torch.ones((1, 2)).repeat(batch_size, 1)
            
            if not transformation_type == TransformationType.SAME:
                scale_factor[:, :] += (3 * torch.rand(2 * batch_size)).reshape(batch_size, 2)
            params["scale"] = scale_factor
        
        if is_ind_func_check:
            output = kornia.geometry.transform.scale(image, params["scale"])
            plot_imgs_tensor(output, "scale")

    if not is_ind_func_check:
        img_list = apply_params(image, params)
        output = img_list[-1]
        # debug code
        if is_debug:
            img_list.append(image)
            trans_to_apply.append('original')
            print(params)
            for i in range(len(img_list)):
                if i == len(img_list) - 2:
                    print("{} - min: {}, max: {}".format(trans_to_apply[i], torch.min(img_list[i]).item(),
                                                         torch.max(img_list[i]).item()))
                plot_imgs_tensor(img_list[i], trans_to_apply[i])
        elif is_plot_images:
            plot_imgs_tensor(output, "adapted")
            plot_imgs_tensor(image, "original")

    return output


def compare_emotions(loss, image, image_adapted, emotion_type_labels=None, stats=None):
    emotion_type_labels = ['Valence', 'Arousal'] if emotion_type_labels is None else emotion_type_labels
    emotions = loss.predict_loss_metric(image)
    emotions_adapted = loss.predict_loss_metric(image_adapted)
    emotions_delta = emotions_adapted - emotions

    emotions = interweave_batch_tensors(emotions, emotions_adapted)
    df = pd.DataFrame(emotions.detach().cpu().numpy(), columns=emotion_type_labels)
    df['type'] = ['original' if i % 2 == 0 else 'adapted' for i in range(len(df))]
    print(df.groupby('type').mean())

    df_delta = pd.DataFrame(emotions_delta.detach().cpu().numpy(), columns=emotion_type_labels)
    print("delta")
    print(df_delta.mean(axis=0))

    rec_error = torch.nn.functional.l1_loss(image_adapted, image)
    print(f"reconstruction error: {rec_error.item()}\n")

    if stats is not None:
        stats["rec_error"].append(rec_error.item())

        ix = 0
        for emo_type in emotion_type_labels:
            stats[emo_type.lower()].append(emotions_adapted[0, ix].item())
            stats[f"delta_{emo_type.lower()}"].append(emotions_delta[0, ix].item())
            ix += 1


def init_custom_params(trans_to_apply, batch_size):
    params = {}
    # for loop ensures that params are applied according to order in trans_to_apply
    for name in trans_to_apply:

        # Apply gamma
        if "gamma" == name:
            params["gamma"] = torch.tensor(1.0)

        # Apply sharpening
        # defined from 0 to inf, meaningful 0 to 100
        if "sharp" == name:
            params["sharp"] = 0.0 * torch.ones(batch_size).squeeze(0)

        # Apply white balance
        if "wb" == name:
            params["wb"] = torch.tensor(0.0)

        # Apply brightness
        if "bright" == name:
            # defined from 0 to 1
            params["bright"] = torch.tensor(0.0)

        # Apply exposure
        if "exposure" == name:
            params["exposure"] = -0.1 * torch.ones(batch_size)

        # Apply contrast
        if "contrast" == name:
            # from 0 to 3 gives meaningful adjustments
            params["contrast"] = 0.85 * torch.ones(batch_size)

        # Apply saturation
        if "saturation" == name:
            # defined from 0 to inf, meaningful 0 to 10
            params["saturation"] = 0.85 * torch.ones(batch_size)

        # Apply black and white, produces very nice images, however more restricted than saturation filter.
        if "bw" == name:
            # defined from 0 to 1
            params["bw"] = torch.tensor(0.0)

        # Apply hue
        if "hue" == name:
            # defined from -pi to pi
            params["hue"] = torch.tensor(0.0)

        # Apply blur
        if "blur" == name:
            # from 0 to 10 gives meaningful adjustments
            params["blur"] = 1.0 * torch.ones(batch_size)

        # Apply tone curve adjustment
        if "tone" == name:
            # defined from 0 to 3
            params["tone"] = torch.ones(1, 8, 1).repeat(batch_size, 1, 1, 1)
            # params["tone"][:, 0, 1, 0] *= 1.5

        # Apply color curve adjustment
        if "color" == name:
            # defined from 0 to 3
            params["color"] = torch.ones(3, 8, 1).repeat(batch_size, 1, 1, 1)
            params["color"][:, 0, 1, 0] *= 0.7
            params["color"][:, 2, 1, 0] *= 0.7

        # Apply affine transformation
        if "affine" == name:
            params["affine"] = torch.eye(2, 3)

        # Apply scale transformation
        if "scale" == name:
            # 0=scale_x, 1=scale_y, 2=center_x, 3=center_y
            # params["scale"] = torch.ones(1, 2)
            params["scale"] = torch.ones(1, 4)
            params["scale"][0, 2:4] = 0.0

    return params


def save_output(directory, image_path, image_adapted):
    for i in range(image_adapted.size(0)):
        to_pil = transforms.ToPILImage()
        image_name = image_path[i].split("/")[-1].replace(".jpg", "")
        img_adapted_path = f'{directory}/{image_name}.jpg'
        to_pil(image_adapted[i, :]).save(img_adapted_path)


class TransformationType(Enum):
    SAME = 1
    RANDOM = 2
    MAX = 3
    MIN = 4
    CUSTOM = 5


def rand_tensor_minus_plus_1(shape):
    return 2 * torch.rand(shape) - 1


if __name__ == '__main__':
    main()
