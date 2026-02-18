#! /usr/bin/env -S uv run adapt_images.py

from __future__ import print_function
from __future__ import division
import os
import torch
import PIL.Image
import numpy as np

from pipelines.InversionResamplingStableDiffusionPipeline import InversionResamplingStableDiffusionPipeline
from pipelines.InversionResamplingStableDiffusionXLPipeline import InversionResamplingStableDiffusionXLPipeline

from guidance_classifier.ValenceArousalMidu import ValenceArousalMidu
from datasets.CocoCaptions import CocoCaptions

import pipelines.diff_utils as diff_utils
from paths import MODELS_DIR, COCO_DIR, IMAGE_OUTPUT_DIR


os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
torch.set_num_threads(1)


def main():
    # parameters
    params = {
        "CG_CFG_2": {
            "clf_scale": 0.1,
            "reference_value": None,
            "prompt": "",
            "negative_prompt": "",
            "cfg_scale": 2.0,
            "use_caption": True,
            "is_nto": True,
            "max": True
        }
    }

    
    num_inversion_steps = 5
    num_inference_steps = 5
    end_iteration = num_inversion_steps
    invert_no_cg = False
    normalize_gradient = True
    is_xl = False
    scheduler_type = "ddim"
    save_orig = False

    clf_scales = [0.2]
    if len(clf_scales) > 0:
        params = change_clf_scales(params, clf_scales, is_xl)

    if is_xl:
        pipe = InversionResamplingStableDiffusionXLPipeline(
            num_inference_steps, num_inversion_steps, normalize_gradient=normalize_gradient,
            scheduler_type=scheduler_type)  # device="cuda:0")
    else:
        pipe = InversionResamplingStableDiffusionPipeline(
            num_inference_steps, num_inversion_steps, normalize_gradient=normalize_gradient,
            scheduler_type=scheduler_type)

    guidance_classifier = initialize_guidance_classifier(pipe.pipe, pipe.device, is_xl)
    pipe.guidance_classifier = guidance_classifier

    adapt_coco_images(pipe, params, end_iteration, COCO_DIR, IMAGE_OUTPUT_DIR, invert_no_cg, save_orig)


def adapt_image(pipe, image_path, params, end_iteration=None, output_path=".", caption="", invert_no_cg=True,
                save_orig=True):
    image_name = image_path.split("/")[-1].replace(".jpg", "")
    input_image = PIL.Image.open(image_path)
    if input_image.mode != "RGB":
        input_image = input_image.convert("RGB")

    orig_img_score = get_score(pipe, input_image)
    print_score(orig_img_score, "original")

    for _ , value in params.items():
        if value["reference_value"] is not None:
            value["reference_value"] = get_reference_value_from_alpha(
                value["reference_value"], orig_img_score, pipe.device)

    callback_resampling = None
    if invert_no_cg:
        invert_manager = OutputImageManager(
            f'{output_path}/{image_name}_inverted.jpg', pipe, label="inverted", orig_score=orig_img_score,
            orig_image=input_image)
        callback_resampling = invert_manager.callback

    img_path = f'{output_path}/{image_name}.jpg'
    output_manager = OutputImageManager(
        img_path, pipe, orig_score=orig_img_score, orig_image=input_image, output_path=output_path)
    input_image, _ = pipe.revert_and_sample(input_image, caption, end_iteration, params,
                                            callback_resampling=callback_resampling,
                                            callback_outputs=output_manager.callback)
    if save_orig:
        orig_img_path = f'{output_path}/{image_name}.jpg'
        input_image.save(orig_img_path)


def adapt_coco_images(pipe, params, end_iteration, coco_directory, output_path=".", invert_no_cg=True, save_orig=False):
    dataset_test = CocoCaptions(coco_directory, "val", None)
    data_loader = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=True, num_workers=0)

    ix = 0
    stats = {}
    for _, data in data_loader:
        print(f"[ {ix + 1} / {len(data_loader.dataset)} ]: {data[0][0]}")
        ix += 1

        caption = data[2][0].split("/")[0]

        adapt_image(pipe, data[1][0], params, end_iteration, output_path, invert_no_cg=invert_no_cg, caption=caption, save_orig=save_orig)

        print_stats(stats)


def initialize_guidance_classifier(pipe, device, is_xl):
    if diff_utils.is_local():
        directory = "trained_models"
    else:
        directory = MODELS_DIR
        # directory = "/data1/chris/classifiers"

    # initialize guidance classifier
    if is_xl:
        guidance_classifier = ValenceArousalMidu(
            pipe, device, ckp_path=f"{directory}/clf_best_cont_midu_va_1024_2024_07_22_16_01_14", is_sdxl=True)
    else:
        guidance_classifier = ValenceArousalMidu(
            pipe, device, ckp_path=f"{directory}/clf_new_params_midu_va_512_2024_07_11_09_10_03")

    return guidance_classifier


def print_score(score, label, orig_score=None):
    if orig_score is None:
        print(f"Score {label}: valence {score[0, 0].item():.4f}, arousal {score[0, 1].item():.4f}")
        return 0.0, 0.0

    delta = score - orig_score
    print(f"Score {label}: valence {score[0, 0].item():.4f} delta {delta[0, 0].item():.4f}, "
          f"arousal {score[0, 1].item():.4f} delta {delta[0, 1].item():.4f}")
    return delta[0, 0].item(), delta[0, 1].item()


def get_score(pipe, image, prompts=None):
    prompts = prompts if prompts is not None else ["", ""]
    image = pipe.transform_image(image)
    score = pipe.guidance_classifier.predict_score(
        pipe.get_latents_from_img(image, torch.float16), pipe.pipe.scheduler.timesteps[-1], prompts)

    return score


def get_reference_value_from_alpha(alpha, score, device):
    reference = score + torch.ones(score.shape).to(score.device) * alpha
    # Clip the values of the tensor between 0 and 1 (feasible range of valence and arousal
    return torch.clamp(reference, min=0.0, max=1.0).to(device)


def print_stats(stats):
    for label, data in stats.items():
        stats_str = f"{label}: "
        for stat, values in data.items():
            stats_str += f"{stat}: mean {np.mean(values):.4f}, std {np.std(values):.4f}; "
        print(stats_str)


def change_clf_scales(params, clf_scales, is_xl):
    new_dict = {}
    for config, values in params.items():
        if values["clf_scale"] > 0.0:
            for scale in clf_scales:
                values = values.copy()
                values["clf_scale"] = scale
                if is_xl:
                    new_dict[f"XL_{config}_{scale:.2f}"] = values
                else:
                    new_dict[f"{config}_{scale:.2f}"] = values
        else:
            if is_xl:
                new_dict[f"XL_{config}"] = values
            else:
                new_dict[config] = values

    return new_dict


class OutputImageManager:
    """
    Class managing results of pipeline
    """

    def __init__(self, img_path, pipe, prompts=None, label=None, orig_score=None, orig_image=None, output_path=None):
        self.image_path = img_path
        self.pipe = pipe
        self.prompts = prompts
        self.label = label
        self.orig_score = orig_score
        self.orig_image = orig_image
        self.output_path = output_path

    def callback(self, image, label=None):
        if label is None:
            label = self.label
            image_path = self.image_path
        else:
            if self.output_path is None:
                image_path = self.image_path.replace(".jpg", f"_{label}.jpg")
            else:
                image_name = self.image_path.split("/")[-1]

                # code for analysis
                image_path = f"{self.output_path}/{label}/{image_name}"
                if not os.path.exists(f"{self.output_path}/{label}"):
                    os.makedirs(f"{self.output_path}/{label}")

        image.save(image_path)
        score = get_score(self.pipe, image, self.prompts)
        print_score(score, label, self.orig_score)

        rec_error = 0.0
        if self.orig_image is not None:
            orig_image_tensor = self.pipe.transform_image(self.orig_image)
            image_tensor = self.pipe.transform_image(image)
            # mean absolute error as reconstruction error
            rec_error = torch.mean(torch.abs(image_tensor - orig_image_tensor)).item()
            print("Reconstruction error: {:.4f}".format(rec_error))

if __name__ == '__main__':
    main()
