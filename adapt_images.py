from __future__ import print_function
from __future__ import division
import os
import sys
import torch
import PIL.Image
import numpy as np
import pandas as pd

from InversionResamplingStableDiffusionPipeline import InversionResamplingStableDiffusionPipeline
from InversionResamplingStableDiffusionXLPipeline import InversionResamplingStableDiffusionXLPipeline

from guidance_classifier.ValenceArousalMidu import ValenceArousalMidu
import diff_utils


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
        },
        # "CFG_2": {
        #     "clf_scale": 0.0,
        #     "reference_value": None,
        #     "prompt": "The image should elicit low arousal and neutral valence in its viewers",
        #     "negative_prompt": "high arousal, high valence, low valence",
        #     "cfg_scale": 2.0,
        #     "use_caption": True,
        #     "is_nto": True
        # },
        # "CG_CFG_4": {
        #     "clf_scale": 0.1,
        #     "reference_value": None,
        #     "prompt": "",
        #     "negative_prompt": "",
        #     "cfg_scale": 4.0,
        #     "use_caption": True,
        #     "is_nto": True
        # },
        # "CFG_4": {
        #     "clf_scale": 0.0,
        #     "reference_value": None,
        #     "prompt": "The image should elicit low arousal and neutral valence in its viewers",
        #     "negative_prompt": "high arousal, high valence, low valence",
        #     "cfg_scale": 4.0,
        #     "use_caption": True,
        #     "is_nto": True
        # },
        # "CG_CFG_7.5": {
        #     "clf_scale": 0.1,
        #     "reference_value": None,
        #     "prompt": "",
        #     "negative_prompt": "",
        #     "cfg_scale": 7.5,
        #     "use_caption": True,
        #     "is_nto": True
        # },
        # "CFG_7.5": {
        #     "clf_scale": 0.0,
        #     "reference_value": None,
        #     "prompt": "The image should elicit low arousal and neutral valence in its viewers",
        #     "negative_prompt": "high arousal, high valence, low valence",
        #     "cfg_scale": 7.5,
        #     "use_caption": True,
        #     "is_nto": True
        # },
        # "CG": {
        #     "clf_scale": 0.1,
        #     "reference_value": None,
        #     "prompt": "",
        #     "negative_prompt": "",
        #     "cfg_scale": 1.0,
        #     "use_caption": True
        # },
        # "CFG_2_edit": {
        #     "clf_scale": 0.0,
        #     "reference_value": None,
        #     "prompt": "",
        #     "negative_prompt": "",
        #     "cfg_scale": 2.0,
        #     "use_caption": False,
        #     "is_nto": True
        # },
        # "CFG_4_edit": {
        #     "clf_scale": 0.0,
        #     "reference_value": None,
        #     "prompt": "",
        #     "negative_prompt": "",
        #     "cfg_scale": 4.0,
        #     "use_caption": False,
        #     "is_nto": True
        # },
        # "CFG_7.5_edit": {
        #     "clf_scale": 0.0,
        #     "reference_value": None,
        #     "prompt": "",
        #     "negative_prompt": "",
        #     "cfg_scale": 7.5,
        #     "use_caption": False,
        #     "is_nto": True
        # },
        # "INV_2": {
        #     "clf_scale": 0.0,
        #     "reference_value": None,
        #     "prompt": "",
        #     "negative_prompt": "",
        #     "cfg_scale": 2.0,
        #     "use_caption": True,
        #     "is_nto": True
        # },
        # "INV_4": {
        #     "clf_scale": 0.0,
        #     "reference_value": None,
        #     "prompt": "",
        #     "negative_prompt": "",
        #     "cfg_scale": 4.0,
        #     "use_caption": True,
        #     "is_nto": True
        # },
        # "INV_7.5": {
        #     "clf_scale": 0.0,
        #     "reference_value": None,
        #     "prompt": "",
        #     "negative_prompt": "",
        #     "cfg_scale": 7.5,
        #     "use_caption": True,
        #     "is_nto": True
        # }
        # "CG_CFG_2_POS_02": {
        #     "clf_scale": 0.1,
        #     "reference_value": 0.5,
        #     "prompt": "",
        #     "negative_prompt": "",
        #     "cfg_scale": 2.0,
        #     "use_caption": True,
        #     "is_nto": True
        # },
        # "CG_CFG_2_POS_01": {
        #     "clf_scale": 0.1,
        #     "reference_value": 0.25,
        #     "prompt": "",
        #     "negative_prompt": "",
        #     "cfg_scale": 2.0,
        #     "use_caption": True,
        #     "is_nto": True
        # },
        # "CG_CFG_2_NEUT": {
        #     "clf_scale": 0.1,
        #     "reference_value": 0.0,
        #     "prompt": "",
        #     "negative_prompt": "",
        #     "cfg_scale": 2.0,
        #     "use_caption": True,
        #     "is_nto": True
        # },
        # "CG_CFG_2_NEG_01": {
        #     "clf_scale": 0.1,
        #     "reference_value": -0.25,
        #     "prompt": "",
        #     "negative_prompt": "",
        #     "cfg_scale": 2.0,
        #     "use_caption": True,
        #     "is_nto": True
        # },
        # "CG_CFG_2_NEG_02": {
        #     "clf_scale": 0.1,
        #     "reference_value": -0.5,
        #     "prompt": "",
        #     "negative_prompt": "",
        #     "cfg_scale": 2.0,
        #     "use_caption": True,
        #     "is_nto": True
        # }
    }

    num_inversion_steps = 50
    num_inference_steps = 50
    end_iteration = num_inversion_steps
    invert_no_cg = False
    # invert_no_cg = True
    normalize_gradient = True
    is_xl = True
    index = 0
    # scheduler_type = "dpm"
    scheduler_type = "ddim"
    save_orig = False

    # clf_scales = []
    # clf_scales = [0.1, 0.3, 0.5, 0.7, 1.0]
    clf_scales = [0.2]
    if len(clf_scales) > 0:
        params = change_clf_scales(params, clf_scales, is_xl)

    # models
    # model_id = "aesthetic_midu"
    model_id = "va_midu"
    # model_id = "va_lat"
    # model_id = "arousal_midu"
    # model_id = "arousal_lat"
    if is_xl:
        pipe = InversionResamplingStableDiffusionXLPipeline(
            num_inference_steps, num_inversion_steps, normalize_gradient=normalize_gradient,
            scheduler_type=scheduler_type)  # device="cuda:0")
    else:
        pipe = InversionResamplingStableDiffusionPipeline(
            num_inference_steps, num_inversion_steps, normalize_gradient=normalize_gradient,
            scheduler_type=scheduler_type)

    guidance_classifier = initialize_guidance_classifier(model_id, pipe.pipe, pipe.device, is_xl)
    pipe.guidance_classifier = guidance_classifier

    if diff_utils.is_local():
        # directory = "/media/chris/Elements/GitRepos/SocialMediaExperimentalPlatform/Media/NAPS"
        directory = "/media/chris/Elements/GitRepos/SocialMediaExperimentalPlatform/Media/Instagram/OriginalPosts"
        output_path = "SD"
    else:
        directory = "/data/cgebhard/OriginalPosts"
        # directory = "/data/cgebhard/NAPS"
        # output_path = f"/data/cgebhard/feeds_data/{diff_utils.create_timestamp_folder_name()}"
        # output_path = f"/data/cgebhard/NAPS_SD"
        # output_path = f"/data/cgebhard/COCO_SD"
        output_path = f"/data/cgebhard/relative_change/diffusion"

    adapt_feed_images(pipe, directory, params, end_iteration, output_path, invert_no_cg, save_orig)
    # adapt_fixed_images(pipe, directory, params, end_iteration, output_path, invert_no_cg, save_orig)
    # adapt_coco_images(pipe, params, end_iteration, output_path, invert_no_cg, save_orig)


def adapt_image(pipe, image_path, params, end_iteration=None, output_path=".", caption="", invert_no_cg=True,
                save_orig=True, stats=None):
    image_name = image_path.split("/")[-1].replace(".jpg", "")
    input_image = PIL.Image.open(image_path)
    if input_image.mode != "RGB":
        input_image = input_image.convert("RGB")

    orig_img_score, setting = get_score(pipe, input_image)
    print_score(orig_img_score, "original", setting=setting)

    for adaptation, value in params.items():
        if value["reference_value"] is not None:
            value["reference_value"] = get_reference_value_from_alpha(
                value["reference_value"], orig_img_score, pipe.device)

    callback_resampling = None
    if invert_no_cg:
        invert_manager = OutputImageManager(
            f'{output_path}/{image_name}_inverted.jpg', pipe, label="inverted", orig_score=orig_img_score,
            orig_image=input_image, stats=stats)
        callback_resampling = invert_manager.callback

    img_path = f'{output_path}/{image_name}.jpg'
    output_manager = OutputImageManager(
        img_path, pipe, orig_score=orig_img_score, orig_image=input_image, stats=stats, output_path=output_path)
    input_image, _ = pipe.revert_and_sample(input_image, caption, end_iteration, params,
                                            callback_resampling=callback_resampling,
                                            callback_outputs=output_manager.callback)
    if save_orig:
        orig_img_path = f'{output_path}/{image_name}.jpg'
        input_image.save(orig_img_path)


def adapt_fixed_images(pipe, directory, params, end_iteration, output_path=".", invert_no_cg=True, save_orig=False):
    # image_path_list = diff_utils.get_fixed_exp_image_data("caption_files/fixed_images.json", directory)
    image_path_list = diff_utils.get_fixed_exp_image_data("caption_files/NAPS_images.json", directory)
    stats = {}
    for index in range(len(image_path_list)):
        # if index not in [0, 1, 7, 12]:
        #     continue

        print(f"\n[{index}]: " + image_path_list[index]["image_url"])
        # end_iteration_i = (end_iteration if "diffusion_params" not in image_path_list[index] else
        #                    image_path_list[index]["diffusion_params"]["end_iteration"])
        caption = image_path_list[index]["captions"][0]

        # edit in key signalizes that the edited caption is taken
        for key in params.keys():
            if "edit" in key:
                params[key]["prompt"] = image_path_list[index]["caption_edit"]

        adapt_image(pipe, image_path_list[index]["image_url"], params, end_iteration, output_path,
                    invert_no_cg=invert_no_cg, caption=caption, stats=stats, save_orig=save_orig)

    print_stats(stats)


def adapt_coco_images(pipe, params, end_iteration, output_path=".", invert_no_cg=True, save_orig=False):
    base_directory = "../emotion-adaptation/coco" if diff_utils.is_local() else "/data/cgebhard/coco"
    sys.path.append('../emotion-adaptation' if diff_utils.is_local() else "/local/home/cgebhard/emotion-adaptation")
    from datasets.CocoCaptions import CocoCaptions

    dataset_test = CocoCaptions(base_directory, "val", None)
    data_loader = torch.utils.data.DataLoader(dataset_test, batch_size=1, shuffle=True, num_workers=0)
    neutral_captions = diff_utils.load_json("./caption_files/neutral_captions_val2017.json")

    ix = 0
    stats = {}
    for _, data in data_loader:
        print(f"[ {ix + 1} / {len(data_loader.dataset)} ]: {data[0][0]}")
        ix += 1

        caption = data[2][0].split("/")[0]

        # edit in key signalizes that the edited caption is taken
        for key in params.keys():
            if "edit" in key:
                params[key]["prompt"] = neutral_captions[data[0][0]]

        adapt_image(pipe, data[1][0], params, end_iteration, output_path, invert_no_cg=invert_no_cg, caption=caption,
                    stats=stats, save_orig=save_orig)

        print_stats(stats)


def adapt_feed_images(pipe, directory, params, end_iteration, output_directory=".", invert_no_cg=True, save_orig=False):
    image_path_list = diff_utils.get_feed_exp_image_data(
        "caption_files/feed_images.json", directory, output_directory)
    stats = {}
    for index in range(len(image_path_list)):
        # if index == 500:
        #     break
        print(f"\n[{index}/{len(image_path_list)}]: " + image_path_list[index]["image_path"])
        # caption = image_path_list[index]["captions"][0]
        caption = image_path_list[index]["captions"][np.random.randint(len(image_path_list[index]["captions"]))]
        # output_path = image_path_list[index]["output_path"]
        output_path = output_directory

        adapt_image(pipe, image_path_list[index]["image_path"], params, end_iteration, output_path,
                    invert_no_cg=invert_no_cg, caption=caption, stats=stats, save_orig=save_orig)

    print_stats(stats)

    # Flatten the dictionary by extracting the single element from each list
    stats = {key: {k: v[0] for k, v in value.items()} for key, value in stats.items()}
    df = pd.DataFrame.from_dict(stats, orient='index')
    df.to_csv(f'{output_directory}/stats.csv')


def initialize_guidance_classifier(model_id, pipe, device, is_xl):
    if diff_utils.is_local():
        directory = "trained_models"
    else:
        directory = "/data/cgebhard/classifiers"
        # directory = "/data1/chris/classifiers"

    # initialize guidance classifier
    # midu
    if model_id == "va_midu":
        if is_xl:
            guidance_classifier = ValenceArousalMidu(
                pipe, device, ckp_path=f"{directory}/clf_best_cont_midu_va_1024_2024_07_22_16_01_14", is_sdxl=True)
        else:
            guidance_classifier = ValenceArousalMidu(
                pipe, device, ckp_path=f"{directory}/clf_new_params_midu_va_512_2024_07_11_09_10_03")
    else:
        raise ValueError(f"Model ID {model_id} not recognized")

    return guidance_classifier


def print_score(score, label, orig_score=None, setting="va"):
    if setting not in ["va", "valence", "arousal", "aesthetic"]:
        raise ValueError("Invalid setting. Must be 'va', 'valence', 'arousal' or 'aesthetic'.")

    if orig_score is None:
        if setting == "va":
            print(f"Score {label}: valence {score[0, 0].item():.4f}, arousal {score[0, 1].item():.4f}")
            return 0.0, 0.0
        elif setting == "valence" or setting == "aesthetic" or setting == "arousal":
            print(f"Score {label}: {setting} {score[0].item():.4f}")
            return 0.0

    delta = score - orig_score

    if setting == "va":
        print(f"Score {label}: valence {score[0, 0].item():.4f} delta {delta[0, 0].item():.4f}, "
              f"arousal {score[0, 1].item():.4f} delta {delta[0, 1].item():.4f}")
        return delta[0, 0].item(), delta[0, 1].item()
    elif setting == "valence" or setting == "aesthetic" or setting == "arousal":
        print(f"Score {label}: {setting} {score[0].item():.4f} delta {delta[0].item():.4f}")
        return delta[0].item()


def get_score(pipe, image, prompts=None):
    prompts = prompts if prompts is not None else ["", ""]
    image = pipe.transform_image(image)
    score = pipe.guidance_classifier.predict_score(
        pipe.get_latents_from_img(image, torch.float16), pipe.pipe.scheduler.timesteps[-1], prompts)

    setting = "va"
    if isinstance(pipe.guidance_classifier, AestheticMidu):
        setting = "aesthetic"
    elif isinstance(pipe.guidance_classifier, ArousalMidu) or isinstance(pipe.guidance_classifier, ArousalLatents):
        setting = "arousal"
    return score, setting


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

    def __init__(self, img_path, pipe, prompts=None, label=None, orig_score=None, orig_image=None, stats=None,
                 output_path=None):
        self.image_path = img_path
        self.pipe = pipe
        self.prompts = prompts
        self.label = label
        self.orig_score = orig_score
        self.orig_image = orig_image
        self.stats = stats
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

                # code for feeds result generation
                # label = image_name
                # image_path = f"{self.output_path}/{image_name}"
                # if not os.path.exists(self.output_path):
                #     os.makedirs(self.output_path)

                # code for analysis
                image_path = f"{self.output_path}/{label}/{image_name}"
                if not os.path.exists(f"{self.output_path}/{label}"):
                    os.makedirs(f"{self.output_path}/{label}")

        image.save(image_path)
        score, setting = get_score(self.pipe, image, self.prompts)
        delta = print_score(score, label, self.orig_score, setting)

        rec_error = 0.0
        if self.orig_image is not None:
            orig_image_tensor = self.pipe.transform_image(self.orig_image)
            image_tensor = self.pipe.transform_image(image)
            # mean absolute error as reconstruction error
            rec_error = torch.mean(torch.abs(image_tensor - orig_image_tensor)).item()
            print("Reconstruction error: {:.4f}".format(rec_error))

        if self.stats is not None:
            if label not in self.stats:
                self.stats[label] = {}
                if setting in ["va", "valence"]:
                    self.stats[label]["valence"] = []
                    self.stats[label]["delta_valence"] = []
                if setting in ["va", "arousal"]:
                    self.stats[label]["arousal"] = []
                    self.stats[label]["delta_arousal"] = []
                if setting == "aesthetic":
                    self.stats[label]["aesthetic"] = []
                    self.stats[label]["delta_aesthetic"] = []
                self.stats[label]["rec_error"] = []

            if setting == "va":
                self.stats[label]["valence"].append(score[0, 0].item())
                self.stats[label]["delta_valence"].append(delta[0])
                self.stats[label]["arousal"].append(score[0, 1].item())
                self.stats[label]["delta_arousal"].append(delta[1])

                # per image stats if necessary
                self.stats[image_name + "/" + label] = {}
                self.stats[image_name + "/" + label]["valence"] = [score[0, 0].item()]
                self.stats[image_name + "/" + label]["delta_valence"] = [delta[0]]
                self.stats[image_name + "/" + label]["arousal"] = [score[0, 1].item()]
                self.stats[image_name + "/" + label]["delta_arousal"] = [delta[1]]
            else:
                self.stats[label][setting].append(score[0].item())
                self.stats[label][f"delta_{setting}"].append(delta)

            self.stats[label]["rec_error"].append(rec_error)


if __name__ == '__main__':
    main()
