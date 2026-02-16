import os
import json
import random
import torch
import numpy as np
from datetime import datetime
from scipy.optimize import curve_fit

from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, LMSDiscreteScheduler, StableDiffusionPipeline
from diffusers import DDIMScheduler, DDPMScheduler
from torchvision.transforms import functional

import plot_utils


def condition_image_tensor(image, condition_array):
    """
    Conditions an image tensor according to the passed condition tensor
    :param image: numpy image
    :param condition_array: batch of tensors conditioning the images
    :return: same image tensors with additional dimensions reflecting the conditioning
    """
    if condition_array is None:
        condition_array = np.zeros((1, 1, 2))
    else:
        dim = 1 if len(condition_array.shape) == 1 else condition_array.shape[1]
        condition_array = condition_array.reshape(1, 1, dim)
        if dim < 2:
            condition_array = np.concatenate((condition_array, np.zeros((1, 1, 1))), axis=2)

    conditioning = np.ones((image.shape[0], image.shape[1], 2)) * condition_array
    return np.concatenate((image, conditioning), axis=2)


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def load_controlnet_sd_pipeline(device=None, controlnet_path="lllyasviel/sd-controlnet-canny",
                                pipe_path="runwayml/stable-diffusion-v1-5"):
    # Load the SD pipeline and add a hook
    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
    if device is not None:
        controlnet = controlnet.to(device)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(pipe_path, controlnet=controlnet,
                                                             torch_dtype=torch.float16).to(device)
    if device is not None:
        pipe = pipe.to(device)

    return pipe


def get_guidance_controlnet_pipe(device, controlnet_path="lllyasviel/sd-controlnet-canny",
                                 pipe_path="runwayml/stable-diffusion-v1-5"):
    pipe = load_controlnet_sd_pipeline(device, controlnet_path, pipe_path)
    pipe.scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012,
                                          beta_schedule="scaled_linear", num_train_timesteps=1000)
    pipe.scheduler.set_timesteps(30)

    def hook_fn(module, input, output):
        module.output = output

    pipe.unet.mid_block.register_forward_hook(hook_fn)
    return pipe


def load_sd_pipeline(device, pipe_path="stabilityai/stable-diffusion-2-1-base"):
    # Load the SD pipeline and add a hook
    pipe = StableDiffusionPipeline.from_pretrained(pipe_path).to(device)
    pipe.scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                          num_train_timesteps=1000)
    pipe.scheduler.set_timesteps(30)

    def hook_fn(module, input, output):
        module.output = output

    pipe.unet.mid_block.register_forward_hook(hook_fn)
    return pipe


def get_random_image_path(directory_path):
    # Check if the directory exists
    if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
        return None  # Return None if the directory doesn't exist or is not a directory

    # Get a list of all image files in the directory
    image_files = [f for f in os.listdir(directory_path) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]

    # Check if there are any image files in the directory
    if not image_files:
        return None  # Return None if there are no image files in the directory

    # Choose a random image file from the list
    random_image_file = random.choice(image_files)

    # Return the path to the random image
    random_image_path = os.path.join(directory_path, random_image_file)
    return random_image_path


def decode_latents(pipe, latents):
    latents = 1 / pipe.vae.config.scaling_factor * latents
    image = pipe.vae.decode(latents, return_dict=False)[0]
    return image.clamp(-1, 1)


def decode_to_pil(pipe, latents):
    with torch.no_grad():
        image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
    do_denormalize = [True] * image.shape[0]
    return pipe.image_processor.postprocess(image.detach(), output_type="pil", do_denormalize=do_denormalize)


def decode_latents_np(pipe, latents):
    image = decode_latents(pipe, latents)
    return convert_tensor_to_np(image)


def convert_tensor_to_np(image_tensor):
    image_tensor = (image_tensor / 2 + 0.5)
    # cast to float32 as this does not cause significant overhead and is compatible with bfloat16
    return image_tensor.cpu().permute(0, 2, 3, 1).float().numpy()


def get_scheduler_guidance_scaling(scheduler, t, get_scaling_for_eps=False, **kwargs, ):
    """
    The guidance is directly applied to x_{t-1} and x_{t}, so we need to scale the guidance formulas given for
    predicted noise.
    This function assumes noise prediction models.
    From supplemental materials, and Algorithm of the paper:
    @misc{dhariwal2021diffusion,
      title={Diffusion Models Beat GANs on Image Synthesis},
      author={Prafulla Dhariwal and Alex Nichol},
      year={2021},
      eprint={2105.05233},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
    }
    """
    if isinstance(scheduler, DDPMScheduler):
        coef = scheduler._get_variance(t)
        return coef

    elif isinstance(scheduler, DDIMScheduler):
        '''Mostly copied from DDIMScheduler.step()'''

        eta = kwargs.get("eta", 0.0)

        # 1. get previous step value (=t-1)
        prev_timestep = t - scheduler.config.num_train_timesteps // scheduler.num_inference_steps  ##For debug

        # 2. compute alphas, betas
        alpha_prod_t = scheduler.alphas_cumprod[t]
        alpha_prod_t_prev = scheduler.alphas_cumprod[
            prev_timestep] if prev_timestep >= 0 else scheduler.final_alpha_cumprod

        alpha_t = alpha_prod_t / alpha_prod_t_prev

        # 5. compute variance: "sigma_t(η)" -> see formula (16)
        # σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
        variance = scheduler._get_variance(t, prev_timestep)
        std_dev_t = eta * variance ** (0.5)
        eps_coef_in_step = (1 - alpha_prod_t_prev - std_dev_t ** 2) ** (0.5)

        ##This is the exact formulation from the paper but it is not equivalent to the DDPM when eta=1
        coef = (torch.sqrt((1 - alpha_prod_t) / alpha_t) - eps_coef_in_step) * torch.sqrt(1 - alpha_prod_t)
        # coef = (torch.sqrt( (1 - alpha_prod_t) / alpha_t )  - eps_coef_in_step) * eps_coef_in_step

        # coef *= get_time_based_scale(scheduler, t)
        return coef

    else:
        return 1.0


def is_local():
    # Check for a directory that only exists locally
    # return os.path.exists("/home/cgebhard/Downloads")
    return os.path.exists("/home/chris/Downloads")


def get_fixed_exp_image_data(file_path, base_directory):
    data = load_json(file_path)
    data = data["data"]
    for item in data:
        item['image_url'] = base_directory + "/" + item['image_url']

    return data


def get_feed_exp_image_data(file_path, base_directory, output_directory):
    data = load_json(file_path)
    for image_data in data:
        rel_img_path = image_data["relative_path"]
        image_data['image_path'] = base_directory + "/" + rel_img_path
        image_data['output_path'] = output_directory + "/" + "/".join(rel_img_path.split("/")[0:-1])

    return data


def load_json(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def resize_and_center_image(img_tensor, size=512, is_centered=True):
    """
    Resize the image tensor on its long edge to be size px,
    and then center the image to be size x size with black padding.

    Args:
    img_tensor (Tensor): The image tensor to resize and center.

    Returns:
    Tensor: The resized and centered image tensor.
    """
    # Determine the long edge
    original_height, original_width = img_tensor.shape[1], img_tensor.shape[2]
    long_edge = max(original_height, original_width)

    # Calculate the resize scale
    scale = size / long_edge

    # Calculate new size, ensuring long edge is size
    new_height = int(original_height * scale)
    new_width = int(original_width * scale)

    # Resize the image
    resized_img = functional.resize(img_tensor, [new_height, new_width], antialias=True)

    if is_centered:
        # Calculate padding to center the image
        pad_height = (size - new_height) // 2
        pad_width = (size - new_width) // 2

        # Apply padding
        resized_img = functional.pad(resized_img, padding_mode='constant', fill=0, padding=[pad_width, pad_height,
                                     size - new_width - pad_width, size - new_height - pad_height])

    return resized_img


def get_prompt_embeddings_sd(
        pipe, prompt: str, negative_prompt: str, do_cfg: bool = False) -> (torch.Tensor, torch.Tensor):
    # Encode prompts
    num_images_per_prompt = 1

    (prompt_embeds, negative_prompt_embeds) = pipe.encode_prompt(
        prompt=prompt,
        device=pipe._execution_device,
        num_images_per_prompt=num_images_per_prompt,
        do_classifier_free_guidance=do_cfg,
        negative_prompt=negative_prompt,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        lora_scale=None
    )

    if do_cfg:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

    return prompt_embeds


def get_prompt_embeddings_sdxl(
        pipe, prompt: str, negative_prompt: str,
        do_cfg: bool = False, batch_size: int = 1) -> (torch.Tensor, torch.Tensor, torch.Tensor):
    # Parameters
    height = pipe.default_sample_size * pipe.vae_scale_factor
    width = pipe.default_sample_size * pipe.vae_scale_factor
    original_size = (height, width)
    target_size = (height, width)
    num_images_per_prompt = 1
    cross_attention_kwargs = None
    crops_coords_top_left = (0, 0)
    device = pipe._execution_device

    # 0. Default height and width to unet
    height = pipe.default_sample_size * pipe.vae_scale_factor
    width = pipe.default_sample_size * pipe.vae_scale_factor

    # 1. Check inputs. Raise error if not correct
    pipe.check_inputs(
        prompt,  # prompt
        None,  # prompt_2
        height,
        width,
        1,  # callback_steps
        negative_prompt,  # negative_prompt
        None,  # negative_prompt_2
        None,  # prompt_embeds
        None,  # negative_prompt_embeds
        None,  # pooled_prompt_embeds
        None,  # negative_pooled_prompt_embeds
    )

    # 3. Encode input prompt
    text_encoder_lora_scale = (
        cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
    )
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds,
    ) = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=None,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        do_classifier_free_guidance=do_cfg,
        negative_prompt=negative_prompt,
        negative_prompt_2=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
        lora_scale=text_encoder_lora_scale,
    )

    # 7. Prepare added time ids & embeddings
    add_text_embeds = pooled_prompt_embeds
    add_time_ids = get_add_time_ids(
        pipe, original_size, crops_coords_top_left, target_size, dtype=prompt_embeds.dtype
    )

    if do_cfg:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        add_text_embeds = torch.cat([negative_pooled_prompt_embeds, add_text_embeds], dim=0)
        add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

    prompt_embeds = prompt_embeds.to(device)
    add_text_embeds = add_text_embeds.to(device)
    add_time_ids = add_time_ids.to(device).repeat(batch_size * num_images_per_prompt, 1)
    added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

    return prompt_embeds, added_cond_kwargs


def get_add_time_ids(pipe, original_size, crops_coords_top_left, target_size, dtype):
    """
    Function to ensure version compatability of SDXL
    :param pipe: pipeline
    :param original_size: original image size
    :param crops_coords_top_left: top left of image
    :param target_size: target image size
    :param dtype: torch.dtype of tensors
    :return:
    """
    try:
        add_time_ids = pipe._get_add_time_ids(
            original_size, crops_coords_top_left, target_size, dtype=dtype,
            text_encoder_projection_dim=pipe.text_encoder_2.config.projection_dim)
    except TypeError as e:
        add_time_ids = pipe._get_add_time_ids(
            original_size, crops_coords_top_left, target_size, dtype=dtype)

    return add_time_ids


def fit_time_distance(time, dis, ref_dis=None, do_plot=True):
    time = np.asarray(time)
    dis = np.asarray(dis)
    other = [] if ref_dis is None else [np.asarray(ref_dis)]

    try:
        popt = curve_fit(exponential_func, time, dis, p0=(1, 0.1, 0.1))
        print(f"Exp Function: f(t) = {popt[0][0]} * exp({popt[0][1]} * t) + {popt[0][2]}")
        points_fit = exponential_func(time, popt[0][0], popt[0][1], popt[0][2])
        other.append(points_fit)
    except RuntimeError:
        pass

    if do_plot:
        plot_utils.plot_value_over_time(time, dis, other)


def exponential_func(t, a, b, c):
    return a * np.exp(b * t) + c

def create_timestamp_folder_name():
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
