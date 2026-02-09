import torch
from PIL.Image import Image
from torch import Tensor, dtype
import torchvision.transforms as transforms
from diffusers import (DDIMScheduler, DPMSolverMultistepScheduler,
                       DDIMInverseScheduler, DPMSolverMultistepInverseScheduler)

from tqdm.auto import tqdm


class InversionResamplingDiffusionPipeline:
    """
    Base class defining functionality of a diffusion pipeline that uses a deterministic sampler to invert images to
    their noisy latents using Null-text embeddings and edits them via using classifier- or classifier-free guidance.
    """

    def __init__(self, num_inference_steps: int, num_inversion_steps: int = None, input_size: int = 512,
                 device: str = 'cuda', normalize_gradient: bool = True):
        self.num_inference_steps = num_inference_steps
        self.num_inversion_steps = num_inference_steps if num_inversion_steps is None else num_inversion_steps
        self.device = device
        self.normalize_gradient = normalize_gradient
        self.transform = transforms.Compose([
            transforms.Resize(input_size, antialias=True),
            transforms.CenterCrop((input_size, input_size)),
            transforms.ToTensor()
        ])

        self.to_pil = transforms.ToPILImage()
        self.guidance_classifier = None
        self.pipe = None
        self.scheduler_type = "ddim"
        self.config = None

        self._is_null_text_opt = False
        self.uncond_embeddings_list = None
        self.pivot_latents = []

    @property
    def is_null_text_opt(self):
        return self._is_null_text_opt

    @is_null_text_opt.setter
    def is_null_text_opt(self, value):
        self._is_null_text_opt = value
        # set scheduler type to ddim if null text optimization as it is only working with ddim
        scheduler_type = "ddim" if value else self.scheduler_type
        self._initialize_scheduler(scheduler_type, self.config)

    def _initialize_scheduler(self, scheduler_type, config=None):
        config = self.pipe.scheduler.config if config is None else config
        if scheduler_type == "ddim":
            # DDIMScheduler, num_inference_steps: 50 - 100
            self.pipe.scheduler = DDIMScheduler.from_config(config)
        else:
            # DPMSolverMultistepScheduler, num_inference_steps: 25 - 50
            # Using sde-dpmsolver++ (stochastic) as algorithm_type for sampling results in different image,
            # use dpmsolver++ (deterministic and default)
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(config)
        self.pipe.scheduler.set_timesteps(self.num_inference_steps)

    def _initialize_inverse_scheduler(self, config=None):
        config = self.pipe.scheduler.config if config is None else config
        # inverse scheduler needs to be reinitialized for every run, otherwise it produces artifacts
        # (probably due to drift which is a known limitation).
        if isinstance(self.pipe.scheduler, DDIMScheduler):
            inverse_scheduler = DDIMInverseScheduler.from_config(config)
        else:
            inverse_scheduler = DPMSolverMultistepInverseScheduler.from_config(config)

        inverse_scheduler.set_timesteps(self.num_inversion_steps)
        return inverse_scheduler

    def revert_and_sample(self, image: Image, caption: str, end_iteration: int, params: dict,
                          guidance_classifier: object = None, callback_resampling=None,
                          callback_outputs=None) -> (Image, dict):
        end_iteration = end_iteration if end_iteration is not None else self.num_inversion_steps
        start_iteration = (0 if self.num_inference_steps != self.num_inversion_steps
                           else self.num_inference_steps - end_iteration)
        guidance_classifier = guidance_classifier if guidance_classifier is not None else self.guidance_classifier
        image = self.transform_image(image)

        # as it is null-text inversion, prompts are empty
        latents = self.reverse_sample(image, "", "", end_iteration)
        # img_noise = diff_utils.decode_to_pil(pipe, latents)[0]
        # img_noise.save('img_noise.jpg')

        if callback_resampling is not None:
            # just uses caption and no guidance scale
            # image_inv = self.sample(caption, "", latents, start_iteration, guidance_scale=1.0)
            image_inv = self.sample("", "", latents, start_iteration, guidance_scale=1.0)
            callback_resampling(image_inv)

        output_images = {}
        self.uncond_embeddings_list = None
        null_text_guidance_scale = -1
        for key, params in params.items():
            self.guidance_classifier.is_minimized = False if "max" in params and params["max"] else True
            self.guidance_classifier.reference_value = params["reference_value"]
            if params["reference_value"] is not None:
                print(f"reference value: {params['reference_value']}")
            prompt = params["prompt"] if not params["use_caption"] else caption + " " + params["prompt"]

            # recompute optimized null-text embeddings if cfg guidance scale changed from previous setting
            if "is_nto" in params and params["is_nto"] and null_text_guidance_scale != params["cfg_scale"]:
                null_text_guidance_scale = params["cfg_scale"]
                self.is_null_text_opt = True
                self.uncond_embeddings_list = self.null_optimization(null_text_guidance_scale, caption)
            elif "is_nto" not in params or not params["is_nto"]:
                self.is_null_text_opt = False
                self.uncond_embeddings_list = None
                null_text_guidance_scale = -1

            output_images[key] = self.sample(prompt, params["negative_prompt"], latents, start_iteration,
                                             guidance_scale=params["cfg_scale"],
                                             guidance_clf_scale=params["clf_scale"],
                                             guidance_classifier=guidance_classifier)

            if callback_outputs is not None:
                callback_outputs(output_images[key], key)

        return self.to_pil(image), output_images

    def _null_optimization(self, guidance_scale, prompt_embeds, added_cond_kwargs=None,
                           num_inner_steps=10, epsilon=1e-5, verbose=False):
        """
        Null-text optimization code taken from
        https://github.com/google/prompt-to-prompt/#null-text-inversion-for-editing-real-images.

        Using the NTO default implementation for SDXL does not work as gradients of the UNET flush due to its float16
        precision. To perform NTO on float32 and the SDXL UNET inference on float16, I followed the Automatic Mixed
        Precision tutorial of PyTorch: https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html#adding-gradscaler

        :param guidance_scale: Classifier-free guidance scale
        :param prompt_embeds: Prompt embeddings
        :param added_cond_kwargs: Added conditional embeddings
        :param num_inner_steps: Number of inversion steps per inference iteration
        :param epsilon: Epsilon parameter specifying the optimization stopping condition
        :param verbose: Flag indicating if method prints information to stdout
        :return:
        """
        # Prepare timesteps
        self.pipe.scheduler.set_timesteps(self.num_inference_steps, device=self.pipe._execution_device)

        uncond_embeddings, cond_embeddings = prompt_embeds.chunk(2)
        uncond_cond_kwargs, cond_cond_kwargs = None, None
        is_sdxl = False
        if added_cond_kwargs is not None:
            uncond_cond_kwargs = self.get_added_kwargs_cond_uncond(added_cond_kwargs, is_cond=False)
            cond_cond_kwargs = self.get_added_kwargs_cond_uncond(added_cond_kwargs, is_cond=True)
            is_sdxl = True

        uncond_embeddings_list = []
        latent_cur = self.pivot_latents[-1]

        bar = tqdm(total=num_inner_steps * self.num_inference_steps)
        for i in range(self.num_inference_steps):

            uncond_embeddings = uncond_embeddings.clone().detach()
            base_lr = 1e-2
            if is_sdxl:
                uncond_embeddings = uncond_embeddings.to(torch.float32)
                base_lr = 1e-1
            uncond_embeddings.requires_grad = True

            optimizer = torch.optim.Adam([uncond_embeddings], lr=base_lr * (1. - i / 100.))
            criterion = torch.nn.MSELoss()
            scaler = torch.cuda.amp.GradScaler(enabled=is_sdxl)
            # scaler = torch.amp.GradScaler('cuda', enabled=is_sdxl)

            latent_prev = self.pivot_latents[len(self.pivot_latents) - i - 2]
            t = self.pipe.scheduler.timesteps[i]
            with torch.no_grad():
                noise_pred_cond = self.pipe.unet(
                    latent_cur, t, encoder_hidden_states=cond_embeddings, added_cond_kwargs=cond_cond_kwargs)["sample"]

            j = 0
            for j in range(num_inner_steps):
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=is_sdxl):
                    noise_pred_uncond = self.pipe.unet(latent_cur, t,
                                                       encoder_hidden_states=uncond_embeddings,
                                                       added_cond_kwargs=uncond_cond_kwargs)["sample"]
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    latents_prev_rec = prev_step(self.pipe, noise_pred, t.item(), latent_cur)
                    # loss is float32 because ``mse_loss`` layers ``autocast`` to float32.
                    loss = criterion(latents_prev_rec, latent_prev)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                # loss.backward()
                # optimizer.step()
                loss_item = loss.item()

                bar.update()
                if verbose:
                    print(f"Step {j}: loss = {loss_item}")
                if loss_item < epsilon + i * 2e-5:
                    break

            for j in range(j + 1, num_inner_steps):
                bar.update()

            if is_sdxl:
                uncond_embeddings = uncond_embeddings.to(torch.float16)
            uncond_embeddings_list.append(uncond_embeddings[:1].detach())

            with torch.no_grad():
                context = torch.cat([uncond_embeddings, cond_embeddings])
                latents_input = torch.cat([latent_cur] * 2)
                noise_pred = self.pipe.unet(latents_input, t, encoder_hidden_states=context,
                                            added_cond_kwargs=cond_cond_kwargs)["sample"]
                noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
                latent_cur = prev_step(self.pipe, noise_pred, t.item(), latent_cur)

        bar.close()
        return uncond_embeddings_list

    def reverse_sample(self, image: Tensor, prompt: str, negative_prompt: str, end_iteration: int = None) -> Tensor:
        raise NotImplementedError

    def null_optimization(self, guidance_scale, caption, num_inner_steps=10, epsilon=1e-5, verbose=False):
        raise NotImplementedError

    def sample(self, prompt: str, negative_prompt: str, latents: Tensor, start_iteration: int = None,
               guidance_scale: float = 7.5, guidance_clf_scale: float = 0.0, guidance_rescale: float = 0.0,
               guidance_classifier: object = None, is_initial_noise: bool = False) -> Image:
        raise NotImplementedError

    def get_latents_from_img(self, image: Tensor, dtype_inst: dtype = None) -> Tensor:
        raise NotImplementedError

    def transform_image(self, image: Image) -> Tensor:
        image = self.transform(image)
        # image = diff_utils.resize_and_center_image(image)
        return image

    @staticmethod
    def rescale_noise_cfg(noise_cfg: Tensor, noise_pred_text: Tensor, guidance_rescale: float = 0.0) -> Tensor:
        """
        Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
        Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
        """
        std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
        std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
        # rescale the results from guidance (fixes overexposure)
        noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
        # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
        noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
        return noise_cfg

    @staticmethod
    def get_added_kwargs_cond_uncond(added_cond_kwargs, is_cond, is_clone=True):
        ix = 1 if is_cond else 0
        new_cond_kwargs = {
            "text_embeds": added_cond_kwargs["text_embeds"][ix].clone().detach().unsqueeze(0),
            "time_ids": added_cond_kwargs["time_ids"][ix].clone().detach().unsqueeze(0)
        }

        if is_clone:
            for key in new_cond_kwargs.keys():
                new_cond_kwargs[key] = new_cond_kwargs[key].clone()

        return new_cond_kwargs


def prev_step(pipe, model_output: torch.FloatTensor, timestep: int, sample: torch.FloatTensor):
    prev_timestep = timestep - pipe.scheduler.config.num_train_timesteps // pipe.scheduler.num_inference_steps
    alpha_prod_t = pipe.scheduler.alphas_cumprod[timestep]
    alpha_prod_t_prev = pipe.scheduler.alphas_cumprod[
        prev_timestep] if prev_timestep >= 0 else pipe.scheduler.final_alpha_cumprod
    beta_prod_t = 1 - alpha_prod_t
    pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    pred_sample_direction = (1 - alpha_prod_t_prev) ** 0.5 * model_output
    prev_sample = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction
    return prev_sample
