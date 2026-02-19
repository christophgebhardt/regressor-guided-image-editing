import math
import torch
import pipelines.diff_utils as diff_utils
from tqdm.auto import tqdm
from diffusers import StableDiffusionPipeline

from pipelines.InversionResamplingDiffusionPipeline import InversionResamplingDiffusionPipeline


class InversionResamplingStableDiffusionPipeline(InversionResamplingDiffusionPipeline):
    """
    Class managing functionality of a diffusion pipeline that uses a deterministic sampler to invert images to their
    noisy latents using Null-text embeddings and then can edit them via using classifier- or classifier-free guidance.
    """

    def __init__(self, num_inference_steps: int, num_inversion_steps: int = None,
                 pipe_path: str = "stabilityai/sd-turbo", device: str = 'cuda',
                 scheduler_type: str = 'dpm', normalize_gradient: bool = True):
        # Load the SD pipeline and add a hook
        super().__init__(num_inference_steps, num_inversion_steps, device=device, normalize_gradient=normalize_gradient)
        self.pipe = StableDiffusionPipeline.from_pretrained(pipe_path).to(device)
        # pipe.enable_model_cpu_offload()
        self._initialize_scheduler(scheduler_type)
        self.scheduler_type = scheduler_type

    def reverse_sample(self, image, prompt, negative_prompt, end_iteration=None):
        with torch.no_grad():
            inverse_scheduler = self._initialize_inverse_scheduler()
            end_iteration = self.num_inversion_steps if end_iteration is None else end_iteration
            latents = self.get_latents_from_img(image)
            text_embeddings = diff_utils.get_prompt_embeddings_sd(self.pipe, prompt, negative_prompt)

            # Loop through the sampling timesteps
            self.pivot_latents.append(latents)
            for i, t in tqdm(enumerate(inverse_scheduler.timesteps)):
                if i == end_iteration:
                    break

                latents = inverse_scheduler.scale_model_input(latents, t)
                noise_pred = self.pipe.unet(latents, t, encoder_hidden_states=text_embeddings).sample

                # Code restricts timestep to ensure that next one is still within timestep range of training (1000).
                # However, does not seem to be necessary.
                # if t + inverse_scheduler.config.num_train_timesteps // self.num_inference_steps >= 1000:
                #     t = 999 - inverse_scheduler.config.num_train_timesteps // self.num_inference_steps
                latents = inverse_scheduler.step(noise_pred, t, latents).prev_sample
                self.pivot_latents.append(latents)

            return latents

    def sample(self, prompt, negative_prompt, latents, start_iteration=None, guidance_scale=7.5,
               guidance_clf_scale=0.0, guidance_rescale=0.0, guidance_classifier=None, is_initial_noise=False):
        start_iteration = 0 if start_iteration is None else start_iteration
        # Default height and width to unet
        height = self.pipe.unet.config.sample_size * self.pipe.vae_scale_factor
        width = self.pipe.unet.config.sample_size * self.pipe.vae_scale_factor
        device = self.pipe._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        do_classifier_guidance = guidance_classifier is not None and guidance_clf_scale > 0.0

        # Prepare prompt embeddings
        prompt_embeds = diff_utils.get_prompt_embeddings_sd(
            self.pipe, prompt, negative_prompt, do_classifier_free_guidance)

        # Prepare timesteps
        self.pipe.scheduler.set_timesteps(self.num_inference_steps, device=device)
        timesteps = self.pipe.scheduler.timesteps

        # Prepare latent variables
        if is_initial_noise:
            num_channels_latents = self.pipe.unet.config.in_channels
            latents = self.pipe.prepare_latents(
                latents.shape[0],
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator=None,
                latents=latents
            )

        # Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.pipe.prepare_extra_step_kwargs(generator=None, eta=0.0)
        prompt_embeds_cfg = prompt_embeds.detach().clone()

        # Denoising loop
        # norm_avg = torch.tensor(0.0).to(device)
        latents_clf = None
        for i, t in tqdm(enumerate(timesteps)):
            if i < start_iteration:
                continue

            # derived distance in latent space that latents can take to pivot (does not work)
            ref_dis = 0.0001 * math.exp(0.1 * i) - 0.00009
            # ref_dis = 0.0008 * math.exp(0.12 * i) - 0.00079

            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            if latents_clf is not None:
                latent_model_input[1, :] = latents_clf

            latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, t)

            if prompt_embeds_cfg.shape[0] == 2 and self.uncond_embeddings_list is not None:
                prompt_embeds_cfg[0] = self.uncond_embeddings_list[i]

            with torch.no_grad():
                noise_pred = self.pipe.unet(latent_model_input, t, encoder_hidden_states=prompt_embeds_cfg)[0]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = self.rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

            if do_classifier_guidance:
                # Set requires grad
                latents = latents.clone().detach().requires_grad_()
                # Note: the first element in prompt_embeds is the null-text embedding the second the text embedding
                prompt_embeds_clf = prompt_embeds[0].unsqueeze(0) if do_classifier_free_guidance else prompt_embeds
                # get classifier loss
                loss = guidance_classifier(latents, t, prompt_embeds_clf)
                # get gradient
                cond_grad = torch.autograd.grad(loss, latents)[0]
                # print(f"Norm: {cond_grad.norm()}")

                if self.normalize_gradient:
                    # normalize (add a small epsilon to avoid division by zero)
                    cond_grad /= (cond_grad.norm() + 1e-10)

                # modify the latents based on this gradient
                latents = latents.detach() - guidance_clf_scale * cond_grad

        # print(f"average norm: {(norm_avg / self.num_inference_steps).item()}")
        return diff_utils.decode_to_pil(self.pipe, latents)[0]

    def get_latents_from_img(self, image, dtype=None):
        sample_image = self.pipe.image_processor.preprocess(image).to(self.pipe._execution_device)
        latents = self.pipe.vae.encode(sample_image, return_dict=False)[0].sample()
        return self.pipe.vae.config.scaling_factor * latents

    def null_optimization(self, guidance_scale, caption, num_inner_steps=10, epsilon=1e-5, verbose=False):
        # Prepare prompt embeddings
        prompt_embeds = diff_utils.get_prompt_embeddings_sd(
            self.pipe, caption, "", do_cfg=True)

        return self._null_optimization(
            guidance_scale, prompt_embeds, num_inner_steps=num_inner_steps, epsilon=epsilon, verbose=verbose)
