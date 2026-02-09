import torch
from tqdm.auto import tqdm
from diffusers import StableDiffusionXLPipeline

from diff_utils import decode_to_pil, get_prompt_embeddings_sdxl
from InversionResamplingDiffusionPipeline import InversionResamplingDiffusionPipeline


class InversionResamplingStableDiffusionXLPipeline(InversionResamplingDiffusionPipeline):
    """
    Class managing functionality of a diffusion pipeline that uses a deterministic sampler to invert images to their
    noisy latents using Null-text embeddings and then can edit them via using classifier- or classifier-free guidance.
    """

    def __init__(self, num_inference_steps: int, num_inversion_steps: int = None, pipe_path: str = "stabilityai/stable-diffusion-xl-base-1.0",
                 device: str = 'cuda', scheduler_type: str = 'dpm', normalize_gradient: bool = True):
        # Load the SD pipeline and add a hook
        super().__init__(num_inference_steps, num_inversion_steps, 1024, device, normalize_gradient)
        self.pipe = StableDiffusionXLPipeline.from_pretrained(pipe_path, torch_dtype=torch.float16, variant="fp16",
                                                              use_safetensors=True).to(device)
        if scheduler_type == 'dpm':
            # adjusted based on suggestion of
            # https://huggingface.co/docs/diffusers/en/api/pipelines/stable_diffusion/stable_diffusion_xl
            # explanation can be found here:
            # https://github.com/huggingface/diffusers/pull/5541
            # euler_at_final=True/False allows trading off the numerical stability and sample details.
            # use_karras_sigmas=True results in timestep duplicates in DPMSolverMultistepInverseScheduler
            # which are then automatically removed, resulting in lower number of inversion steps than specified.
            self.config = dict(self.pipe.scheduler.config)
            self.config["use_karras_sigmas"] = True
            self.config["use_lu_lambdas"] = True
            # self.config["euler_at_final"] = True
        else:
            self.config = self.pipe.scheduler.config

        # pipe.enable_model_cpu_offload()
        self._initialize_scheduler(scheduler_type, self.config)
        self.scheduler_type = scheduler_type

    def reverse_sample(self, image, prompt, negative_prompt, end_iteration=None):
        with torch.no_grad():
            inverse_scheduler = self._initialize_inverse_scheduler(self.config)
            end_iteration = self.num_inversion_steps if end_iteration is None else end_iteration

            prompt_embeds, added_cond_kwargs = get_prompt_embeddings_sdxl(self.pipe, prompt, negative_prompt)
            latents = self.get_latents_from_img(image, prompt_embeds.dtype)

            # Loop through the sampling timesteps
            self.pivot_latents.append(latents)
            for i, t in tqdm(enumerate(inverse_scheduler.timesteps)):
                if i == end_iteration:
                    break

                latents = inverse_scheduler.scale_model_input(latents, t)
                # predict the noise residual
                noise_pred = self.pipe.unet(
                    latents,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=None,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                latents = inverse_scheduler.step(noise_pred, t, latents).prev_sample
                self.pivot_latents.append(latents)

            return latents

    def sample(self, prompt, negative_prompt, latents, start_iteration=None, guidance_scale=7.5,
               guidance_clf_scale=0.0, guidance_rescale=0.0, guidance_classifier=None, is_initial_noise=False):
        # Parameters
        start_iteration = 0 if start_iteration is None else start_iteration
        height = self.pipe.default_sample_size * self.pipe.vae_scale_factor
        width = self.pipe.default_sample_size * self.pipe.vae_scale_factor
        batch_size = 1
        num_images_per_prompt = 1

        # 2. Define call parameters
        denoising_end = None
        generator = None
        cross_attention_kwargs = None
        eta = 0.0
        device = self.pipe._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0
        do_classifier_guidance = guidance_classifier is not None and guidance_clf_scale > 0.0

        # 3. Prepare prompt embeddings
        prompt_embeds, added_cond_kwargs = get_prompt_embeddings_sdxl(
            self.pipe, prompt, negative_prompt, do_classifier_free_guidance)

        # 4. Prepare timesteps
        self.pipe.scheduler.set_timesteps(self.num_inference_steps, device=device)
        timesteps = self.pipe.scheduler.timesteps

        # 5. Prepare latent variables
        if is_initial_noise:
            num_channels_latents = self.pipe.unet.config.in_channels
            latents = self.pipe.prepare_latents(
                batch_size * num_images_per_prompt,
                num_channels_latents,
                height,
                width,
                prompt_embeds.dtype,
                device,
                generator,
                latents,
            )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.pipe.prepare_extra_step_kwargs(generator, eta)
        prompt_embeds_cfg = prompt_embeds.detach().clone()

        # 8. Denoising loop
        num_warmup_steps = max(len(timesteps) - self.num_inference_steps * self.pipe.scheduler.order, 0)

        # 7.1 Apply denoising_end
        if denoising_end is not None and type(denoising_end) == float and denoising_end > 0 and denoising_end < 1:
            discrete_timestep_cutoff = int(
                round(
                    self.pipe.scheduler.config.num_train_timesteps
                    - (denoising_end * self.pipe.scheduler.config.num_train_timesteps)
                )
            )
            self.num_inference_steps = len(list(filter(lambda ts: ts >= discrete_timestep_cutoff, timesteps)))
            timesteps = timesteps[:self.num_inference_steps]

        with self.pipe.progress_bar(total=self.num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if i < start_iteration:
                    continue

                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, t)

                if prompt_embeds_cfg.shape[0] == 2 and self.uncond_embeddings_list is not None:
                    prompt_embeds_cfg[0] = self.uncond_embeddings_list[i]

                # predict the noise residual
                with torch.no_grad():
                    noise_pred = self.pipe.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds_cfg,
                        cross_attention_kwargs=cross_attention_kwargs,
                        added_cond_kwargs=added_cond_kwargs,
                        return_dict=False,
                    )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = self.rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.pipe.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if do_classifier_guidance:
                    # Set requires grad
                    latents = latents.detach().requires_grad_()
                    # Note: the first element in prompt_embeds is the null-text embedding the second the text embedding
                    prompt_embeds_clf = prompt_embeds[0].unsqueeze(0) if do_classifier_free_guidance else prompt_embeds
                    added_cond_kwargs_clf = self.get_added_kwargs_cond_uncond(
                        added_cond_kwargs, is_cond=False, is_clone=False)

                    # get classifier loss
                    loss = guidance_classifier(latents, t, [prompt_embeds_clf, added_cond_kwargs_clf])
                    # get gradient
                    cond_grad = torch.autograd.grad(loss, latents)[0]
                    # norm_avg += torch.linalg.norm(cond_grad.detach())

                    if self.normalize_gradient:
                        # normalize (add a small epsilon to avoid division by zero)
                        cond_grad /= (cond_grad.norm() + 1e-10)

                    # modify the latents based on this gradient
                    # sigma = diff_utils.get_scheduler_guidance_scaling(self.pipe.scheduler, t)
                    latents = latents.detach() - guidance_clf_scale * cond_grad

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.pipe.scheduler.order == 0):
                    progress_bar.update()

        # make sure the VAE is in float32 mode, as it overflows in float16
        if self.pipe.vae.dtype == torch.float16 and self.pipe.vae.config.force_upcast:
            self.pipe.upcast_vae()
            latents = latents.to(next(iter(self.pipe.vae.post_quant_conv.parameters())).dtype)

        return decode_to_pil(self.pipe, latents)[0]

    def get_latents_from_img(self, image, dtype):
        image = self.pipe.image_processor.preprocess(image)
        image = image.to(device=self.pipe._execution_device, dtype=dtype)

        # make sure the VAE is in float32 mode, as it overflows in float16
        if self.pipe.vae.config.force_upcast:
            image = image.float()
            self.pipe.vae.to(dtype=torch.float32)

        latents = self.pipe.vae.encode(image).latent_dist.sample()

        if self.pipe.vae.config.force_upcast:
            self.pipe.vae.to(dtype)

        latents = latents.to(dtype)
        return self.pipe.vae.config.scaling_factor * latents

    def null_optimization(self, guidance_scale, caption, num_inner_steps=10, epsilon=1e-5, verbose=False):
        # Prepare prompt embeddings
        prompt_embeds, added_cond_kwargs = get_prompt_embeddings_sdxl(self.pipe, caption, "", do_cfg=True)

        return self._null_optimization(
            guidance_scale, prompt_embeds, added_cond_kwargs, num_inner_steps, epsilon, verbose)
