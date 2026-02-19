import torch
import torch.nn as nn
from torch import Tensor
from diffusers import StableDiffusionXLPipeline

from guidance_classifier.GuidanceClassifier import GuidanceClassifier
from pipelines.diff_utils import get_prompt_embeddings_sd, get_prompt_embeddings_sdxl


class MiduClassifier(GuidanceClassifier):
    """
    Base class for guidance classifier
    """

    def __init__(self, pipe, device: str, ckp_path: str = None, num_outputs: int = 1, is_minimized: bool = True,
                 is_sdxl: bool = False):
        """
        Constructor
        :param device: device on which loss is computed (cpu or cuda:x)
        :param pipe: diffusion pipe
        :param ckp_path: path to checkpoint to load
        :param num_outputs: number of outputs of midu classifier
        :param is_minimized: flag indicating if optimized towards minimizing or maximizing metric of interest
        :param is_sdxl: flag indicating if classifier is trained for stable diffusion (False) or its XL version (True)
        """
        super(MiduClassifier, self).__init__(device)
        self.pipe = pipe
        self.is_minimized = is_minimized
        self.pipe.unet.mid_block.register_forward_hook(self.__hook_fn)
        self.model = self._create_midu_classifier(self.device, num_outputs, is_sdxl)
        if ckp_path is not None:
            self.model.load_state_dict(torch.load(ckp_path))
            self.model.eval()
        self.criterion = nn.MSELoss()
        self.reference_value = None

    def forward(self, latents: Tensor, t: float, prompt_embeds: Tensor = None) -> Tensor:
        """
        forward propagation of loss
        :param latents: tensor with noisy latents
        :param t: timestep
        :param prompt_embeds: text prompt embeddings
        :return: tensor with loss
        """
        # input new latents into unet
        self._set_midu_layer(latents, t, prompt_embeds)

        # calculate aesthetic loss
        return self._calculate_score(self.pipe.unet.mid_block.output.to(torch.float32), self.model,
                                     self.device, self.is_minimized, self.reference_value)

    def get_loss(self, latents: Tensor, labels: Tensor, t: Tensor, prompts: list) -> (Tensor, Tensor):
        """
        Predicts the metric based on which the loss is computed for a batch of images
        :param latents: tensor with noisy latents
        :param labels: tensor with labels
        :param t: timestep
        :param prompts: list with positive and negative prompts
        :return:
        """
        self._set_midu_layer_no_grad(latents, t, prompts)
        outputs = self.model(self.pipe.unet.mid_block.output.to(torch.float32))
        loss = self.criterion(outputs, labels)
        return loss, outputs

    def predict_score(self, latents: Tensor, t: Tensor, prompts: list) -> Tensor:
        """
        Predicts the metric based on which the loss is computed for a batch of images
        :param latents: tensor with noisy latents
        :param t: timestep
        :param prompts: list with positive and negative prompts
        :return:
        """
        self._set_midu_layer_no_grad(latents, t, prompts)
        with torch.no_grad():
            outputs = self.model(self.pipe.unet.mid_block.output.to(torch.float32))
        return outputs

    def _set_midu_layer_no_grad(self, latents, t, prompts):
        """
        Set midu layer with no grad and converting prompts to prompt_embeds
        :param latents: tensor with noisy latents
        :param t: timestep
        :param prompts: list with positive and negative prompts
        :return:
        """
        with torch.no_grad():
            if isinstance(self.pipe, StableDiffusionXLPipeline):
                prompt_embeds, added_cond_kwargs = get_prompt_embeddings_sdxl(
                    self.pipe, prompts[0], prompts[1], batch_size=latents.shape[0])
                prompt_embeds = [prompt_embeds, added_cond_kwargs]
            else:
                prompt_embeds = get_prompt_embeddings_sd(self.pipe, prompts[0], prompts[1])

            self._set_midu_layer(latents, t, prompt_embeds)

    def _set_midu_layer(self, latents: Tensor, t: float, prompt_embeds: Tensor):
        """
        Set midu layer
        :param latents: tensor with noisy latents
        :param t: timestep
        :param prompt_embeds: text prompt embeddings
        :return:
        """
        latents = self.pipe.scheduler.scale_model_input(latents, t)
        # input new latents into unet
        if isinstance(self.pipe, StableDiffusionXLPipeline):
            latents = latents.to(prompt_embeds[0].dtype)
            self.pipe.unet(
                latents, t,
                encoder_hidden_states=prompt_embeds[0],
                cross_attention_kwargs=None,
                added_cond_kwargs=prompt_embeds[1])
        else:
            self.pipe.unet(latents, t, encoder_hidden_states=prompt_embeds)

    @staticmethod
    def __hook_fn(module, input, output):
        module.output = output

    @staticmethod
    def _create_midu_classifier(device, num_outputs=10, is_sdxl=False):
        # create the midu classifier
        if is_sdxl:
            m = nn.Sequential(
                nn.Conv2d(1280, 512, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),  # Reduces to [512, 16, 16]
                nn.Conv2d(512, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),  # Reduces to [256, 8, 8]
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),  # Reduces to [128, 4, 4]
                nn.Conv2d(128, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),  # Reduces to [64, 2, 2]
                # nn.AdaptiveAvgPool2d((2, 2)),  # Ensures output [64, 2, 2]
                nn.Flatten(),
                nn.Linear(64 * 2 * 2, 128),
                nn.ReLU(),
                nn.Linear(128, num_outputs)
            )
        else:
            m = nn.Sequential(
                nn.Conv2d(1280, 256, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.ReLU(),
                # nn.MaxPool2d(2, 2),
                nn.AdaptiveAvgPool2d(output_size=(2, 2)),
                nn.Flatten(),
                nn.Linear(128 * 4, 64),
                nn.ReLU(),
                nn.Linear(64, num_outputs)
                # nn.Linear(64, 32),
                # nn.ReLU(),
                # nn.Linear(32, num_outputs)
            )
        return m.to(device)

    @staticmethod
    def _calculate_score(x, m, device, is_minimized=True, reference_value=None):
        """
        Define the aesthetic loss (lower = better)
        :param x: latents
        :param m: model
        :param device: device
        :param is_minimized: flag indicating if optimized towards minimizing or maximizing valence and arousal
        :param reference_value: for valence arousal score calculation
        :return:
        """
        return Tensor(0)
