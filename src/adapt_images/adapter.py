import PIL
import torch

from adapt_images.config import GuidanceConfig
from adapt_images.output import OutputImageManager
from pipelines.InversionResamplingDiffusionPipeline import InversionResamplingDiffusionPipeline

class ImageAdapter:
    def __init__(self, pipe: InversionResamplingDiffusionPipeline, scorer):
        self.pipe = pipe
        self.scorer = scorer

    def adapt(
            self,
            image_path, 
            config: GuidanceConfig, 
            output_manager: OutputImageManager,
            end_iteration: int,
            caption: str = ""
        ):
        
        # load image
        image_name = image_path.split("/")[-1].replace(".jpg", "")
        input_image = PIL.Image.open(image_path)
        if input_image.mode != "RGB":
            input_image = input_image.convert("RGB")

        # score original image
        orig_img_score = self.scorer.score(input_image)
        self.scorer.print_score(orig_img_score, "original")

        # reference value from alpha
        if config.reference_value is not None:
            ref_val_matrix = torch.ones(orig_img_score.shape).to(orig_img_score.device) * config.reference_value
            config.reference_value = orig_img_score + ref_val_matrix
            torch.clamp(config.reference_value, min=0.0, max=1.0).to(self.pipe.device)

        # give information about current image to output manager
        output_manager.set_image_name(image_name)
        output_manager.set_orig_image_score(orig_img_score)
        output_manager.set_orig_image(input_image)

        # adapt image
        self.pipe.revert_and_sample(
            image = input_image,
            caption = caption,
            end_iteration = end_iteration,
            params = config_to_dict(config),
            callback_resampling = None,
            callback_outputs = output_manager.callback
        )


def config_to_dict(config: GuidanceConfig):
    '''
    used to convert a GuidanceConfig to a dict,
    because the diffusion pipline has not been refactored
    yet.
    '''
    return {
        f"{config.label}": {
            "clf_scale": config.clf_scale,
            "reference_value": config.reference_value,
            "prompt": config.prompt,
            "negative_prompt": config.negative_prompt,
            "cfg_scale": config.cfg_scale,
            "use_caption": config.use_caption,
            "is_nto": config.is_nto,
            "max": config.max
        }
    }