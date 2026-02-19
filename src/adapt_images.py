#! /usr/bin/env -S uv run adapt_images.py

import torch
from pathlib import Path

from adapt_images.config import GuidanceConfig, AdaptConfig
from adapt_images.scoring import ImageScorer
from adapt_images.adapter import ImageAdapter
from adapt_images.output import OutputImageManager
from guidance_classifier.ValenceArousalMidu import ValenceArousalMidu
from pipelines.InversionResamplingStableDiffusionPipeline import InversionResamplingStableDiffusionPipeline
from datasets.CocoCaptions import CocoCaptions

from paths import MODELS_DIR, COCO_DIR, OUT_DIR

VA_MODEL = MODELS_DIR / "clf_new_params_midu_va_512_2024_07_11_09_10_03"
OUTPUT_PATH = OUT_DIR / "adapted_images"


def main():
    adapt_config = AdaptConfig()
    guidance_config = GuidanceConfig(clf_scale=0.2)

    pipe = create_pipeline(adapt_config, VA_MODEL)
    scorer = ImageScorer(pipe)
    output_manager = OutputImageManager(scorer, OUTPUT_PATH)
    adapter = ImageAdapter(pipe, scorer)
    dataset = CocoCaptions(COCO_DIR, "val", None)

    adapt_images(
        dataset = dataset,
        adapter = adapter,
        output_manager = output_manager,
        guidance_config = guidance_config,
        end_iteration = adapt_config.end_iteration
    )



def create_pipeline(config: AdaptConfig, model_path: Path):
    pipe = InversionResamplingStableDiffusionPipeline(
        num_inference_steps = config.num_inference_steps,
        num_inversion_steps = config.num_inversion_steps,
        normalize_gradient = config.normalize_gradient,
        scheduler_type = config.scheduler_type,
    )

    pipe.guidance_classifier = ValenceArousalMidu(
        pipe = pipe.pipe,
        device = pipe.device,
        ckp_path = f"{model_path}")

    return pipe



def adapt_images(
        dataset,
        adapter: ImageAdapter,
        output_manager: OutputImageManager,
        guidance_config,
        end_iteration
    ):

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

    for ix, (_, data) in enumerate(data_loader):
        print(f"[ {ix + 1} / {len(data_loader.dataset)} ]: {data[0][0]}")
        caption = data[2][0].split("/")[0]

        adapter.adapt(
            image_path = data[1][0],
            config = guidance_config,
            output_manager = output_manager,
            end_iteration = end_iteration,
            caption = caption            
        )



if __name__ == '__main__':
    main()
