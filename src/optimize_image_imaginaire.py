#! /usr/bin/env -S uv run optimize_image_imaginaire.py

import os
import sys
import torch
from torchvision import transforms

from datasets.CocoCaptions import CocoCaptions

from baselines.losses.ValenceArousalLoss import ValenceArousalLoss
from baselines.optimize_image import optimize_images
from baselines.run_img_trans import compare_emotions
from baselines.utils import check_init_stats_adapt, print_stats

from external.imaginaire.discriminators.munit import Discriminator
from external.imaginaire.generators.munit import Generator
from external.imaginaire.losses.gan import GANLoss
from external.imaginaire.config import Config

from paths import PROJECT_ROOT, COCO_DIR, MODELS_DIR, OUT_DIR


STATS = {}
VA_MODEL = MODELS_DIR / "va_pred_all"
IMAGINAIRE_MODEL = MODELS_DIR / "imaginaire_munit_200000_s5.pt"
IMAGINAIRE_CONFIG = PROJECT_ROOT / "src/external/imaginaire/imagenet2imagenet.yaml"
OUTPUT_PATH = OUT_DIR / "imaginaire"

def main():
    # parameters
    is_gradient_free = False
    learning_rate = 0.05  # only used for gradient-based opt
    weight_clf_list = [0.2]
    # weight_clf_list = [0.1, 0.3, 0.5, 0.7, 1.0]
    weight_dis = 0.0
    weight_recon = 1.0
    params = {
        # "min": {},
        "pos_01": {"alpha": 0.1},
        "pos_02": {"alpha": 0.2},
        "neg_01": {"alpha": -0.1},
        "neg_02": {"alpha": -0.1},
        "neutral": {"alpha": 0.0}
        # "max": {}
    }
    verbose = False

    save_orig_img = False
    input_size = 1024
    crop_size = 1024
    batch_size = 1
    num_steps = 300  # 100 steps seem to be sufficient, COCO requires 300
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # parameters: end

    # inputs
    va_loss = ValenceArousalLoss(f"{VA_MODEL}", device, 1,
                                 is_input_range_0_1=False, is_minimized=True, requires_grad=not is_gradient_free)
    eval_params = {"emotion_type_labels": None}


    data_transforms = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])



    dataset_test = CocoCaptions(COCO_DIR, "val", data_transforms)
    data_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)

    cfg = Config(IMAGINAIRE_CONFIG)
    gen = Generator(cfg.gen, cfg.data)
    state_dict = torch.load(IMAGINAIRE_MODEL)
    gen_sate_dict = get_relevant_states(state_dict['net_G'])
    gen.load_state_dict(gen_sate_dict)
    gen = gen.to(device)

    dis = None
    gan_loss = None
    if weight_dis > 0:
        dis = Discriminator(cfg.dis, cfg.data)
        dis_sate_dict = get_relevant_states(state_dict['net_D'])
        dis.load_state_dict(dis_sate_dict)
        dis = dis.to(device)
        gan_loss = GANLoss(cfg.trainer.gan_mode)

    for weight_clf in weight_clf_list:
        output_path_i = f'{OUTPUT_PATH}/weight_{weight_clf:<1.2f}'

        if not os.path.exists(output_path_i):
            os.makedirs(output_path_i)

        params_default = {
            "gen": gen, "clf": va_loss, "dis": dis, "weight_clf": weight_clf, "weight_dis": weight_dis,
            "weight_recon": weight_recon, "gan_loss": gan_loss
        }

        for key in params.keys():
            params[key] = {**params[key], **params_default}

        optimize_images(data_loader, params, initialize_imaginaire, objective_function_imaginaire, device,
                        output_path_i, output_transform, eval_params, is_gradient_free, verbose, learning_rate,
                        num_steps=num_steps, save_orig_img=save_orig_img)

        print(f"weight_clf: {weight_clf}; weight_dis: {weight_dis}; weight_recon: {weight_recon}")
        print_stats(STATS)


def initialize_imaginaire(image, obj_params):
    with torch.no_grad():
        content, style = obj_params["gen"].autoencoder_a.encode(image)
    obj_params["content"] = content
    obj_params["orig_image"] = image
    return style, obj_params


def objective_function_imaginaire(x_opt, gen, orig_image, content, clf, weight_clf, weight_dis, weight_recon, dis=None,
                                  target=None, gan_loss=None):
    if len(x_opt.shape) == 1:
        x_opt = x_opt.view(1, 8, 1, 1).to(torch.float32)

    content = content.detach()
    img = gen.autoencoder_a.decode(content, x_opt)
    # image needs to be clamped as decoder tends to generate images that go beyond bounds (done in imaginaire repo too)
    img = torch.clamp(img, min=-1, max=1)

    loss = weight_clf * clf(img, target=target)

    if dis is not None and weight_dis > 0:
        out_ba, _, _ = dis.discriminator_a(img)
        dis_loss = gan_loss(out_ba, True, dis_update=False)
        # we compute the relu of the negative gan loss as imaginaire MUNIT uses Hinge loss. Hence, we want to penalize
        # negative outputs while accepting positive ones.
        loss += weight_dis * torch.relu(-dis_loss)

    if weight_recon > 0:
        # L1 reconstruction on content as used in imaginaire
        content_new, _ = gen.autoencoder_a.encode(img)
        loss += weight_recon * torch.nn.functional.l1_loss(content_new, content)


    return loss


def get_relevant_states(state_dict, use_averaged_model=False):
    # Conditional logic to keep or delete keys
    if use_averaged_model:
        # Keep only keys that contain 'averaged_model'
        filtered_dict = {k.replace("module.", ""): v for k, v in state_dict.items() if 'averaged_model' in k}
    else:
        # Delete keys that contain 'averaged_model'
        filtered_dict = {k.replace("module.", ""): v for k, v in state_dict.items() if 'averaged_model' not in k}

    if "num_updates_tracked" in filtered_dict:
        del filtered_dict["num_updates_tracked"]
    return filtered_dict


def output_transform(image, x_opt, obj_params, eval_params, adaptation, image_path=None):
    if len(x_opt.shape) == 1:
        x_opt = x_opt.view(1, 8, 1, 1).to(torch.float32)

    # generate results
    image_adapted = obj_params["gen"].autoencoder_a.decode(obj_params["content"].detach(), x_opt)
    # image needs to be clamped as decoder tends to generate images that go beyond bounds (done in imaginaire repo too)
    image_adapted = torch.clamp(image_adapted, min=-1, max=1)

    # evaluation
    check_init_stats_adapt(STATS, adaptation)
    compare_emotions(obj_params["clf"], image, image_adapted, eval_params["emotion_type_labels"], STATS[adaptation])

    return transform(image), transform(image_adapted)


def transform(image):
    return (image + 1.0) * 0.5


if __name__ == '__main__':
    main()
