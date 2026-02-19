from __future__ import print_function
from __future__ import division
import os
import time
import copy
import datetime
import numpy as np

import torch
import wandb
import sys
from tqdm.auto import tqdm
from random import randint
from torch.utils.data import DataLoader
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, DDIMScheduler, DPMSolverMultistepScheduler
from diffusers.optimization import get_cosine_schedule_with_warmup
from torchvision import transforms

from src.clf.ClfWrapper import ClfWrapper
from guidance_classifier.ValenceLatents import ValenceLatents
from guidance_classifier.ValenceMidu import ValenceMidu
from guidance_classifier.ValenceArousalLatents import ValenceArousalLatents
from guidance_classifier.ValenceArousalMidu import ValenceArousalMidu
from guidance_classifier.IntensityLatents import IntensityLatents
from guidance_classifier.IntensityMidu import IntensityMidu

import pipelines.diff_utils as diff_utils
IS_LOCAL = diff_utils.is_local()

# setting path
sys.path.append('../emotion-adaptation')
from datasets.ImageNetKaggle import ImageNetKaggle
from datasets.CocoCaptions import CocoCaptions
from datasets.ValenceArousalDataset import ValenceArousalDataset


os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
torch.set_num_threads(1)


def main():
    custom_desc = "best"

    # Specify setting
    setting = "va"
    # setting = "emonet"
    # setting = "mikel"

    log_wandb = False
    is_midu = False
    input_size_sd = 1024
    # input_size_sd = 512
    is_sdxl = input_size_sd == 1024
    # scheduler_type = 'dpm'  # using dpm does not seem to work
    scheduler_type = 'ddim'

    # dataset_id = "coco"
    # dataset_id = "imagenet"
    dataset_id = "va"

    # compute input parameters
    input_desc = "midu" if is_midu else "latents"
    comment = (f"clf_{custom_desc}_{input_desc}_{setting}_{input_size_sd}"
               f"_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}")
    print(comment)

    device = 'cuda'
    if IS_LOCAL:
        num_workers = 0
        batch_size = 2 if input_size_sd == 512 else 1
    else:
        num_workers = 4
        # batch_size = 16 if input_size_sd == 512 else 8  # for 512 input 16 is the max
        batch_size = 8

    # 1e-5 seems to work better than 1e-3 and 1e-7
    learning_rate = 1e-5

    # config from diffusers training
    # learning_rate = 1e-4
    # lr_warmup_steps = 500

    num_epochs = 100
    # step_size = num_epochs
    # step_size = 5
    # gamma = 0.1

    # Load the SD pipeline
    if input_size_sd == 512:
        pipe_path = "stabilityai/stable-diffusion-2-1-base"
        pipe = StableDiffusionPipeline.from_pretrained(pipe_path).to(device)
    elif input_size_sd == 1024:
        pipe_path = "stabilityai/stable-diffusion-xl-base-1.0"
        pipe = StableDiffusionXLPipeline.from_pretrained(pipe_path, torch_dtype=torch.float16, variant="fp16",
                                                         use_safetensors=True).to(device)
    else:
        raise ValueError("Input size not supported")

    if scheduler_type == "ddim":
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    else:
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # initialize guidance classifier and model name
    if setting == "va":
        guidance_classifier = (ValenceArousalMidu(pipe, device, is_sdxl=is_sdxl) if is_midu else
                               ValenceArousalLatents(device, pipe.scheduler, is_sdxl=is_sdxl))
        model_name = "va_pred_all"
    elif setting == "emonet":
        model_name = "EmoNet_valence_moments_resnet50_5_best.pth.tar"
        guidance_classifier = ValenceMidu(pipe, device) if is_midu else ValenceLatents(device)
    elif setting == "mikel":
        model_name = "emo_pred_ldl"
        guidance_classifier = IntensityMidu(pipe, device) if is_midu else IntensityLatents(device)
    else:
        raise ValueError("setting does not exist")

    # off-the-shelf classifiers
    transform = transforms.Compose([
        transforms.Resize(input_size_sd, antialias=True),
        transforms.RandomCrop(input_size_sd),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    model_directory = "/home/cgebhard/emotion-adaptation/trained_models" if IS_LOCAL else "/data/cgebhard/classifiers"
    clf_wrapper = ClfWrapper(model_name, device, model_directory)

    if dataset_id == "imagenet":
        if IS_LOCAL:
            data_path = '/media/cgebhard/Data/Data/Datasets/ImageNet'
        else:
            data_path = '/data/cgebhard/ImageNet'
        constructor = ImageNetKaggle
        splits = ["train", "val", "test"]
    elif dataset_id == "coco":
        data_path = '/home/cgebhard/emotion-adaptation/coco'
        constructor = CocoCaptions
        splits = ["train", "val"]
    elif dataset_id == "va":
        if IS_LOCAL:
            data_path = '/home/cgebhard/emotion-adaptation/exploration'
        else:
            data_path = '/data/cgebhard'
        constructor = ValenceArousalDataset
        splits = ["train", "val"]
        # set clf_wrapper to None as original labels are used
        clf_wrapper = None
    else:
        raise ValueError("dataset_id does not have a valid value")

    dataloaders = {}
    for split in splits:
        print(split)

        dataloaders[split] = torch.utils.data.DataLoader(constructor(data_path, split, transform),
                                                         batch_size=batch_size, num_workers=num_workers, shuffle=True)

    optimizer = torch.optim.Adam(guidance_classifier.model.parameters(), lr=learning_rate, weight_decay=5e-5)
    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma, verbose=False)
    lr_scheduler = None

    # optimizer = torch.optim.AdamW(guidance_classifier.model.parameters(), lr=learning_rate)
    # lr_scheduler = get_cosine_schedule_with_warmup(
    #     optimizer=optimizer,
    #     num_warmup_steps=lr_warmup_steps,
    #     num_training_steps=(len(dataloaders["train"]) * num_epochs),
    # )

    guidance_classifier = train_guidance_classifier(guidance_classifier, dataloaders, optimizer, pipe, clf_wrapper,
                                                    device, num_epochs, lr_scheduler, comment, log_wandb)

    if IS_LOCAL:
        torch.save(guidance_classifier.model.state_dict(), f"trained_models/{comment}")
    else:
        torch.save(guidance_classifier.model.state_dict(), f"/data/cgebhard/classifiers/{comment}")


def train_guidance_classifier(guidance_classifier, dataloaders, optimizer, pipe, clf_wrapper, device, num_epochs=25,
                              lr_scheduler=None, comment="", log_wandb=False):
    # Initialize wandb project
    if log_wandb:
        wandb.init(
            project="Guidance Classifier",
            name=comment if comment else datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
            # config={"learning_rate": optimizer.param_groups[0]['lr'], "epochs": num_epochs}
        )

    phases = ['train', 'val']
    since = time.time()

    # define prompts
    prompts = [
        "",  # positive_prompt
        # "a professional, detailed, high-quality image",  # positive_prompt
        ""  # negative_prompt
    ]

    best_model_wts = copy.deepcopy(guidance_classifier.model.state_dict())
    best_loss = torch.inf

    train_it = 0
    val_it = 1

    # get random label to be aware of output shape
    image, rand_label = next(iter((dataloaders["train"])))
    rand_label = clf_wrapper.get_label(image.to(device)) if clf_wrapper is not None else rand_label

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        labels_list = []
        output_list = []
        if log_wandb:
            # Create data structure for wandb outputs
            for j in range(rand_label.size(1)):
                labels_list.append([[] for _ in range(pipe.scheduler.config.num_train_timesteps)])
                output_list.append([[] for _ in range(pipe.scheduler.config.num_train_timesteps)])

        # Each epoch has a training and validation phase
        for phase in phases:
            if phase == 'train':
                guidance_classifier.model.train()  # Set model to training mode
            else:
                guidance_classifier.model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_metric_loss = []
            # Create data structure for output comparison
            for j in range(rand_label.size(1)):
                running_metric_loss.append(0.0)

            # Iterate over data.
            for images, labels_orig in tqdm(dataloaders[phase], desc=f"Processing {phase}"):
                images = images.to(device)
                labels = clf_wrapper.get_label(images) if clf_wrapper is not None else labels_orig.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                if len(labels_orig) > 2 and clf_wrapper is not None:
                    # TODO: fix when using coco to include caption as prompts and negative prompts
                    prompt_list = labels_orig[1][2][0].split("/")
                    prompt_list = prompt_list[randint(0, len(prompt_list) - 1)]
                else:
                    prompt_list = [images.shape[0] * [prompts[0]], images.shape[0] * [prompts[1]]]

                noisy_latents, time_steps = get_noisy_latents(pipe, images)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    loss, outputs = guidance_classifier.get_loss(noisy_latents, labels, time_steps, prompt_list)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        train_it += 1
                    elif log_wandb:
                        val_it += 1
                        for j in range(outputs.size(1)):
                            for k in range(time_steps.size(0)):
                                output_list[j][time_steps[k].item()].append(outputs[k, j].item())
                                labels_list[j][time_steps[k].item()].append(labels[k, j].item())

                # statistics
                running_loss += loss.item() * images.size(0)
                for j in range(outputs.size(1)):
                    running_metric_loss[j] += torch.sum(torch.abs(outputs[:, j] - labels[:, j])).item()

                if phase == 'train' and (train_it - 1) % 1000 == 0:
                    epoch_loss = running_loss / (train_it * images.shape[0])
                    print_str = f"{phase} Loss: {epoch_loss:.4f} "
                    if log_wandb:
                        wandb.log({"Loss/{}".format(phase): epoch_loss}, step=train_it)

                    for j in range(outputs.size(1)):
                        epoch_mae_loss = running_metric_loss[j] / (train_it * images.shape[0])
                        if log_wandb:
                            wandb.log({"MAE/Metric{}/{}".format(j, phase): epoch_mae_loss}, step=train_it)
                        print_str += f" Metric {j} MAE: {epoch_mae_loss:.4f}"

                    if not log_wandb:
                        print(print_str)

                #     if (train_it - 1) % 5000 == 0 and train_it != 1:
                #         break
                #
                # batch_size = images.size(0)
                # if phase == 'val' and val_it % int(5000.0 / batch_size) == 0:
                #     val_it += 1
                #     break

            if phase == 'val':
                val_it = 0
                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                if log_wandb:
                    wandb.log({"Loss/{}".format(phase): epoch_loss})

                    for j in range(outputs.size(1)):
                        epoch_mae_loss = running_metric_loss[j] / len(dataloaders[phase].dataset)
                        wandb.log({"MAE/Metric{}/{}".format(j, phase): epoch_mae_loss})

                    log_prediction_stats(labels_list, "Labels", epoch)
                    log_prediction_stats(output_list, "Outputs", epoch)

                if epoch_loss < best_loss:
                    # deep copy the model
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(guidance_classifier.model.state_dict())
                    if IS_LOCAL:
                        torch.save(guidance_classifier.model.state_dict(),
                                   f"trained_models/{comment}")
                    else:
                        torch.save(guidance_classifier.model.state_dict(),
                                   f"/data/cgebhard/classifiers/{comment}")
        print()
        if lr_scheduler is not None:
            lr_scheduler.step()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    if best_loss != 0:
        print('Best loss: {:4f}'.format(best_loss))

    if log_wandb:
        wandb.finish()

    # load best model weights
    guidance_classifier.model.load_state_dict(best_model_wts)
    return guidance_classifier


def get_noisy_latents(pipe, images):
    """
    Code taken from https://huggingface.co/docs/diffusers/v0.17.1/tutorials/basic_training
    :param pipe: diffusion pipeline
    :param images: batch of images
    :return: latents with added noise, time steps of latents
    """
    with torch.no_grad():
        device = pipe._execution_device

        if isinstance(pipe, StableDiffusionXLPipeline):
            latents = get_latents_from_img_1024(pipe, images)
        else:
            latents = get_latents_from_img_1024(pipe, images)

        # Sample noise to add to the images
        noise = torch.randn(latents.shape, device=device)
        bs = latents.shape[0]

        # Sample a random timestep for each image
        time_steps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (bs,),
                                   device=device, dtype=torch.int64)

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = pipe.scheduler.add_noise(latents, noise, time_steps)
        return noisy_latents, time_steps


def get_latents_from_img_512(pipe, image):
    sample_image = pipe.image_processor.preprocess(image).to(pipe._execution_device)
    latents = pipe.vae.encode(sample_image, return_dict=False)[0].sample()
    return pipe.vae.config.scaling_factor * latents


def get_latents_from_img_1024(pipe, image):
    dtype = torch.float16
    image = pipe.image_processor.preprocess(image)
    image = image.to(device=pipe._execution_device, dtype=dtype)

    # make sure the VAE is in float32 mode, as it overflows in float16
    if pipe.vae.config.force_upcast:
        image = image.float()
        pipe.vae.to(dtype=torch.float32)

    latents = pipe.vae.encode(image).latent_dist.sample()

    if pipe.vae.config.force_upcast:
        pipe.vae.to(dtype)

    latents = latents.to(dtype)
    return pipe.vae.config.scaling_factor * latents


def log_prediction_stats(time_value_list, descriptor, epoch_num):
    for i in range(len(time_value_list)):
        desc = f"{descriptor}/Metric{i}/Epoch{epoch_num}"
        mean = []
        st_dev = []
        min_val = []
        max_val = []
        times = []
        for t in range(len(time_value_list[i])):
            if len(time_value_list[i][t]) > 0:
                mean.append(np.mean(time_value_list[i][t]))
                st_dev.append(np.std(time_value_list[i][t]))
                min_val.append(np.min(time_value_list[i][t]))
                max_val.append(np.max(time_value_list[i][t]))
                times.append(t)
            # else:
            #     mean.append(0.0)
            #     st_dev.append(0.0)
            #     min_val.append(0.0)
            #     max_val.append(0.0)

        desc = f"{descriptor}/Metric{i}/Epoch{epoch_num}"
        plot_wandb(times, [mean, min_val, max_val], ["Mean", "Min", "Max"], desc, title=f"Epoch{epoch_num}")
        desc = f"{descriptor}/Metric{i}/StDev/Epoch{epoch_num}"
        plot_wandb(times, [st_dev], ["StDev"], desc, title=f"Epoch{epoch_num}")


def plot_wandb(x_values, y_values, keys, desc, title=""):
    wandb.log({desc: wandb.plot.line_series(
        xs=x_values,
        ys=y_values,
        keys=keys,
        title=title,
        xname="time steps")})


if __name__ == '__main__':
    main()
