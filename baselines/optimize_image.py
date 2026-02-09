import os
import torch
import yaml
import clip
import numpy as np
from tqdm.auto import tqdm
from torchvision import transforms
from scipy.optimize import minimize

from utils import plot_imgs_tensor, has_display
from datasets.CocoCaptions import CocoCaptions
from datasets.InstagramDataset import InstagramDataset


def optimize_images(data_loader, params, initialize, objective_function, device, output_path, output_transform,
                    eval_params, is_gradient_free=False, verbose=False, learning_rate=0.1, num_steps=100,
                    is_save_output=True, save_orig_img=True, show_results=True):
    ix = 0
    for image, image_path in data_loader:
        image = image.to(device)
        if isinstance(data_loader.dataset, CocoCaptions) or isinstance(data_loader.dataset, InstagramDataset):
            image_path = image_path[1]

        print(f"[ {ix + 1} / {len(data_loader.dataset)} ]: {image_path[0]} \n")
        ix += 1
        if ix > 500:
            break

        for adaptation, obj_params in params.items():
            image = image.detach()
            obj_params["clf"].is_minimized = True if adaptation != "max" else False

            # initialization
            x0, obj_params = initialize(image, obj_params)
            if "alpha" in obj_params:
                obj_params["target"] = get_condition_from_alpha(obj_params["alpha"], obj_params["clf"], image)
                del obj_params["alpha"]

            # optimization
            if is_gradient_free:
                x_opt = optimization_gradient_free(x0, obj_params, objective_function, verbose=verbose)
            else:
                x_opt = optimization(x0, obj_params, objective_function, verbose=verbose,
                                     learning_rate=learning_rate, num_steps=num_steps)

            # output transformation and evaluation
            print("\n" + adaptation)
            image_save, image_adapted = output_transform(
                image, x_opt, obj_params, eval_params, adaptation, image_path[0])
            if has_display() and show_results:
                plot_imgs_tensor(torch.cat((image_save, image_adapted), dim=0), ["original", adaptation])

            if is_save_output:
                if isinstance(data_loader.dataset, InstagramDataset):
                    # relative_path = "/".join(image_path[0].split("OriginalPosts", 1)[-1].split("/")[0:-1])
                    # output_path_tmp = output_path + relative_path

                    output_path_tmp = output_path + "/" + adaptation
                    if not os.path.exists(output_path_tmp):
                        os.makedirs(output_path_tmp)
                else:
                    output_path_tmp = output_path

                save_output(output_path_tmp, image_path[0], image_save, image_adapted, adaptation, save_orig_img)


def optimization(x0, params, objective_function, learning_rate=0.1, lr_rampdown_length=0.25,
                 lr_rampup_length=0.05, num_steps=100, verbose=False):
    x_opt = x0.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([x_opt], betas=(0.9, 0.999), lr=learning_rate)

    best_loss = torch.inf
    first_loss = 0
    best_step = 0
    best_x = x_opt.clone().detach()
    # run optimization loop
    bar = tqdm(total=num_steps, desc="Optimization")
    for step in range(num_steps):
        # Learning rate schedule.
        t = step / num_steps
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        loss = objective_function(x_opt, **params)
        if loss < best_loss:
            best_loss = loss
            best_step = step
            best_x = x_opt.clone().detach()
        if step == 0:
            first_loss = loss

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        msg = f'[ step {step+1:>4d}/{num_steps}] '
        msg += f'[ loss: {float(loss):<5.4f}] '
        msg += f'[ lr: {float(lr):<5.4f}] '
        if verbose: print(msg)
        bar.update()

    print(f'[ step {best_step+1:>4d}/{num_steps}] [ best loss: {float(best_loss):<5.4f}]'
          f' [ first loss: {float(first_loss):<5.4f}]')
    return best_x


def read_yaml(yaml_file_path):
    with open(yaml_file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data


def save_output(directory, image_path, image_orig_tensor, image_adapted_tensor, adaptation_flag, save_orig_img=True):
    to_pil = transforms.ToPILImage()
    image_name = image_path.split("/")[-1].replace(".jpg", "")
    if save_orig_img:
        orig_img_path = f'{directory}/{image_name}.jpg'
        img_adapted_path = f'{directory}/{image_name}_{adaptation_flag}.jpg'
        to_pil(image_orig_tensor[0]).save(orig_img_path)
    else:
        img_adapted_path = f'{directory}/{image_name}.jpg'

    to_pil(image_adapted_tensor[0]).save(img_adapted_path)


def get_condition_from_alpha(alpha, clf, img):
    condition = clf.predict_loss_metric(img)
    condition += torch.ones(condition.shape).to(condition.device) * alpha
    # Clip the values of the tensor between 0 and 1 (feasible range of valence and arousal
    return torch.clamp(condition, min=0.0, max=1.0)


def optimization_gradient_free(x0, params, objective_function, verbose=False):
    x0 = x0.detach().cpu().flatten().numpy()
    args = (params, objective_function, verbose)
    result = minimize(wrapper_objective_function, x0, args=args, options={'disp': True}, method='Nelder-Mead')

    print("Function value at the optimum:", result.fun)
    print("Number of iterations:", result.nit)
    print("Number of function evaluations:", result.nfev)
    print(result.message)
    return torch.from_numpy(result.x).to(torch.device("cuda"))


num_func_eval = 1
def wrapper_objective_function(x_opt, params, objective_function, verbose):
    x_opt = torch.from_numpy(x_opt).to(torch.device("cuda"))
    loss = objective_function(x_opt, **params).item()

    if verbose:
        global num_func_eval
        print(f'[{num_func_eval}] [loss:{loss: 3.6f}]')
        num_func_eval += 1

    return loss


CLIP_MODEL = None
def compute_clip_loss(image1, image2):
    global CLIP_MODEL

    # If needed, resize the images to 224x224 using interpolation
    transforms_list = [transforms.Resize((224, 224))]
    if image1.min() >= 0:
        transforms_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))

    clip_transform = transforms.Compose(transforms_list)

    # Since your images are already normalized to [-1, 1], you don't need further normalization
    # Just make sure they are resized correctly and add batch dimension
    image1_preprocessed = clip_transform(image1)
    image2_preprocessed = clip_transform(image2)

    # Get image features
    # with torch.no_grad():
    if CLIP_MODEL is None:
        CLIP_MODEL, _ = clip.load("ViT-B/32", device=image1.device)

    image1_features = CLIP_MODEL.encode_image(image1_preprocessed)
    image2_features = CLIP_MODEL.encode_image(image2_preprocessed)

    # Normalize features
    image1_features = image1_features / image1_features.norm(dim=-1, keepdim=True)
    image2_features = image2_features / image2_features.norm(dim=-1, keepdim=True)

    # Compute cosine similarity
    cosine_similarity = (image1_features * image2_features).sum(dim=-1)[0]

    # Compute CLIP loss as 1 - cosine similarity
    return 1 - cosine_similarity
