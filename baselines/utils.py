import torch
import os
import argparse
import datetime
from torchvision import transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datasets.PandasFrameDataset import PandasFrameDataset
from datasets.LDLDatasetPCLabeled import LDLDatasetPCLabeled
from torchvision.utils import make_grid

from PIL import Image, ImageDraw


def has_display():
    return bool(os.environ.get('DISPLAY', None))


def get_torch_device(device_id: str):
    if device_id == "cpu":
        return torch.device("cpu")

    if device_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = device_id

    # Detect if we have a GPU available
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def initialize_arg_parser():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-bs", "--batch_size", type=int, default=16, help="Batch size for training (default = 16)")
    arg_parser.add_argument("-ne", "--num_epochs", type=int, default=50,
                            help="Number of epochs to train for (default = 50)")
    arg_parser.add_argument("-d", "--device", type=str, default=None,
                            help="Setting the torch device on which the script is supposed to run. Use the respective "
                            "gpu id (e.g., '0') to specify gpu usage. Use the id 'cpu' for cpu usage. If id is set to "
                            "None it will take either cuda:0 or cpu (default = None)")
    arg_parser.add_argument("-lr", "--learning_rate", type=float, default=1e-5,
                            help="Learning rate of the optimizer (default = 1e-5)")
    arg_parser.add_argument("-nw", "--num_workers", type=int, default=4,
                            help="Number of workers used in dataloader, 0 = no parallelization (default = 4)")
    return arg_parser


def update_arg_default(arg_parser, argument, default_value):
    for action in arg_parser._actions:
        if action.dest == argument:
            action.default = default_value
            return


def get_model_path(directory, base_name, timestamp_str, args_str):
    return "{}/{}_{}_{}".format(directory, base_name, timestamp_str, args_str)


def get_str_timestamp():
    return datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")


def initialize_emotional_data(dataset_id, batch_size, num_workers, transform, phases=None, shuffle=True):
    data_dir = "exploration"
    if phases is None:
        phases = ["train", "val"]

    # Data augmentation and normalization for training, validation, and test (same for all at the moment)
    data_transforms = {}
    for phase in phases:
        data_transforms[phase] = transform

    print("Initializing Datasets and Dataloaders...")
    img_labels_dict = create_dataset_splits(dataset_id, phases, data_dir)

    dataloaders_dict = {}
    for phase in phases:
        dataset = PandasFrameDataset(img_labels_dict[phase], data_dir, data_transforms[phase], img_file_ix=1)
        dataloaders_dict[phase] = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloaders_dict


def create_dataset_splits(dataset_id, phases, data_dir):
    dataset = "all" if dataset_id != "psych" and dataset_id != "cgna" and dataset_id != "emo6" else dataset_id
    c_is = [0, 1, 2, 3, 4, 5, 6, 15]
    img_labels = pd.read_csv(os.path.join(data_dir, "EmotionalDatabases/{}.csv".format(dataset)), sep=",").iloc[:, c_is]
    img_labels_dict = {}
    if "test" in phases:
        img_labels_dict["train"], img_labels_dict["test"] = perform_val_train_split(img_labels, fraction_val_data=0.1)
    if "val" in phases:
        il = img_labels_dict["train"] if "train" in img_labels_dict else img_labels
        img_labels_dict["train"], img_labels_dict["val"] = perform_val_train_split(il, fraction_val_data=0.2)
    else:
        img_labels_dict["train"] = img_labels

    return img_labels_dict


def initialize_ldl_data(dataset_id, batch_size, num_workers, transform, is_pc_labeled=True, phases=None, shuffle=True):
    if phases is None:
        phases = ["train", "val", "test"]
    data_dir = "./data/LDL"

    # Data augmentation and normalization for training, validation, and test (same for all at the moment)
    data_transforms = {}
    for phase in phases:
        data_transforms[phase] = transform

    print("Initializing Datasets and Dataloaders...")
    img_labels = pd.read_csv(os.path.join(data_dir, "ground_truth.txt"), sep=" ")
    if dataset_id == "flickr" or dataset_id == "twitter":
        img_labels = img_labels[img_labels['Source'] == dataset_id]
    img_labels = img_labels.iloc[:, :9]

    img_labels_dict = {}
    if "test" in phases:
        img_name_test = pd.read_csv(os.path.join(data_dir, "test.txt"),
                                    sep=" ").iloc[:, 0].to_list()
        img_labels_dict["test"] = img_labels.loc[img_labels['images-name'].isin(img_name_test)]

    img_name_train = pd.read_csv(os.path.join(data_dir, "train.txt"),
                                 sep=" ").iloc[:, 0].to_list()
    img_labels = img_labels.loc[img_labels['images-name'].isin(img_name_train)]
    if "val" in phases:
        img_labels_dict["train"], img_labels_dict["val"] = perform_val_train_split(img_labels, fraction_val_data=0.2)
    else:
        img_labels_dict["train"] = img_labels

    dataloaders_dict = {}
    img_dir = "{}/images".format(data_dir)
    for phase in phases:
        dataset = LDLDatasetPCLabeled(img_labels_dict[phase], img_dir, data_transforms[phase]) if is_pc_labeled else \
            PandasFrameDataset(img_labels_dict[phase], img_dir, data_transforms[phase], do_normalize_per_row=True)
        dataloaders_dict[phase] = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloaders_dict


def plot_imgs_tensor(imgs_, title=None, nrow=4):
    if title is not None:
        plt.title(title)
    plt.imshow(make_grid(imgs_, nrow=nrow).permute(1, 2, 0).detach().cpu().numpy())
    plt.show()


def plot_list_imgs_tensor(imgs_list, titles=None):
    for i in range(len(imgs_list)):
        title = None if titles is None else titles[i]
        plot_imgs_tensor(imgs_list[i], title)


def plot_value_during_training(v1_list, v2_list, v1_label, v2_label, title, x_label="iterations", y_label="loss"):
    plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.plot(v1_list, label=v1_label)
    plt.plot(v2_list, label=v2_label)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()


def write_text_to_tensor(imgs, intensity, pos=(10, 10), color=(255, 0, 0)):
    # font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSerifBold.ttf", 20, encoding="unic")
    for i in range(imgs.shape[0]):
        # Convert the PyTorch tensor to a PIL image
        pil_img = Image.fromarray(imgs[i, :, :, :].permute(1, 2, 0).mul(255).byte().numpy())

        # Draw text on the PIL image
        draw = ImageDraw.Draw(pil_img)
        if intensity[i].ndim == 0:
            text = str(intensity[i].item())[0:4]
        else:
            text = str(intensity[i, 0].item())[0:4]
            for j in range(1, intensity[i].shape[0]):
                text += "," + str(intensity[i, j].item())[0:4]

        # draw.text(pos, text, color, font)
        draw.text(pos, text, color)
        # pil_img.show()

        # Convert the PIL image back to a PyTorch tensor
        transform = transforms.Compose([
            transforms.PILToTensor()
        ])
        imgs[i, :, :, :] = transform(pil_img).float() / 255.0


def plot_training_curves(hist, num_epochs, metric_name="Validation Accuracy"):
    """
    Plot the training curves of validation accuracy vs. number of training epochs for the transfer learning method and
    the model trained from scratch
    :param hist: histogram of losses
    :param num_epochs: number of epochs
    :param metric_name: name of metric to print
    :return:
    """
    ohist = [h.cpu().numpy() for h in hist]

    plt.title("{} vs. Number of Training Epochs".format(metric_name))
    plt.xlabel("Training Epochs")
    plt.ylabel("{}".format(metric_name))
    plt.plot(range(1, num_epochs + 1), ohist, label="Pretrained")
    plt.ylim((0, 1.))
    plt.xticks(np.arange(1, num_epochs + 1, 1.0))
    plt.legend()
    plt.show()


def perform_val_train_split(labels, fraction_val_data=0.2):
    rng = np.random.default_rng()
    idx_val = rng.choice(labels.shape[0], size=int(fraction_val_data * labels.shape[0]), replace=False)
    idx_train = [x for x in range(labels.shape[0]) if x not in idx_val]
    return labels.iloc[idx_train], labels.iloc[idx_val]


def print_column_stats(df, columns_to_check=None):
    for column in df:
        if columns_to_check is not None and column not in columns_to_check:
            continue

        # Compute and print statistics over a column
        print(f"{column} Mean: {df[column].mean():.4f}; Standard Deviation: {df[column].std():.4f}; "
              f"Minimum: {df[column].min():.4f}; Maximum: {df[column].max():.4f}")


def print_real_fake_stats(real_list, fake_list):
    print_column_stats(pd.DataFrame({'predicted metric real': real_list, 'predicted metric fake': fake_list}))


def interweave_batch_tensors(batch1, batch2):
    # Reshape the batches to have a shape of (N, C * H * W)
    reshaped_batch1 = batch1.view(batch1.size(0), -1)
    reshaped_batch2 = batch2.view(batch2.size(0), -1)

    # Interweave the rows of the batches
    merged_batch = torch.stack((reshaped_batch1, reshaped_batch2), dim=1)
    return merged_batch.view(-1, merged_batch.size(-1))


def cohen_d(x, y):
    """
    Computes cohen_d between two populations. Correct if the population S.D. is expected to be equal for the two groups.
    :param x: population 1
    :param y: population 2
    :return:
    """
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)


def get_image_files(directory, extensions=None):
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png']
    image_files = []

    for root, _, files in os.walk(directory):
        for file in files:
            _, extension = os.path.splitext(file)
            if extension.lower() in extensions:
                image_files.append(os.path.join(root, file))

    return image_files


def is_local():
    # Check for a directory that only exists locally
    # return os.path.exists("/home/cgebhard/Downloads")
    return os.path.exists("/home/chris/Downloads")


def print_stats(stats):
    for label, data in stats.items():
        stats_str = f"{label}: "
        for stat, values in data.items():
            if len(values) == 0:
                continue
            stats_str += f"{stat}: mean {np.mean(values):.4f}, std {np.std(values):.4f}; "
        print(stats_str)


def check_init_stats_adapt(stats_dict, adaptation):
    if adaptation not in stats_dict:
        stats_dict[adaptation] = {
            "valence": [], "arousal": [], "delta_valence": [], "delta_arousal": [], "rec_error": []
        }
