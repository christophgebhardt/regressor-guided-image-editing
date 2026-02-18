import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np


class PandasFrameDataset(Dataset):
    """
    Dataset class for LDL dataset annotated with emotion distribution.
    """

    def __init__(self, img_labels, img_dir, transform, do_normalize_per_row=False, img_file_ix=0):
        """
        Constructor
        :param img_labels: pandas dataframe of images and labels
        :param img_dir: directory of images
        :param transform: torchvision.transforms.Compose
        :param do_normalize_per_row: flag indicating if dataframe is normalized per row (default=False)
        :param img_file_ix: index of column in dataframe that specifies image path
        """
        self.img_names = img_labels.iloc[:, img_file_ix].values.tolist()
        self.img_labels = self.label_transform(img_labels.iloc[:, img_file_ix+1:], do_normalize_per_row)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        """
        Length function of dataset
        :return: length of dataset
        """
        return len(self.img_names)

    def __getitem__(self, idx):
        """
        Get function of dataset
        :param idx: index of sample to return
        :return: image and label (emotion distribution [8])
        """
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = Image.open(img_path)
        if image.mode != "RGB":
            image = image.convert('RGB')

        image_tensor = self.transform(image)
        return image_tensor, self.img_labels[idx, :]

    @staticmethod
    def label_transform(label_df, do_normalize_per_row):
        """
        Computes an emotion distribution from the label dataframe
        :param label_df: pandas dataframe
        :param do_normalize_per_row: flag indicating if dataframe is normalized per row (one row = distribution)
        :return: tensor with emotion distribution
        """
        # transform columns that are of type string/object to class labels
        for column in label_df:
            if label_df[column].dtype == "object":
                unique_instances = label_df[column].unique()
                label_df[column] = label_df[column].map(lambda p: np.where(unique_instances == p)[0][0])

        label = torch.from_numpy(np.float32(label_df.values))
        if do_normalize_per_row:
            label /= torch.sum(label, dim=1)[:, None]
        return label


def __sanity_check_dataset(dataset):
    for i in range(len(dataset)):
        image, label = dataset[i]
        if not torch.is_tensor(image):
            print(image)
        if not torch.is_tensor(label) or label.shape[0] != 8:
            print(label)
