import os
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class ImageDirectoryDataset(Dataset):
    def __init__(self, image_dir, transform=None, used_in_torch_fidelity=False, is_recursive=False):
        self.image_dir = image_dir
        self.transform = transforms.ToTensor() if transform is None else transform
        self.used_in_torch_fidelity = used_in_torch_fidelity
        if is_recursive:
            self.image_files = get_files_from_subdirectories(image_dir)
        else:
            self.image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.used_in_torch_fidelity:
            return image.to(torch.uint8)

        return image, img_name


def get_files_from_subdirectories(base_directory):
    # List of supported image file extensions
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

    # Initialize an empty list to store image file paths
    image_files = []

    # Recursively go through all subdirectories and add image files to the list
    for dirpath, _, filenames in os.walk(base_directory):
        for filename in filenames:
            # Check if the file has an image extension
            if os.path.splitext(filename)[1].lower() in image_extensions:
                # Add the full path of the image file to the list
                image_files.append(os.path.join(dirpath, filename))

    return image_files