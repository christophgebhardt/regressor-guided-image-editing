import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class ListDataset(Dataset):

    def __init__(self, root, list_imgs, transform=None):
        """
        Constructor
        :param root: Coco root folder
        :param list_imgs: specifies relative paths from root where images to be found.
        :param transform: torchvision.transforms.Compose (default = None)
        """
        self.root_dir = root
        self.transform = transform if transform is not None else transforms.Compose([transforms.ToTensor()])
        self.image_list = list_imgs

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_path = "{}{}".format(self.root_dir, self.image_list[index])
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert('RGB')

        image = self.transform(image).to(dtype=torch.uint8)
        return image
