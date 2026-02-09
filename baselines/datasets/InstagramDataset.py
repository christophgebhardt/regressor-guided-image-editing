import json

from PIL import Image
from torch.utils.data import Dataset


class InstagramDataset(Dataset):
    """
    Dataset class for Instagram dataset annotated with captions.
    """

    def __init__(self, base_directory, transform, json_file="../diffusion-guidance/caption_files/feed_images.json"):
        """
        Constructor
        :param base_directory: base directory of dataset
        :param transform: torchvision.transforms.Compose
        :param json_file: path to json file if not in base_directory
        """
        self.image_path_list = self.get_feed_exp_image_data(json_file, base_directory)
        # self.image_path_list = self.image_path_list[20:]
        self.transform = transform

    def __getitem__(self, idx):
        """
        Get function of dataset
        :param idx: index of sample to return
        :return: image and label (captions)
        """
        image = Image.open(self.image_path_list[idx]["image_path"])
        if image.mode != "RGB":
            image = image.convert('RGB')

        image_tensor = self.transform(image)
        return image_tensor, ["/".join(self.image_path_list[idx]["captions"]), self.image_path_list[idx]["image_path"]]

    def __len__(self):
        """
        Length function of dataset
        :return: length of dataset
        """
        return len(self.image_path_list)

    @staticmethod
    def get_feed_exp_image_data(file_path, base_directory):
        with open(file_path, 'r') as file:
            data = json.load(file)
        for image_data in data:
            rel_img_path = image_data["relative_path"]
            image_data['image_path'] = base_directory + "/" + rel_img_path

        return data
