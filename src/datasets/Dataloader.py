import json
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class Dataloader(Dataset):

    def __init__(self, root, transform = None):
        """
        Constructor
        :param root: Coco root folder
        """
        ann_file = "{}/annotations/captions.json".format(root)
        self.root_dir = "{}/images".format(root)
        self.transform = transforms.ToTensor() if transform is None else transform
        self.image_list = self.get_images(ann_file)


    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_name = "{}".format(str(self.image_list[index]["id"]).zfill(12))
        image_path = "{}/{}".format(self.root_dir, image_name)
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert('RGB')

        image = self.transform(image)
        return image, [image_name, image_path, "/".join(self.image_list[index]["captions"]).replace("\n", "")]

    @staticmethod
    def get_images(ann_file):
        with open(ann_file) as f:
            data: dict = json.load(f)

        images = []
        for key, value in data.items():
            images.append({"id": key, "captions": [value]})

        return images
