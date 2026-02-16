import json
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class CocoCaptions(Dataset):

    def __init__(self, root, split, transform=None):
        """
        Constructor
        :param root: Coco root folder
        :param split: specifies the respective split of the data. Possible values are "train", "val", and "test".
        :param transform: torchvision.transforms.Compose (default = None)
        """
        ann_file = "{}/annotations/captions_{}2017.json".format(root, split)
        self.root_dir = "{}/{}2017".format(root, split)
        self.transform = transforms.ToTensor() if transform is None else transform
        self.image_list = self.get_images(ann_file)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        image_name = "{}.jpg".format(str(self.image_list[index]["id"]).zfill(12))
        image_path = "{}/{}".format(self.root_dir, image_name)
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return image, [image_name, image_path, "/".join(self.image_list[index]["captions"]).replace("\n", "")]

    @staticmethod
    def get_images(ann_file):
        with open(ann_file) as f:
            data = json.load(f)["annotations"]

        captions = {}
        for item in data:
            if item["image_id"] not in captions:
                captions[item["image_id"]] = []
            captions[item["image_id"]].append(item["caption"])

        images = []
        for key, data in captions.items():
            images.append({"id": key, "captions": data})

        return images
