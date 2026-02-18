import torch
from datasets.PandasFrameDataset import PandasFrameDataset
from baselines.losses.CompoundEmotionVector import compute_compound_emotion_vector


class LDLDatasetPCLabeled(PandasFrameDataset):
    """
    Dataset class for LDL dataset annotated with emotion distribution as well as emotion vectors.
    """

    def __init__(self, img_labels, img_dir, transform):
        """
        Constructor
        :param img_labels: pandas dataframe of images and labels
        :param img_dir: directory of images
        :param transform: torchvision.transforms.Compose
        """
        super().__init__(img_labels, img_dir, transform)
        self.emo_vec_labels = self.pc_label_transform(self.img_labels)

    def __getitem__(self, idx):
        """
        Get function of dataset
        :param idx: index of sample to return
        :return: image and label (emotion distribution [8] and emotion vectors [3] stacked)
        """
        image, label_emo_dis = super().__getitem__(idx)
        return image, torch.cat((label_emo_dis, self.emo_vec_labels[idx, :]), dim=0)

    @staticmethod
    def pc_label_transform(img_labels):
        """
        Computes an emotion distribution from the label dataframe
        :param img_labels: tensor with emotion distribution
        :return: tensor with emotion vectors
        """
        polarity, emp_type, intensity = compute_compound_emotion_vector(img_labels)
        return torch.stack((polarity, emp_type, intensity), dim=1)
