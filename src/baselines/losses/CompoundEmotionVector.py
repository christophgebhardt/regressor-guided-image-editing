import torch
import numpy as np
from torch import Tensor


emotions_type = torch.tensor([11.0, 7.0, 9.0, 5.0, 13.0, 15.0, 3.0, 1.0]) * 0.125 * np.pi


class CompoundEmotionVector:
    """
    Compound emotion vector as described in "A Circular-Structured Representation for Visual Emotion Distribution
    Learning". Class computes properties over batch of data. Class assumes that columns of tensors are ordered in the
    following way: Amusement Awe Contentment Excitement Anger Disgust Fear Sadness (order of ground_truth.txt of
    LDL datasets [http://47.105.62.179:8081/sentiment_web/datasets/LDL.tar.gz])
    """

    def __init__(self, polarity: Tensor = None, theta: Tensor = None, intensity: Tensor = None,
                 emo_type: Tensor = None):
        """
        Constructor
        :param polarity: polarity of compound emotion vector
        :param theta: theta of compound emotion vector
        :param intensity: intensity of compound emotion vector
        :param emo_type: tensor describing the angles of the 8 basic emotions. Assumes that columns of
        tensors are ordered in the following way: Amusement Awe Contentment Excitement Anger Disgust Fear Sadness
        (order of ground_truth.txt of LDL datasets [http://47.105.62.179:8081/sentiment_web/datasets/LDL.tar.gz])
        """
        self.polarity = polarity
        self.theta = theta
        self.intensity = intensity
        self.emo_type = emo_type

    def init_from_emotions_distribution(self, emotions: Tensor):
        """
        Initializes compound emotion vector from batch of emotions distributions (tensor).
        :param emotions: batch of emotions distributions (tensor)
        :return: self
        """
        self.polarity, self.theta, self.intensity = compute_compound_emotion_vector(emotions, self.emo_type)
        return self

    def __str__(self):
        """
        Returns string representation of CompoundEmotionVector
        :return: sting representation
        """
        return "polarity: {}\ntheta: {}\nintensity: {}".format(self.polarity, self.theta, self.intensity)

    @staticmethod
    def init_compound_emotions_vector(emotions: Tensor, emo_type: Tensor):
        """
        Initializes compound emotion vector from batch of emotion vectors (:,3) or emotion distribution (:,8) (tensor).
        :param emotions: tensor with emotion vectors (:,3) or emotion distribution (:,8)
        :param emo_type: tensor describing the angles of the 8 basic emotions. Assumes that columns of
        tensors are ordered in the following way: Amusement Awe Contentment Excitement Anger Disgust Fear Sadness
        (order of ground_truth.txt of LDL datasets [http://47.105.62.179:8081/sentiment_web/datasets/LDL.tar.gz])
        :return: instance of CompoundEmotionVector
        """
        if emotions.shape[1] == 3:
            return CompoundEmotionVector(emotions[:, 0], emotions[:, 1], emotions[:, 2], emo_type)
        else:
            return CompoundEmotionVector(emo_type=emo_type).init_from_emotions_distribution(emotions)


def compute_compound_emotion_vector(emotions: Tensor, emo_type: Tensor = None) -> (Tensor, Tensor, Tensor):
    """
    Computes polarity, theta, intensity of compound emotion vector for batch of emotions distributions (tensor).
    :param emo_type: tensor describing the angles of the 8 basic emotions. Assumes that columns of
    tensors are ordered in the following way: Amusement Awe Contentment Excitement Anger Disgust Fear Sadness
    (order of ground_truth.txt of LDL datasets [http://47.105.62.179:8081/sentiment_web/datasets/LDL.tar.gz])
    :param emotions: batch of emotions distributions (tensor)
    :return: tuple of tensors describing polarity, theta, intensity of compound emotion vector
    """
    if emo_type is None:
        emo_type = emotions_type
    emo_x_j = emotions * torch.cos(emo_type)
    emo_y_j = emotions * torch.sin(emo_type)
    emo_x = torch.sum(emo_x_j, dim=1)
    emo_y = torch.sum(emo_y_j, dim=1)
    intensity = torch.sqrt(emo_x * emo_x + emo_y * emo_y)
    theta_atan2 = torch.atan2(emo_y, emo_x)
    theta = torch.remainder(theta_atan2, 2 * np.pi)
    polarity = (torch.abs(theta_atan2) > np.pi / 2.0).float()

    return polarity, theta, intensity
