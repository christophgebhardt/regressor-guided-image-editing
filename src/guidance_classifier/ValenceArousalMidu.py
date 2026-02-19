from guidance_classifier.MiduClassifier import MiduClassifier
from guidance_classifier.guidance_scores import valence_arousal_score


class ValenceArousalMidu(MiduClassifier):
    """
    Class for valence midu classifier
    """

    def __init__(self, pipe, device: str, is_minimized: bool = True, ckp_path: str = None, is_sdxl: bool = False):
        """
        Constructor
        :param pipe: Diffuser stable diffusion pipeline
        :param device: device on which loss is computed (cpu or cuda:x)
        :param is_minimized: flag indicating if optimized towards minimizing or maximizing metric of interest
        :param ckp_path: path to checkpoint to load
        :param is_sdxl: flag indicating if classifier is trained for stable diffusion (False) or its XL version (True)
        """
        super(ValenceArousalMidu, self).__init__(pipe, device, ckp_path, num_outputs=2, is_minimized=is_minimized,
                                                 is_sdxl=is_sdxl)

    @staticmethod
    def _calculate_score(x, m, device, is_minimized=True, reference_value=None):
        """
        Define the valence arousal score
        :param x: latents
        :param m: model
        :param device: device
        :param is_minimized: flag indicating if optimized towards minimizing or maximizing valence and arousal
        :param reference_value: reference_value for score calculation
        :return:
        """
        return valence_arousal_score(m(x), device, is_minimized, reference_value)
