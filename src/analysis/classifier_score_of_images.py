import sys
import torch
from PIL import Image

# setting path
PATH_EMO_ADAPT = '/home/cgebhard/emotion-adaptation'
sys.path.append(PATH_EMO_ADAPT)
from models.EmotionPredictionModel import get_emo_pred_transform
from losses.ValenceArousalLoss import ValenceArousalLoss
from losses.CompoundEmotionLoss import CompoundEmotionLoss


def get_classifier_score_of_images(image_paths, model_id):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    transform = get_emo_pred_transform(input_size=480, crop_size=448)

    if model_id == "emonet":
        model = ValenceArousalLoss(f"{PATH_EMO_ADAPT}/trained_models/EmoNet_valence_moments_resnet50_5_best.pth.tar",
                                   device, 1.0, loss="valence", is_input_range_0_1=False)
    elif model_id == "va":
        model = ValenceArousalLoss(f"{PATH_EMO_ADAPT}/trained_models/va_pred_all", device, 1.0,
                                   is_input_range_0_1=False)
    elif model_id == "emo_pred":
        model = CompoundEmotionLoss(f"{PATH_EMO_ADAPT}/trained_models/emo_pred_ldl", device, 1.0,
                                    loss="intensity", is_input_range_0_1=False).to(device)
    else:
        raise ValueError("Id does not exist!")

    image_tensors = torch.stack([process_image(path, transform) for path in image_paths])
    image_tensors = image_tensors.to(device)

    with torch.no_grad():
        emo_net_label = model.predict_loss_metric(image_tensors).detach().cpu()

    return emo_net_label


def process_image(image_path, transform):
    """
    Function to load an image and apply transformations
    """
    image = Image.open(image_path).convert('RGB')
    return transform(image)
