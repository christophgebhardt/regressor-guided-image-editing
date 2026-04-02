import sys
import torch

sys.path.append('../../emotion-adaptation')
from models.EmotionPredictionModel import get_emo_pred_transform
from losses.ValenceArousalLoss import ValenceArousalLoss


transform = None
model_va = None
device = None
def predict_valence_arousal(image):
    global model_va, transform, device
    if model_va is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        transform = get_emo_pred_transform(input_size=480, crop_size=448)
        model_va = ValenceArousalLoss("../../emotion-adaptation/trained_models/va_pred_all", device, 1.0,
                                      is_input_range_0_1=False)

    image_tensor = transform(image).to(device).unsqueeze(0)
    with torch.no_grad():
        va_label = model_va.predict_loss_metric(image_tensor).detach().cpu()

    valence = va_label[0, 0].item()
    arousal = va_label[0, 1].item()

    return valence, arousal
