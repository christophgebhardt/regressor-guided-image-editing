import torch


def valence_arousal_score(predicted_va, device, is_minimized=True, reference_value=None):
    """
    Define the valence arousal score
    :param predicted_va: predicted valence and arousal
    :param device: device
    :param is_minimized: flag indicating if optimized towards minimizing or maximizing valence and arousal
    :param reference_value: for score calculation
    :return:
    """
    if reference_value is not None:
        target = reference_value
    else:
        target = torch.ones(predicted_va.size(0), 2).to(device)
        if is_minimized:
            target[:, 0] = 0.5 * target[:, 0]
            target[:, 1] = 0.0 * target[:, 1]

    error = (target - predicted_va).squeeze().squeeze()
    return torch.sum(error * error)


def arousal_score(predicted_arousal, device, is_minimized=True, reference_value=None):
    """
    Define the arousal score
    :param predicted_arousal: predicted arousal
    :param device: device
    :param is_minimized: flag indicating if optimized towards minimizing or maximizing arousal
    :param reference_value: for score calculation
    :return:
    """
    if reference_value is not None:
        target = reference_value
    else:
        # high arousal
        target = torch.ones(predicted_arousal.size(0)).to(device)
        if is_minimized:
            # low arousal:
            target = 0.0 * target

    # access arousal only if model predicts arousal and valence
    predicted_arousal = predicted_arousal[:, 1] if predicted_arousal.size(1) > 1 else predicted_arousal
    error = (target - predicted_arousal).squeeze().squeeze()
    return error * error


def valence_score(predicted_valence, device, is_minimized=True, reference_value=None):
    """
    Define the valence score
    :param predicted_valence: predicted valence
    :param device: device
    :param is_minimized: flag indicating if optimized towards minimizing or maximizing valence
    :param reference_value: for score calculation
    :return:
    """
    # access arousal only if model predicts arousal and valence

    if reference_value is not None:
        target = reference_value
    else:
        # high valence
        target = torch.ones(predicted_valence.size(0)).to(device)
        if is_minimized:
            # neutral valence:
            target = 0.5 * target
            # target = 0.0 * target

    # access valence only if model predicts arousal and valence
    predicted_valence = predicted_valence[:, 0] if predicted_valence.size(1) > 1 else predicted_valence
    error = (target - predicted_valence).squeeze().squeeze()
    return error * error
