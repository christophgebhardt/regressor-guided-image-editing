from PIL import Image
import numpy as np
import cv2
from skimage import measure


def calculate_colorfulness(image_path):
    """
    Calculates the colorfulness of an image based on "David Hasler and Sabine E Suesstrunk. Measuring colorfulness in
    natural images. In Human vision and electronic imaging VIII, volume 5007, pages 87–96. International Society for
    Optics and Photonics, 2003."
    """
    # Open the image file
    image = Image.open(image_path)
    # Convert the image to LAB color space channels
    l_channel, a_channel, b_channel = _get_lab_channels(_convert_to_lab(image))

    # Calculate the mean and standard deviation of the a* and b* channels
    a_mean, a_std = np.mean(a_channel), np.std(a_channel)
    b_mean, b_std = np.mean(b_channel), np.std(b_channel)

    # Compute the mean and standard deviation of the euclidean distance
    # from the mean of a* and b* channels
    a_diff = a_channel - a_mean
    b_diff = b_channel - b_mean
    color_diff = np.sqrt(a_diff ** 2 + b_diff ** 2)
    mean_color_diff = np.mean(color_diff)
    std_color_diff = np.std(color_diff)

    # Combine the statistics into the colorfulness metric
    colorfulness = std_color_diff + 0.3 * mean_color_diff

    return colorfulness


def compute_mean_brightness(image_path):
    """
    Computes mean brightness as metric following "GANalyze: Toward Visual Definitions of Cognitive Image Properties"
    """
    # Open the image file
    image = Image.open(image_path)

    # Convert image to grayscale
    grayscale_image = image.convert('L')  # 'L' mode is for grayscale

    # Convert grayscale image to a numpy array
    np_image = np.array(grayscale_image)

    # Calculate the average pixel value
    average_brightness = np.mean(np_image)

    return average_brightness


def compute_mean_saturation(image_path):
    """
    Computes the mean of saturation as mentioned in "Measuring colorfulness in natural images."
    """
    image = Image.open(image_path).convert('HSV')
    np_image = np.array(image)
    saturation = np_image[:, :, 1]
    mean_saturation = np.mean(saturation)
    return mean_saturation


def compute_rms_contrast(image_path):
    """
    Calculates the root mean square contrast as mentioned in Eli Peli, "Contrast in complex images,"
    J. Opt. Soc. Am. A 7, 2032-2040 (1990).
    """
    image = Image.open(image_path).convert('L')
    np_image = np.array(image)
    rms_contrast = np.std(np_image)
    return rms_contrast


def compute_lighting_diversity(image_path):
    """
    Calculating lighting diversity as the standard deviation of the l-channel of a LAB image.
    """
    image = Image.open(image_path).convert('RGB')
    l_channel, _, _ = _get_lab_channels(_convert_to_lab(image))
    return np.std(l_channel)


def compute_blur_effect(image_path):
    """
    Computes the no-reference perceptual blur metric as described in Crete et al. “The blur effect: perception and
    estimation with a new no-reference perceptual blur metric” Proc. SPIE 6492, Human Vision and Electronic Imaging XII.
    https://scikit-image.org/docs/stable/auto_examples/filters/plot_blur_effect.html
    """
    # Open the image using PIL
    image = Image.open(image_path).convert('RGB')  # Ensure it's in RGB format

    # Convert the image to grayscale for blur computation
    grayscale_image = np.array(image.convert('L'))  # 'L' mode is for grayscale

    # Compute the blur effect using skimage
    blur_metric = measure.blur_effect(grayscale_image)

    return blur_metric


def _convert_to_lab(image):
    # Convert to RGB if image is not already in that mode
    if image.mode != 'RGB':
        image = image.convert('RGB')
    # Convert the PIL image to a NumPy array
    np_image = np.array(image)
    # Convert from RGB to LAB color space using OpenCV
    lab_image = cv2.cvtColor(np_image, cv2.COLOR_RGB2LAB)
    return lab_image


def _get_lab_channels(lab_image):
    # Extract the a* and b* channels
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    return l_channel, a_channel, b_channel
