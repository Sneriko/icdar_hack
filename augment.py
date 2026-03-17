import random

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

from params import AUGMENTATION_PROBABILITY


def brightness(image, factor=None):
    """
    Adjust brightness.
    """
    factor = factor or 0.75 + 0.5 * random.random()
    return ImageEnhance.Brightness(image).enhance(factor)


def contrast(image, factor=None):
    """
    Adjust contrast.
    """
    factor = factor or 0.5 + random.random()
    return ImageEnhance.Contrast(image).enhance(factor)


def bleedthrough(image):
    """
    Simulate ink bleedthrough.

    Blends the image with a blurred and reversed (left-to-right) version
    of itself.
    """
    blurfactor = random.randint(3, 6)
    blend = blur(image.transpose(Image.FLIP_LEFT_RIGHT), blurfactor)
    alpha = 0.5 * random.random()  # 0 to 0.5
    blended = Image.blend(image, blend, alpha)
    contrastfactor = 1 + random.random() * 0.5  # 1 to 1.5
    return contrast(blended, contrastfactor)


def blur(image, factor=None):
    """
    Blur image.
    """
    factor = factor or random.randint(1, 2)
    return image.filter(ImageFilter.GaussianBlur(factor))


def noise(image, factor=None):
    """
    Add gaussian noise.
    """
    factor = factor or random.randint(0, 35)
    image_ = np.asarray(image)
    noise_ = np.random.normal(0, factor, image_.shape)
    noisy_image = np.clip(image_ + noise_, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_image)


def swap_colorspace(image):
    """
    Swap BGR <-> RGB
    """
    image = np.asarray(image)
    return Image.fromarray(image[:, :, ::-1])


def augment(image):
    """
    Augment an image.
    """
    augmentations = [brightness, contrast, bleedthrough, blur, noise, swap_colorspace]
    random.shuffle(augmentations)
    for augmentation in augmentations:
        if random.random() < AUGMENTATION_PROBABILITY:
            image = augmentation(image)
    return image
