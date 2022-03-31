import random
import torch
import torchvision.transforms.functional as F

from PIL import Image, ImageFilter
from kornia.augmentation import RandomAffine

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = F.to_tensor(image)
        if not landmarks is 0:
            landmarks = torch.tensor(landmarks, dtype=torch.float32)
        return (image, landmarks)


class Rescale(object):
    """Rescale the image in a sample to a given size.
    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample

        h, w = image.size[:2]
        new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        m = F.resize((new_h, new_w), Image.BICUBIC)
        img = m(image)

        return (img, landmarks)


class ToGray(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample
        G = F.to_grayscale(num_output_channels=1)
        if image.size[0] > 1:
            image = G(image)
        return (image, landmarks)


class GaussianBlur(object):
    """Blurs image with randomly chosen Gaussian blur"""

    def __call__(self, sample):
        image, landmarks = sample
        kernel = random.randint(1, 5)
        image = image.filter(ImageFilter.GaussianBlur(radius=kernel))
        return (image, landmarks)


def get_twin(im, degrees=(-5, 5), translate=(0.2, 0.2), scale=(1.2, 1.5), shear=(-15, 15)):
    image_tens = im.type(torch.float32)
    aug = RandomAffine(degrees=degrees, translate=translate, scale=scale, shear=shear,  return_transform=True)
    image_tens_warp, H = aug(image_tens)
    return image_tens_warp, H
