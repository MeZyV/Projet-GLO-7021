from torchvision import transforms
import torch
import skimage.transform

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        T = transforms.ToTensor()
        image = T(image)
        if not landmarks is 0:
            landmarks = torch.tensor(landmarks,dtype=torch.float32)
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

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        
        img = skimage.transform.resize(image, (new_h, new_w))
        
        landmarks[:,:2] = landmarks[:,:2] * [new_w / w, new_h / h]
        landmarks = landmarks.astype(int)
        return (img, landmarks)
        

class ToGray(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, landmarks_bool=True):
        self.landmarks_bool = landmarks_bool

    def __call__(self, sample):

        if self.landmarks_bool:
            image, landmarks = sample
        else:
            image = sample

        if image.size()[0] > 1:
            G = transforms.Grayscale(num_output_channels=1)
            image = G(image)

        if self.landmarks_bool:
            return image, landmarks
        else:
            return image




class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image."""

    def __call__(self, sample):
        image, landmarks = sample
        col = transforms.ColorJitter(brightness=(0.5,1.4))
        image = col(image)
        return (image, landmarks)   



class GaussianBlur(object):
    """Blurs image with randomly chosen Gaussian blur"""

    def __call__(self, sample):
        image, landmarks = sample
        gaus = transforms.GaussianBlur(9, sigma=(1, 1.8))
        image = gaus(image)
        return (image, landmarks)