import random
import torchvision.transforms.functional as F
import torch
from torchvision.transforms import ToTensor as _ToTensor

class Compose:
    def __init__(self, transforms): self.transforms = transforms
    def __call__(self, image, target):
        for t in self.transforms: image, target = t(image, target)
        return image, target

class ToTensor(_ToTensor):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

class RandomHorizontalFlip:
    def __init__(self, prob=0.5): self.prob = prob
    def __call__(self, image, target):
        if random.random() < self.prob:
            width, _ = image.size
            image = F.hflip(image)
            if "boxes" in target:
                bbox = target["boxes"]
                bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
                target["boxes"] = bbox
        return image, target