from abc import abstractmethod
import random
from typing import Dict

import torch
import torchvision.transforms.functional as TF


class AbstractTransform:

    @abstractmethod
    def apply(self, input: Dict[str, torch.Tensor]) -> Dict[str,  torch.Tensor]:
        pass


class RandomHorizontalFlip(AbstractTransform):

    def __init__(self, p: float):
        self.p = p

    def apply(self, input: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if random.random() < self.p:
            input = {k: TF.hflip(v) for k, v in input.items()}

        return input


class RandomVerticalFlip(AbstractTransform):

    def __init__(self, p: float):
        self.p = p

    def apply(self, input: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if random.random() < self.p:
            input = {k: TF.vflip(v) for k, v in input.items()}

        return input


class RandomRotation(AbstractTransform):

    _angles = [0, 90, 180, 270]

    def apply(self, input: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        r_angle = random.choice(self._angles)
        n_rot90 = r_angle // 90
        input = {k: torch.rot90(input=v, k=n_rot90, dims=[1, 2]) for k, v in input.items()}
        return input


class Resize(AbstractTransform):

    def __init__(self, size: int):
        self.size = size

    def apply(self, input: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        input = {k: TF.resize(v, [self.size, self.size], interpolation=TF.InterpolationMode.NEAREST) \
                    for k, v in input.items()}
        return input


class Compose(AbstractTransform):

    def __init__(self, *args: AbstractTransform):
        self.transforms = args

    def apply(self, input: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for tr in self.transforms:
            input = tr.apply(input)
        return input
