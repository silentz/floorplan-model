import os
import re
import zipfile
import urllib.request
from collections import defaultdict
from typing import Dict, Any

import torch
import torchvision
from torch.utils.data import Dataset
import torch.nn.functional as F

from . import transforms


_datasets = {
    'newyork': {
        'url': 'http://silentz.ml/newyork.zip',
        'arc': 'newyork.zip',
        'dst': 'newyork',
    },
}


def _ensure_dataset(root: str,
                    url: str,
                    archive: str,
                    dest: str):
    dst_path = os.path.join(root, dest)
    arc_path = os.path.join(root, archive)

    if not os.path.isdir(dst_path):
        urllib.request.urlretrieve(url, arc_path)

        if arc_path.endswith('.zip'):
            with zipfile.ZipFile(arc_path, 'r') as zip:
                zip.extractall(root)


class EnsureDatasetMixin:

    def __init__(self, root: str, dataset: str):
        _ensure_dataset(
                root=root,
                url=_datasets[dataset]['url'],
                archive=_datasets[dataset]['arc'],
                dest=_datasets[dataset]['dst'],
            )


class NewYorkDataset(EnsureDatasetMixin, Dataset):

    _rgb2roomtype = {
        0: [  0,  0,  0], # background
        1: [192,192,224], # closet
        2: [192,255,255], # bathroom
        3: [224,255,192], # livingroom/kitchen
        4: [255,224,128], # bedroom
        5: [255,160, 96], # hall
        6: [255,224,224], # balcony
        7: [224,224,224], # unused
        8: [224,224,128]  # unused
    }

    def __init__(self, root: str, subdir: str):
        super().__init__(root, 'newyork')
        self._root = os.path.join(root, subdir)

        all_files = os.listdir(self._root)
        all_items = defaultdict(lambda: dict())

        for filename in all_files:
            base_name = filename.split('.')[0]
            full_path = os.path.join(self._root, filename)

            if base_name.count('_') == 0:
                all_items[base_name]['origin'] = full_path
            if base_name.count('_') == 1:
                name, postfix = base_name.split('_')
                all_items[name][postfix] = full_path

        self._items = list(all_items.values())

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        read_image = lambda x: \
                torchvision.io.read_image(x, torchvision.io.ImageReadMode.RGB)

        image = read_image(self._items[idx]['origin'])
        rooms = read_image(self._items[idx]['rooms'])
        close = read_image(self._items[idx]['close'])
        walls = read_image(self._items[idx]['wall'])

        rooms = rooms.permute(1, 2, 0)
        plan_mask = torch.zeros(image.shape[1], image.shape[2], dtype=torch.uint8)

        for class_idx, rgb in self._rgb2roomtype.items():
            rgb = torch.tensor(rgb, dtype=torch.uint8)
            mask = (rooms == rgb)
            mask = torch.all(mask, dim=2)
            plan_mask[mask] = class_idx

        walls = torch.all(walls > 0, dim=0)
        plan_mask[walls] = 9

        close = torch.all(close > 0, dim=0)
        plan_mask[close] = 10

        result = {
                'image': image,
                'mask': plan_mask.unsqueeze(dim=0),
            }

        return result


class AugmentedNewYorkDataset(NewYorkDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.transforms = transforms.Compose(
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(),
                transforms.Resize(512),
            )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = super().__getitem__(idx)
        item = self.transforms.apply(item)
        return item

