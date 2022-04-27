import os
import re
import zipfile
import urllib.request
from collections import defaultdict
from typing import Dict, Any

import torch
import torchvision
from torch.utils.data import Dataset


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
        walls = read_image(self._items[idx]['wall'])

        return {'image': image, 'rooms': rooms, 'walls': walls}

