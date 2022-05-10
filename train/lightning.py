import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid

import wandb
import pytorch_lightning as pl
from typing import Any, Dict, List

from src.collate import collate_fn, Batch
from src.model import Model
from src.utils import mask2rgb
from src.sampler import InfiniteSampler


class DataModule(pl.LightningDataModule):

    def __init__(self, train_dataset: Dataset,
                       train_batch_size: int,
                       train_num_workers: int,
                       val_dataset: Dataset,
                       val_batch_size: int,
                       val_num_workers: int):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.train_dataloader_kwargs = {
                'batch_size': train_batch_size,
                'num_workers': train_num_workers,
                'collate_fn': collate_fn,
                'sampler': InfiniteSampler(train_dataset),
            }
        self.val_dataloader_kwargs = {
                'batch_size': val_batch_size,
                'num_workers': val_num_workers,
                'collate_fn': collate_fn,
            }

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, **self.train_dataloader_kwargs)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, **self.val_dataloader_kwargs)


class Module(pl.LightningModule):

    def __init__(self, n_classes: int,
                       pretrained_vgg: bool,
                       freeze_vgg: bool):
        super().__init__()
        self.model = Model(n_classes=n_classes,
                           pretrained_vgg=pretrained_vgg,
                           freeze_vgg=freeze_vgg)
        self.save_hyperparameters()

    def configure_optimizers(self):
        optim = torch.optim.Adam(
                self.model.parameters(),
                lr=0.0001,
            )
        return optim

    def _normalize(self, input: torch.Tensor) -> torch.Tensor:
        input = input / 255.
        input[:, 0, :, :] = (input[:, 0, :, :] - 0.485) / 0.229
        input[:, 1, :, :] = (input[:, 1, :, :] - 0.456) / 0.224
        input[:, 2, :, :] = (input[:, 2, :, :] - 0.406) / 0.225
        return input

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input  = self._normalize(input)
        logits = self.model(input)
        return logits

    def training_step(self, batch: Batch, batch_idx: int) -> Any:
        images = self._normalize(batch.images)
        logits = self.model(images)
        loss = F.cross_entropy(logits, batch.masks)

        self.log('train_loss', loss.item())
        return loss

    def validation_step(self, batch: Batch, batch_idx: int) -> Dict[str, Any]:
        images = self._normalize(batch.images)
        logits = self.model(images)
        masks = torch.argmax(logits, dim=1)

        return {
                'images': batch.images.detach().cpu(),
                'targets': batch.masks.detach().cpu(),
                'masks': masks.detach().cpu(),
            }

    def validation_epoch_end(self, outputs: List[Dict[str, Any]]):
        result = []

        for output in outputs:
            images = output['images']
            targets = output['targets']
            masks = output['masks']

            for image, target, mask in zip(images, targets, masks):
                mask = mask2rgb(mask)
                target = mask2rgb(target)
                sample = make_grid([image, target, mask])
                result.append(wandb.Image(sample.float()))

        self.logger.experiment.log({
                'images': result,
            })
