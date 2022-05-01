import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF

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
                       freeze_vgg: bool,
                       optimizer_lr: float):
        super().__init__()
        self._optim_lr = optimizer_lr
        self.model = Model(n_classes=n_classes,
                           pretrained_vgg=pretrained_vgg,
                           freeze_vgg=freeze_vgg)
        self.save_hyperparameters()

    def configure_optimizers(self):
        optim = torch.optim.Adam(
                self.model.parameters(),
                lr = self._optim_lr,
            )
        return optim

    def _normalize(self, input: torch.Tensor) -> torch.Tensor:
        input = input / 255
        input = TF.normalize(input, mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        return input

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input

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
                'masks': masks.detach().cpu(),
            }

    def validation_epoch_end(self, outputs: List[Dict[str, Any]]):
        images = []

        for output in outputs:
            for image, mask in zip(output['images'], output['masks']):
                mask = mask2rgb(mask)
                image_pair = torchvision.utils.make_grid([image, mask])
                image = wandb.Image(image_pair.float())
                images.append(image)

        self.logger.experiment.log({
                'images': images,
            })
