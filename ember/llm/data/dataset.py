from itertools import islice
from typing import Iterator

import lightning as L
from datasets import load_dataset
from lightning.fabric.utilities.data import suggested_max_num_workers
from omegaconf import DictConfig
from torch.utils.data import DataLoader, IterableDataset


def streaming_safe_num_workers(dataset, suggested: int) -> int:
    num_shards = getattr(dataset, "num_shards", None)
    if num_shards is not None:
        return min(num_shards, suggested)
    return suggested


class LimitedIterableDataset(IterableDataset):
    """
    Iterable wrapper that optionally limits the number of yielded samples.
    Intended for validation only.
    """

    def __init__(self, dataset, limit: int | None = None) -> None:
        self.dataset = dataset
        self.limit = limit

    def __iter__(self) -> Iterator:
        it = iter(self.dataset)
        if self.limit is None:
            yield from it
        else:
            yield from islice(it, self.limit)

    @property
    def num_shards(self) -> int | None:
        return getattr(self.dataset, "num_shards", None)


class HFDataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig, collator) -> None:
        super().__init__()
        self.cfg = cfg
        self.collator = collator
        self.suggested_workers = suggested_max_num_workers(cfg.hparams.env.n_devices)

        self.train_ds = None
        self.val_ds = None

    def setup(self, stage: str | None = None) -> None:
        streaming = self.cfg.hparams.data.streaming

        train = load_dataset(
            self.cfg.hparams.data.dataset,
            split="train",
            streaming=streaming,
        )
        val = load_dataset(
            self.cfg.hparams.data.dataset,
            split="validation",
            streaming=streaming,
        )

        # Training: unbounded stream
        self.train_ds = train

        # Validation: optionally bounded (recommended)
        self.val_ds = LimitedIterableDataset(
            val,
            limit=self.cfg.hparams.data.n_val_samples,
        )

    def train_dataloader(self) -> DataLoader:
        num_workers = streaming_safe_num_workers(self.train_ds, self.suggested_workers)

        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.hparams.data.batch_size,
            num_workers=num_workers,
            collate_fn=self.collator,
            persistent_workers=num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        num_workers = streaming_safe_num_workers(self.val_ds, self.suggested_workers)

        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.hparams.data.batch_size,
            num_workers=num_workers,
            collate_fn=self.collator,
            persistent_workers=num_workers > 0,
        )
