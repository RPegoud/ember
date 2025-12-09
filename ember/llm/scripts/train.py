import os

import hydra
import lightning as L
import torch
from datasets import load_dataset
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from ..data import HFTokenizer
from ..layers import TopKSampler
from ..models import Transformer
from ..utils import GenerateCallback, Logger

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class Collator:
    def __init__(self, tokenizer: HFTokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, batch: list[str]) -> torch.Tensor:
        texts = [x["text"] for x in batch]
        return self.tokenizer(texts, mode="train")


class DataModule(L.LightningDataModule):
    """DataModule for the TinyStories dataset."""

    def __init__(self, cfg: DictConfig, collator: Collator) -> None:
        super().__init__()
        self.cfg = cfg
        self.train_ds = load_dataset(cfg.hparams.data.dataset, split="train")
        self.val_ds = load_dataset(cfg.hparams.data.dataset, split="validation")
        self.collator = collator

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.hparams.data.batch_size,
            num_workers=self.cfg.hparams.data.num_workers,
            persistent_workers=True,
            collate_fn=self.collator,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.hparams.data.batch_size,
            num_workers=self.cfg.hparams.data.num_workers,
            persistent_workers=True,
            collate_fn=self.collator,
        )


PROMPTS = ["Once upon a time", "Bob was a robot", "One day, a dragon"]


@hydra.main(version_base=None, config_path="../../config/llm", config_name="train")
def main(cfg: DictConfig):
    logger = Logger()
    logger.log_config(cfg)

    tokenizer = HFTokenizer(cfg.tokenizer.path)
    sampler = TopKSampler(top_k=50, temperature=1.0)

    collator = Collator(tokenizer)
    data_module = DataModule(cfg, collator)

    model = Transformer(
        vocab_size=tokenizer.vocab_size,
        model_dim=cfg.model.model_dim,
        hidden_dim=cfg.model.hidden_dim,
        attention=cfg.model.attention,
        n_attn_blocks=cfg.model.n_attn_blocks,
        learning_rate=cfg.model.learning_rate,
        pad_token_id=tokenizer.pad_token_id,
    )

    wandb_logger = WandbLogger(project="Ember")
    ckpt_callback = ModelCheckpoint(
        dirpath="ember/llm/checkpoints",
        filename="best-{train_loss:.4f}",
        save_top_k=1,
        monitor="train_loss",
        mode="min",
    )
    gen_callback = GenerateCallback(
        PROMPTS,
        max_new_tokens=cfg.hparams.callbacks.generate.max_new_tokens,
        tokenizer=tokenizer,
        sampler=sampler,
    )

    trainer = L.Trainer(
        max_epochs=cfg.hparams.trainer.max_epochs,
        precision=cfg.hparams.trainer.precision,
        gradient_clip_val=cfg.hparams.trainer.gradient_clip_val,
        accumulate_grad_batches=cfg.hparams.trainer.accumulate_grad_batches,
        log_every_n_steps=cfg.hparams.trainer.log_every_n_steps,
        limit_train_batches=cfg.hparams.trainer.limit_train_batches,
        limit_val_batches=cfg.hparams.trainer.limit_val_batches,
        val_check_interval=cfg.hparams.trainer.val_check_interval,
        callbacks=[ckpt_callback, gen_callback],
        logger=wandb_logger,
    )
    trainer.fit(
        model=model,
        train_dataloaders=data_module.train_dataloader(),
        val_dataloaders=data_module.val_dataloader(),
    )

    logger.log(f"Best model saved at: {ckpt_callback.best_model_path}", color="green")


#
if __name__ == "__main__":
    main()
