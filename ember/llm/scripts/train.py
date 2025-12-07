import hydra
import lightning as L
import torch
from datasets import load_dataset
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from ...utils import Logger
from ..data import HFTokenizer
from ..models import Transformer


class Collator:

    def __init__(self, tokenizer: HFTokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, batch: list[str]) -> torch.Tensor:
        texts = [x["text"] for x in batch]
        return self.tokenizer(texts, mode="train")


@hydra.main(version_base=None, config_path="../../configs/llm", config_name="train")
def main(cfg: DictConfig):
    logger = Logger()
    logger.log_config(cfg)

    tokenizer = HFTokenizer(cfg.tokenizer.path)
    collator = Collator(tokenizer)
    model = Transformer(
        vocab_size=tokenizer.vocab_size,
        model_dim=cfg.model.model_dim,
        hidden_dim=cfg.model.hidden_dim,
        attention=cfg.model.attention,
        n_attn_blocks=cfg.model.n_attn_blocks,
        learning_rate=cfg.model.learning_rate,
        pad_token_id=tokenizer.pad_token_id,
    )

    ds = load_dataset(cfg.hparams.data.dataset, split=cfg.hparams.data.split)
    train_loader = DataLoader(
        ds,
        batch_size=cfg.hparams.data.batch_size,
        persistent_workers=True,
        num_workers=cfg.hparams.data.num_workers,
        collate_fn=collator,
    )

    trainer = L.Trainer(
        max_epochs=cfg.hparams.trainer.max_epochs,
        precision=cfg.hparams.trainer.precision,
        gradient_clip_val=cfg.hparams.trainer.gradient_clip_val,
        accumulate_grad_batches=cfg.hparams.trainer.accumulate_grad_batches,
        log_every_n_steps=cfg.hparams.trainer.log_every_n_steps,
    )
    trainer.fit(model=model, train_dataloaders=train_loader)


if __name__ == "__main__":
    main()
