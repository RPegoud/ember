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
    def __init__(self, tokenizer):
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

    ds = load_dataset("roneneldan/TinyStories", split="train")
    loader = DataLoader(
        ds,
        batch_size=cfg.hyperparams.data.batch_size,
        persistent_workers=True,
        num_workers=cfg.hyperparams.data.num_workers,
        collate_fn=collator,
    )

    trainer = L.Trainer(
        max_epochs=cfg.hyperparams.trainer.max_epochs,
        precision=cfg.hyperparams.trainer.precision,
        gradient_clip_val=cfg.hyperparams.trainer.gradient_clip_val,
        accumulate_grad_batches=cfg.hyperparams.trainer.accumulate_grad_batches,
        log_every_n_steps=cfg.hyperparams.trainer.log_every_n_steps,
    )
    trainer.fit(model=model, train_dataloaders=loader)


if __name__ == "__main__":
    main()
