import os

import hydra
import torch
from lightning.fabric import Fabric
from lightning.fabric.utilities import AttributeDict
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf, open_dict
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data import HFDataModule, HFTokenizer
from ..layers import TopKSampler
from ..models import Transformer
from ..utils import CheckpointCallback, GenerateCallback, Logger

os.environ["TOKENIZERS_PARALLELISM"] = "false"
PROMPTS = [
    "Bob was a tiny robot",
    "Once upon a time",
    "One day, a dragon",
]


class Collator:
    def __init__(self, tokenizer: HFTokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, batch: list[str]) -> torch.Tensor:
        texts = [x["text"] for x in batch]
        # train / val use `train` mode for loss computation, generate uses `inference`
        return self.tokenizer(texts, mode="train")


def train(
    state: AttributeDict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: DictConfig,
    fabric: Fabric,
) -> AttributeDict:
    state.model.train()

    max_steps = cfg.hparams.train.max_steps
    grad_accum = cfg.hparams.train.grad_accumulation_steps
    val_interval = cfg.hparams.train.val_interval

    step_bar = tqdm(
        total=max_steps,
        desc="Training (optimizer steps)",
        disable=not fabric.is_global_zero,
    )

    train_iter = iter(train_loader)

    while state.global_step < max_steps:
        for micro_step in range(grad_accum):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
                state.epoch += 1

            is_accumulating = micro_step < grad_accum - 1

            with fabric.no_backward_sync(state.model, enabled=is_accumulating):
                loss = state.model.get_causal_loss(batch)
                fabric.backward(loss)

        fabric.clip_gradients(
            state.model,
            state.optimizer,
            clip_val=cfg.hparams.train.gradient_clip_val,
        )
        state.optimizer.step()
        state.optimizer.zero_grad(set_to_none=True)

        state.global_step += 1
        step_bar.update(1)

        fabric.log_dict(
            {"train_loss": loss.item(), "epoch": state.epoch}, step=state.global_step
        )

        if state.global_step % val_interval == 0:
            validate(state=state, val_loader=val_loader, cfg=cfg, fabric=fabric)

    step_bar.close()
    fabric.call("on_train_end")

    return state


def validate(
    state: AttributeDict, val_loader: DataLoader, cfg: DictConfig, fabric: Fabric
) -> None:
    state.model.eval()
    val_losses = []
    n_val_steps = cfg.hparams.data.n_val_samples // cfg.hparams.data.batch_size
    with torch.no_grad():
        for val_batch in tqdm(val_loader, desc="Validating", total=n_val_steps):
            val_loss = state.model.get_causal_loss(val_batch)
            val_losses.append(val_loss.item())

    avg_val_loss = sum(val_losses) / len(val_losses)
    fabric.log_dict(
        {"val_loss": avg_val_loss, "step": state.global_step, "epoch": state.epoch}
    )
    fabric.call(  # text generation and checkpointing
        "on_validation_epoch_end",
        state=state,
        fabric=fabric,
        val_loss=avg_val_loss,
    )
    state.model.train()


@hydra.main(version_base=None, config_path="../../config/llm", config_name="train")
def main(cfg: DictConfig) -> None:
    # --- Loggers ---
    logger = Logger()
    logger.log_config(cfg)
    wandb_logger = WandbLogger(project="Ember")

    # --- Config ---
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    target_batch_size = cfg.hparams.data.target_batch_size
    batch_size = cfg.hparams.data.batch_size
    assert (
        target_batch_size % batch_size == 0
    ), f"Expected target batch size to be divisible by physical batch size, got f{target_batch_size=} and {batch_size=}"
    with open_dict(cfg.hparams.train):
        cfg.hparams.train["grad_accumulation_steps"] = target_batch_size // batch_size
    logger(
        f"Set gradient accumulation steps to {cfg.hparams.train.grad_accumulation_steps}",
        color="yellow",
    )

    # --- Tokenizer and Sampler ---
    tokenizer = HFTokenizer(cfg.tokenizer.path, max_seq_len=cfg.model.max_seq_len)
    sampler = TopKSampler(top_k=50, temperature=1.0)

    # --- Dataset and Collator ---
    collator = Collator(tokenizer)
    data_module = HFDataModule(cfg, collator)
    data_module.setup()

    # --- Callbacks ---
    ckpt_callback = CheckpointCallback(save_dir=cfg.ckpt_path, logger=logger, cfg=cfg)
    gen_callback = GenerateCallback(
        PROMPTS,
        max_new_tokens=cfg.hparams.callbacks.generate.max_new_tokens,
        tokenizer=tokenizer,
        sampler=sampler,
        logger=logger,
    )

    # --- Fabric setup ---
    fabric = Fabric(
        devices=cfg.hparams.env.n_devices,
        accelerator=cfg.hparams.env.accelerator,
        precision=cfg.hparams.env.precision,
        callbacks=[ckpt_callback, gen_callback],
        loggers=[wandb_logger],
    )

    fabric.logger.log_hyperparams(cfg_dict)
    fabric.seed_everything(cfg.seed)
    fabric.launch()
    logger(f"Using device: {fabric.device}")

    # --- Model and Optimizer ---
    with fabric.init_module():  # init model directly on device with selected precision
        model = Transformer(
            vocab_size=tokenizer.vocab_size,
            max_seq_len=cfg.model.max_seq_len,
            model_dim=cfg.model.model_dim,
            hidden_dim=cfg.model.hidden_dim,
            attention=cfg.model.attention,
            n_attn_blocks=cfg.model.n_attn_blocks,
            pad_token_id=tokenizer.pad_token_id,
            device=fabric.device,
        )
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg.hparams.optimizer.lr,
            weight_decay=cfg.hparams.optimizer.weight_decay,
        )

    logger.log_model(model)
    model, optimizer = fabric.setup(model, optimizer)
    train_loader, val_loader = fabric.setup_dataloaders(
        data_module.train_dataloader(), data_module.val_dataloader()
    )

    train_state = AttributeDict(
        model=model,
        optimizer=optimizer,
        global_step=0,
        epoch=0,
        cfg_dict=cfg_dict,
    )

    # --- Train and Val ---
    train_state = train(
        state=train_state,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        fabric=fabric,
    )


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    main()
