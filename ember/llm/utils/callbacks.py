from datetime import datetime
from pathlib import Path

from huggingface_hub import upload_folder
from lightning.fabric import Fabric
from lightning.fabric.utilities import AttributeDict
from lightning.fabric.utilities.rank_zero import rank_zero_only
from omegaconf import DictConfig

from ..types import Sampler, Tokenizer
from .logger import Logger


class GenerateCallback:

    def __init__(
        self,
        prompts: list[str],
        tokenizer: Tokenizer,
        max_new_tokens: int,
        sampler: Sampler,
        logger: Logger,
    ):
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.prompts = prompts
        self.logger = logger
        self.sampler = sampler

    @rank_zero_only
    def on_validation_epoch_end(self, state: AttributeDict) -> None:
        state.model.eval()
        generated_texts = state.model.generate(
            self.prompts,
            max_new_tokens=self.max_new_tokens,
            sampler=self.sampler,
            tokenizer=self.tokenizer,
        )

        self.logger(f"Step {state.global_step} Generation", color="blue")
        for idx, text in enumerate(generated_texts):
            self.logger(f"{idx}: {text}", color="blue", style="normal")
            self.logger("-" * 30)
        state.model.train()


class CheckpointCallback:
    def __init__(
        self,
        save_dir: str,
        logger: Logger,
        cfg: DictConfig,
        filename: str = "best.ckpt",
        loss_format: str = "{:.3f}",
    ):
        self.logger = logger
        self.cfg = cfg

        self.timestamp = datetime.now().strftime("%Y_%m_%d__%H_%M")

        self.run_dir = Path(save_dir) / self.timestamp
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.base_name = Path(filename).stem
        self.suffix = Path(filename).suffix or ".ckpt"
        self.loss_format = loss_format

        self.best_val_loss = float("inf")
        self.current_ckpt_path: Path | None = None

        self.logger(f"Checkpoint directory: {self.run_dir}", color="blue")

    @rank_zero_only
    def on_validation_epoch_end(
        self,
        fabric: Fabric,
        state: AttributeDict,
        val_loss: float,
    ):
        if val_loss >= self.best_val_loss:
            return

        # remove previous checkpoint
        if self.current_ckpt_path is not None and self.current_ckpt_path.exists():
            self.current_ckpt_path.unlink()

        self.best_val_loss = val_loss
        loss_str = self.loss_format.format(val_loss)

        filename = f"{self.base_name}_loss={loss_str}_{self.timestamp}{self.suffix}"
        self.current_ckpt_path = self.run_dir / filename

        self.logger(
            f"New best validation loss: {val_loss:.4f}",
            color="cyan",
        )

        fabric.save(self.current_ckpt_path, state)

        self.logger(
            f"Saved new best model → {self.current_ckpt_path.name}",
            color="green",
        )

    @rank_zero_only
    def on_train_end(
        self,
    ):
        if self.cfg.push_to_hf_hub:
            self.logger("Pushing train state to HF Hub ...", color="magenta")
            upload_folder(
                folder_path=self.run_dir,
                repo_id="Ryan-Pegoud/tinyStories",
                repo_type="model",
            )
