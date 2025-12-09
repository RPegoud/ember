import lightning as L

from ..types import Sampler, Tokenizer


class GenerateCallback(L.Callback):
    def __init__(
        self,
        prompts: list[str],
        tokenizer: Tokenizer,
        sampler: Sampler,
        max_new_tokens: int,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.prompts = prompts

    def on_validation_epoch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        pl_module.eval()

        input_ids = self.tokenizer(self.prompts)
        generated_texts = pl_module.generate(
            input_ids, max_new_tokens=self.max_new_tokens, sampler=Sampler
        )

        print(f"--- Step {trainer.global_step} Generation ---")
        for idx, text in enumerate(generated_texts):
            print(f"{idx}: {text}")
        print("-----------------------------\n")

        pl_module.train()
