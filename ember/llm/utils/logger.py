import json
from datetime import datetime

import torch.nn as nn
from colorama import Fore, Style, init
from omegaconf import DictConfig, OmegaConf

COLORS_TO_FORE = {
    "GREEN": Fore.GREEN,
    "BLUE": Fore.BLUE,
    "RED": Fore.RED,
    "CYAN": Fore.CYAN,
    "YELLOW": Fore.YELLOW,
    "WHITE": Fore.WHITE,
    "MAGENTA": Fore.MAGENTA,
}

STR_TO_STYLE = {
    "BRIGHT": Style.BRIGHT,
    "NORMAL": Style.NORMAL,
}


class Logger:

    def __init__(self):
        init(autoreset=True)

    def __call__(self, *args, **kwds):
        return self.log(*args, **kwds)

    def log(
        self,
        message: str,
        color: str = "white",
        style: str = "bright",
    ) -> None:

        print(f"{COLORS_TO_FORE[color.upper()]}{STR_TO_STYLE[style.upper()]}{message}")

    def log_config(self, config: dict | DictConfig, **kwargs):
        if isinstance(config, DictConfig):
            config = OmegaConf.to_container(config, resolve=True)
        self("Config:", "green", **kwargs)
        self(
            f"{json.dumps(config, sort_keys=True, indent=4)}",
            color="green",
            style="normal",
            **kwargs,
        )

    def log_model(self, model: nn.Module, **kwargs) -> None:
        self("Model:", color="blue", **kwargs)
        self(model, color="blue", style="normal", **kwargs)
