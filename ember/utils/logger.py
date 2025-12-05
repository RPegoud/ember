import json
from datetime import datetime

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
        self.ts = datetime.now().strftime("%m-%d_%H-%M")
        init(autoreset=True)

    def log(
        self,
        message: str,
        color: str = "white",
        style: str = "bright",
    ):

        print(f"{COLORS_TO_FORE[color.upper()]}{STR_TO_STYLE[style.upper()]}{message}")

    def log_config(self, config: dict | DictConfig, **kwargs):
        if isinstance(config, DictConfig):
            config = OmegaConf.to_container(config, resolve=True)
        self.log("Config:", "green", **kwargs)
        self.log(
            f"{json.dumps(config, sort_keys=True, indent=4)}",
            "green",
            "normal",
            **kwargs,
        )
