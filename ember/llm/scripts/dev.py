import hydra
import plotly.express as px
import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import DictConfig
from tqdm.auto import tqdm

from ..utils import Logger


@hydra.main(version_base=None, config_path="../../config/llm", config_name="train")
def main(cfg: DictConfig):
    logger = Logger()
    logger.log_config(cfg)
    s = instantiate(cfg.sampler)
    print(s)

    model = nn.Linear(10, 2)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.hparams.optimizer.lr)

    scheduler_factory = hydra.utils.instantiate(cfg.hparams.scheduler)
    scheduler = scheduler_factory(optimizer=optimizer)

    print(f"Initialized Scheduler: {type(scheduler).__name__}")
    print(f"Opt Steps: {optimizer.state_dict()}")

    lrs = []
    for i in tqdm(range(cfg.hparams.train.max_steps)):
        optimizer.step()
        scheduler.step()
        lrs.append(optimizer.state_dict()["param_groups"][0]["lr"])

    fig = px.line(lrs)
    fig.show()


if __name__ == "__main__":
    main()
