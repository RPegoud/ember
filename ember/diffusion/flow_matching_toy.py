from copy import deepcopy

import pandas as pd
import plotly.express as px
import torch
import torch.distributions as distr
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm


def sample_target(batch_size: int) -> torch.Tensor:
    assert batch_size % 2 == 0, f"Batch size must be even, got {batch_size}"

    d2 = distr.Normal(2, 0.5)
    d1 = distr.Normal(-2, 0.5)

    samples = torch.cat(
        [
            d1.sample((batch_size // 2, 1)),
            d2.sample((batch_size // 2, 1)),
        ]
    )
    indices = torch.randperm(samples.nelement())
    return samples[indices]


def sample_noise(batch_size: int) -> torch.Tensor:
    return distr.Normal(0, 1).sample((batch_size, 1))


def get_noisy_state_and_velocity(
    batch_size: int, t: torch.Tensor | None = None
) -> torch.Tensor:
    assert batch_size % 2 == 0, f"Batch size must be even, got {batch_size}"
    if not t:
        t = distr.Uniform(0, 1).sample((batch_size, 1))
    x0 = sample_noise(batch_size)
    x1 = sample_target(batch_size)

    noisy_state = torch.cat((t, (1 - t) * x0 + t * x1), axis=1)
    velocity = x1 - x0
    return noisy_state, velocity


def plot_samples(x_0: torch.Tensor, x_t: torch.Tensor):
    x_0, x_t = tuple(map(tensor_to_numpy, (x_0, x_t)))
    noise = pd.DataFrame(x_0)
    targets = pd.DataFrame(x_t)

    targets["type"] = "target"
    noise["type"] = "noise"
    df = pd.concat((noise, targets), axis=0)
    df.rename(columns={0: "value"}, inplace=True)

    fig = px.histogram(
        df,
        x="value",
        color="type",
        facet_col="type",
        title="Flow-matching samples",
        template="plotly_white",
    )
    fig.show()


def tensor_to_numpy(x: torch.Tensor) -> torch.Tensor:
    return x.detach().flatten().numpy()


@torch.no_grad
def sample(batch_size: int, n_steps: int) -> torch.Tensor:
    noise_schedule = torch.linspace(0, 1, n_steps)
    delta_t = 1 / n_steps

    x_0 = sample_noise(batch_size)
    x_t = deepcopy(x_0)

    for idx, t in tqdm(enumerate(noise_schedule)):
        if idx == n_steps - 1:  # skip the last step
            continue
        t = t.broadcast_to(x_t.shape)
        predicted_velocity = model(torch.cat((t, x_t), dim=1))
        x_t += predicted_velocity * delta_t

    return x_0, x_t


if __name__ == "__main__":
    batch_size = 64
    hidden_size = 512
    train_samples = 32_768

    model = nn.Sequential(
        nn.Linear(2, hidden_size),
        nn.SiLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.SiLU(),
        nn.Linear(hidden_size, 1),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

    losses = []
    for i in tqdm(range(train_samples // batch_size), desc="Training ..."):
        optimizer.zero_grad()
        noisy_states, velocity = get_noisy_state_and_velocity(batch_size)
        preds = model(noisy_states)
        loss = F.mse_loss(preds, velocity)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            print(f"Loss at step {i}: {loss.item():.2e}")

    x_0, x_t = sample(batch_size=8192, n_steps=10)
    plot_samples(x_0, x_t)
