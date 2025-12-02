import torch
import torch.nn as nn


def assert_shape(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 3:
        logits = logits.squeeze(1)
    assert logits.ndim == 2, f"Sampler expects [Batch, Vocab], got {logits.shape}"
    return logits


class TopKSampler(nn.Module):
    def __init__(
        self,
        top_k: int,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.temperature = temperature

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        logits = assert_shape(logits)

        if self.temperature == 0:  # greedy decoding
            return torch.argmax(logits, dim=-1).unsqueeze(-1)

        logits /= self.temperature
        top_k_val, _ = torch.topk(logits, self.top_k, dim=-1)
        min_val = top_k_val[:, -1].unsqueeze(-1)
        logits = torch.where(logits < min_val, torch.tensor(-float("inf")), logits)
        probs = torch.softmax(logits, dim=-1)

        return torch.multinomial(probs, num_samples=1)


class NucleusSampler(nn.Module):
    def __init__(
        self,
        top_p: float,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.top_p = top_p
        self.temperature = temperature

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        logits = assert_shape(logits)

        if self.temperature == 0:  # greedy decoding
            return torch.argmax(logits, dim=-1).unsqueeze(-1)

        logits /= self.temperature
        sorted_logits, sorted_indices = logits.sort(dim=-1, descending=True)
        cum_prob = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cum_prob > self.top_p  # tokens to remove

        # shift the mask to include the first token with cum prod > top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = 0  # do not remove the first token

        filtered_logits = torch.where(
            sorted_indices_to_remove, -float("inf"), sorted_logits
        )
        probs = torch.softmax(filtered_logits, dim=-1)
        samples_idx = torch.multinomial(probs, num_samples=1)

        return torch.gather(sorted_indices, dim=1, index=samples_idx)


class MinPSampler(nn.Module):
    """
    Source: Turning up the heat: min-p sampling for creative and coherent LLM outputs
    Url: https://arxiv.org/pdf/2407.01082
    """

    def __init__(self, min_p: float = 0.05, temperature: float = 1.0):
        super().__init__()
        self.min_p = min_p
        self.temperature = temperature

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        if self.temperature == 0:  # greedy decoding
            return torch.argmax(logits, dim=-1).unsqueeze(-1)

        logits /= self.temperature
        probs = torch.softmax(logits, dim=-1)
        max_p = torch.max(probs, dim=-1, keepdim=True)[0]
        threshold = max_p * self.min_p
        probs = torch.where(probs < threshold, 0.0, probs)
        probs /= probs.sum(dim=-1, keepdim=True)

        return torch.multinomial(probs, num_samples=1)
