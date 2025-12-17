import torch


class KVCache:

    def __init__(
        self,
        n_layers: int,
        max_batch_size: int,
        max_seq_len: int,
        n_heads: int,
        head_dim: int,
        device: str,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Parent Key-Value cache, distributed to attention modules via `LayerKVCache`.

        Args:
            max_batch_size (int): the cache's maximum batch size
            max_seq_len (int): the cache's maximum sequence length
            n_heads (int): the number of attention heads used by the model
            head_dim (int): the dimension of a single head
            device (str): the device holding the kv cache
        """
        self.n_layers = n_layers
        self.batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.current_len = 0

        self.k_cache = [
            torch.zeros((max_batch_size, n_heads, max_seq_len, head_dim)).to(device)
            for _ in range(n_layers)
        ]
        self.v_cache = [
            torch.zeros((max_batch_size, n_heads, max_seq_len, head_dim)).to(device)
            for _ in range(n_layers)
        ]

    def store(
        self, k: torch.Tensor, v: torch.Tensor, layer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache for a specific layer and returns the _full_ view (past+current)
        for attention computation.

        Args:
            k (torch.Tensor): tensor of keys with shape (B, NH, S, HD)
            v (torch.Tensor): tensor of values with shape (B, NH, S, HD)
            layer_idx (int): the index of the current attention layer

        Returns:
            tuple[torch.Tensor]: the full keys and values views (past + current)
        """
        assert (
            k.ndim == 4 and v.ndim == 4
        ), f"Expected keys and values with shape [batch_size, n_heads, seq_len, head_dim], got {k.shape=}, {v.shape=}"

        batch_size, _, seq_len, _ = k.shape
        start_pos = self.current_len
        end_pos = start_pos + seq_len

        self.k_cache[layer_idx][:batch_size, :, start_pos:end_pos, :] = k
        self.v_cache[layer_idx][:batch_size, :, start_pos:end_pos, :] = v

        k_view = self.k_cache[layer_idx][:batch_size, :, :end_pos, :]
        v_view = self.v_cache[layer_idx][:batch_size, :, :end_pos, :]

        return k_view, v_view

    def step(self) -> None:
        """
        Increments the sequence pointer, called once per generation step
        after processing all layers.
        """
        self.current_len += 1

    def initialize_prefill(self, seq_len: int) -> None:
        """
        Sets the initial size of the cache after the prefill step.

        Args:
            seq_len (int): the length of the prefill sequence
        """
        self.current_len = seq_len


class LayerKVCache:
    def __init__(
        self,
        parent_cache: KVCache,
        layer_idx: int,
    ) -> None:
        """
        Layer-specific reference to the KV cache distributed to single attention modules.

        Args:
            parent_cache (KVCache): the global KV cache
            layer_idx (int): the attention module's index
        """
        self.parent_cache = parent_cache
        self.layer_idx = layer_idx

    def update(
        self, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the parent KV cache.

        Args:
            k (torch.Tensor): _description_
            v (torch.Tensor): _description_

        Returns:
            torch.Tensor: the full keys and values views (past + current)
        """
        return self.parent_cache.store(k, v, self.layer_idx)
