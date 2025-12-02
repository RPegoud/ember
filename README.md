# `Ember`: Portable Transformers and Diffusion 🦊

<img src="logo.png" width="170" align="right" />
`Ember` is a PyTorch-based library offering simple and hackable implementations of Transformer architectures and Diffusion models designed for quick iteration and prototyping.


## Features 💎

1. Llama implementation
    - [x] RMSNorm
    - [x] RoPE
    - [x] GQA
    - [x] MLA
    - [x] SwiGLU
    - [ ] Full LLM
        - [x] Tokenizer => encode, decode
        - [x] Weight tying
        - [x] Embeddings
        - [x] Init
        - [x] Samplers (top-K, top-p, min-p)
        - [x] Add RoPE to GQA
        - [ ] Generate loop
    - [x] Convert to PyTorch Lightning
    - [ ] KV cache
    - [ ] Test on TinyStories
    - [ ] (optional) Lightning indexer

2. Triton/Helion Kernels (write pytests at each step)
    - [ ] RMSNorm
    - [ ] SwiGLU
    - [ ] FlashAttention
    - [ ] CrossEntropyLoss

3. Utilities
    - [ ] Comprehensive testing
    - [ ] Custom typing?
    - [ ] Add FLOPs wrapper
    - [ ] Hydra config
    - [ ] Neptune / wandb logging
    - [ ] Push models to hf hub

4. Diffusion
    - [ ] DDPM
    - [ ] Flow-matching
    - [ ] Benchmark on CIFAR-10, CelebA-HQ
