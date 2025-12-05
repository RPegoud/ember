<p align="center">
    <a href="docs/images/stoix.png">
        <img src="logo.png" alt="Ember logo" width="40%"/>
    </a>
</p>

# `Ember`: Portable Transformers and Diffusion Models 🦊


`Ember` is a PyTorch-based library offering simple and hackable implementations of Transformer architectures and Diffusion models designed for quick iteration and prototyping.


## Features 💎

1. Language Models 
    - [x] RMSNorm
    - [x] RoPE
    - [x] GQA
    - [x] MLA
    - [x] SwiGLU
    - [x] Full LLM
        - [x] Tokenizer => encode, decode
        - [x] Weight tying
        - [x] Embeddings
        - [x] Init
        - [x] Samplers (top-K, top-p, min-p)
        - [x] Add RoPE to GQA
        - [x] KV cache
        - [x] Generate loop
    - [x] Convert to PyTorch Lightning
    - [x] Ensure right padding for training, left for generation, add pad token
    - [x] Add caching to MLA
    - [ ] Train on TinyStories
    - [ ] (optional) Lightning indexer

2. Triton/Helion Kernels (write pytests at each step)
    - [ ] RMSNorm
    - [ ] RoPE
    - [ ] SwiGLU
    - [ ] FlashAttention
    - [ ] CrossEntropyLoss

3. Utilities
    - [ ] Comprehensive testing
    - [x] Custom typing
    - [ ] FLOPs wrapper
    - [ ] Hydra config
    - [ ] Neptune / wandb logging
    - [ ] Push models to hf hub
    - [ ] Docstrings / Sphynx

4. Diffusion
    - [ ] DDPM
    - [ ] Flow-matching
    - [ ] Benchmark on CIFAR-10, CelebA-HQ
