#!/usr/bin/env bash
set -euo pipefail

# This setup script assumes the existence of a .env file in the root directory
# with keys WANDB_API_KEY and HF_TOKEN, as well as pyproject.toml and uv.lock files 

echo "▶ Setting up environment..."

# --- 1. Load .env safely ---
if [ ! -f ".env" ]; then
  echo "❌ .env file not found in project root"
  exit 1
fi

# export variables from .env automatically
set -a
source .env
set +a

# --- 2. Install uv (if missing) ---
if ! command -v uv &> /dev/null; then
  echo "▶ Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.cargo/bin:$PATH"
else
  echo "✔ uv already installed"
fi

# --- 3. Sync dependencies ---
echo "▶ Syncing Python environment with uv..."
uv sync

# --- 4. Hugging Face authentication ---
if [[ -n "${HF_TOKEN:-}" ]]; then
  echo "▶ Authenticating with Hugging Face..."
  echo "${HF_TOKEN}" | uv run huggingface-cli login --token --stdin
else
  echo "⚠ HF_TOKEN not set, skipping Hugging Face login"
fi

# --- 5. Weights & Biases authentication ---
if [[ -n "${WANDB_API_KEY:-}" ]]; then
  echo "▶ Authenticating with Weights & Biases..."
  uv run wandb login --relogin "${WANDB_API_KEY}"
else
  echo "⚠ WANDB_API_KEY not set, skipping wandb login"
fi

# --- 6. PyTorch + CUDA sanity check ---
echo "▶ Running CUDA sanity check..."
uv run python - << 'EOF'
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
EOF

# --- 7. Recommended env flags for training ---
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- 8. Setup tmux session ---
echo "Attaching tmux..."
if [ -z "${TMUX:-}" ]; then
  tmux new -A -s ember
fi

echo "✔ Setup complete"