# Installation Guide

## Dependency policy

- `requirements.txt`: minimal ranges for core deps
- `requirements.lock`: pinned, OS/CUDA-independent deps only
- `torch` / `torch-geometric`: install separately per environment

Official references:
- PyTorch install selector: https://pytorch.org/get-started/locally/
- PyG install guide: https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

## CPU-only setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric
pip install -r requirements.lock
```

## CUDA setup (example: cu121)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric
pip install -r requirements.lock
```

## Colab setup

```bash
!pip install -U pip
!pip install torch torchvision torchaudio
!pip install torch-geometric
!pip install -r requirements.lock
```
