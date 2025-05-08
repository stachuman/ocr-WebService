# PDF OCR Web Service

A lightweight web application for OCR (Optical Character Recognition) of PDF documents and images powered by **Qwen2.5‑VL‑7B‑Instruct** (vision‑language) and **Qwen2.5‑7B‑Instruct** (text‑only) models.

---

## Key Features
- Upload PDFs or standalone images.
- Render any PDF page at 150 DPI and run OCR on the current page or the whole document.
- On‑the‑fly text clean‑up and Polish summaries generated with an LLM.
- Responsive UI with progress bar and cancellation.

---

## ⚙️ System Requirements
| Item | Minimum | Recommended |
|------|---------|-------------|
| **OS** | Linux (Ubuntu/Debian 12 tested) | Any modern Linux |
| **GPU** | NVIDIA 24 GB VRAM (RTX 3090/4090) | ≥ 48 GB VRAM for long docs |
| **CUDA** | 12.1 + |
| **Python** | 3.10 or 3.11 |

> **Why CUDA 12.1 +?** `flash_attention_2` (used by Qwen2.5‑VL for faster inference) requires a recent CUDA toolkit citeturn9search2.

---

## Python Dependencies

| Library | Notes |
|---------|-------|
| **PyTorch** ( GPU build) | `torch` ≥ 2.2 compiled for CU 12.1 |
| **flash‑attn** (optional → speed) | `pip install flash-attn --no-build-isolation` |
| **transformers** (dev) | Install **from source** to get `qwen2_5_vl` classes citeturn1view0 |
| **accelerate** | Device placement & 4‑bit loading |
| **qwen‑vl‑utils[decord]==0.0.8** | Helper for `process_vision_info` citeturn1view0 |
| **bitsandbytes** | 4‑bit quantisation support |
| **flask**, **pillow**, **pymupdf** | Web server, image IO, PDF rendering |

A ready‑to‑use **Conda environment** file is provided below (`environment.yml`).

---

## Quick Installation

```bash
# 1. Install Miniconda (skip if already installed)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"

# 2. Clone the repo and create the env
git clone https://github.com/yourusername/PdfToOcr.git
cd PdfToOcr
conda env create -f environment.yml
conda activate ocr
```

### Manual package install (alternative)

```bash
conda create -n ocr python=3.10 -y
conda activate ocr

# GPU PyTorch + FlashAttention build
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install flash-attn --no-build-isolation   # optional but recommended

# Latest Transformers w/ Qwen2.5‑VL
pip install git+https://github.com/huggingface/transformers accelerate

# Rest of the stack
pip install qwen-vl-utils[decord]==0.0.8 bitsandbytes flask pillow pymupdf
```

---

## Running

```bash
conda activate ocr
python pdf_server.py
```

Navigate to **http://localhost:5000**.

For production, enable the provided **systemd** unit (`ocr.service`).

---

## Data Paths
Uploads + page PNGs are stored in `./uploads/`.  
Make sure this directory exists and is writable.

---

## 🔍 environment.yml

```yaml
name: ocr
channels:
  - conda-forge
  - nvidia
  - pytorch
dependencies:
  - python=3.10
  - pip
  - pip:
      # GPU PyTorch CU12.1 wheels
      - torch==2.2.*+cu121 torchvision==0.17.* torchaudio==2.2.* --index-url https://download.pytorch.org/whl/cu121
      # Speed‑ups
      - flash-attn
      - bitsandbytes
      # HF ecosystem – dev transformers for Qwen2.5‑VL
      - git+https://github.com/huggingface/transformers
      - accelerate
      - qwen-vl-utils[decord]==0.0.8
      # App stack
      - flask
      - pillow
      - pymupdf
```

Create the env with:

```bash
conda env create -f environment.yml
```

---

## References
- Qwen2.5‑VL model card citeturn1view0
- Transformers docs for Qwen2.5‑VL citeturn5view0

---

Happy scanning! 📄✨
