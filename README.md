# 📝 PDF OCR Web Service

A lightweight and powerful web-based service for **OCR (Optical Character Recognition)** of PDF documents and images using **Qwen2.5-VL** and **Qwen2.5-7B** models. It allows users to:

- Upload and browse PDFs or images
- Select and render specific PDF pages
- Run OCR on any page or full document
- Clean and structure recognized text
- Generate summaries of the extracted content

---

## ⚙️ Requirements

### ✅ System Requirements
- Linux-based OS (Ubuntu, Debian, etc.)
- NVIDIA GPU with at least 24GB VRAM (e.g., RTX 3090, RTX 4090)
- CUDA Toolkit 12.2 or compatible installed

### ✅ Software Requirements
- Python 3.10+
- Conda (Miniconda or Anaconda recommended)
- `git`, `curl`, `wget`, `gcc`, `make`, and other development tools
- NVIDIA drivers and GPU-enabled PyTorch

---

## 🧪 Environment Setup

Follow the steps below to install and configure the service.

### 1. Install Miniconda

If not already installed:

```bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Restart your shell and initialize Conda:

```bash
source ~/miniconda3/bin/activate
conda init bash
```

### 2. Clone the Repository

```bash
git clone https://github.com/yourusername/PdfToOcr.git
cd PdfToOcr
```

### 3. Create the Conda Environment

```bash
conda create -n ocr python=3.10 -y
conda activate ocr
```

### 4. Install Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers accelerate bitsandbytes
pip install flask pillow pymupdf
```

Optionally install other utilities:

```bash
pip install gunicorn
```

---

## 🚀 Running the Web Service

### 1. Manual Startup (for testing)

```bash
conda activate ocr
python pdf_server.py
```

Service will be available at: [http://localhost:5000](http://localhost:5000)

### 2. Install as Systemd Service (for production)

Create the service file:

```bash
sudo nano /etc/systemd/system/ocr.service
```

Paste the following:

```ini
[Unit]
Description=OCR Web Service
After=network.target

[Service]
User=root
Group=root
WorkingDirectory=/root/PdfToOcr
ExecStart=/bin/bash -c "source /root/miniconda3/bin/activate ocr && python /root/PdfToOcr/pdf_server.py"
Restart=always
RestartSec=5
Environment="PATH=/root/miniconda3/envs/ocr/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

[Install]
WantedBy=multi-user.target
```

Enable and start the service:

```bash
sudo systemctl daemon-reexec
sudo systemctl daemon-reload
sudo systemctl enable ocr
sudo systemctl start ocr
```

Check logs:

```bash
journalctl -u ocr -f
```

---

## 📁 Upload Directory

All uploaded files and temporary images are stored in:

```
/mnt/Public/skany/uploads/
```

Ensure this directory exists and is writable by the service user.

---

## 📦 Models Used

- **OCR Model:** `Qwen/Qwen2.5-VL-7B-Instruct`  
- **Summary Model:** `Qwen/Qwen2.5-7B-Instruct`

Models are automatically downloaded via `transformers` on first use. Ensure you have a working internet connection and enough GPU memory.

---

## 🧹 Cleaning Up GPU Memory

The system is optimized to release GPU memory after each operation using `torch.cuda.empty_cache()` and `gc.collect()` to minimize resource usage between requests.

---

## 📸 UI Features

- Intuitive file upload for PDF & images
- Embedded PDF viewer
- Render PDF pages as high-resolution images
- Interactive OCR with instruction customization
- Summarization with LLM
- Full document processing with progress bar and cancellation support

---

## 🧑‍💻 Development Notes

Make sure `qwen_vl_utils.py` is present in the project root and contains the function `process_vision_info()` used by the VL model.
