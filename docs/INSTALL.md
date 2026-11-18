# Installation

The **lab machine (Ubuntu + NVIDIA GPU)** is the target for real experiments. The development laptop only runs tests and the CPU smoke test. Python **3.12** is recommended; 3.10–3.13 are supported.

## 1. Lab machine: Ubuntu with an NVIDIA GPU

### 1.1 Check the GPU driver
```bash
nvidia-smi
```
It must list the GPU and show a "CUDA Version". If the command is missing, ask the machine's administrator to install the NVIDIA driver (`sudo ubuntu-drivers install`). Installing drivers needs admin rights and a reboot. You do **not** need to install the CUDA toolkit separately, because PyTorch wheels bundle the CUDA runtime.

### 1.2 Get the code and create the environment
```bash
sudo apt install -y git python3.12 python3.12-venv
```
```bash
git clone <your-github-url>/drillcore-logging.git && cd drillcore-logging
```
```bash
bash scripts/setup_ubuntu.sh cu128
```
`scripts/setup_ubuntu.sh <CUDA_TAG>` creates `.venv`, installs **PyTorch 2.15.0 / torchvision 0.29.2** from the PyTorch index for that CUDA tag, then installs `requirements/base.txt` and the package itself. Pick the tag that matches your driver from the selector on <https://pytorch.org/get-started/locally/> (the driver's CUDA version shown by `nvidia-smi` must be ≥ the tag's version). `cu128` is only an example and was **not** verified for torch 2.15; use whatever the selector shows. Use `cpu` for a machine without a GPU.

Manual equivalent:
```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/<CUDA_TAG>
pip install -r requirements/base.txt
pip install -e .
```
Optional, to read Canon raw files directly: `pip install -e '.[raw]'`.

### 1.3 Verify
```bash
source .venv/bin/activate
corelog env-report          # "cuda_available": true and the GPU name must appear
pytest                      # ~40 tests, about 1 minute
corelog smoke-test          # synthetic end-to-end run, prints "[smoke] PASS"
```

### 1.4 Disk layout (recommended)
Keep data and outputs **outside** the git checkout, on the large data disk:
```
/data/drillcore/2026_dataset/   <- from the RAs (paths.data_root)
/data/drillcore/outputs/        <- tiles, features, runs, logs (paths.output_root)
~/drillcore-logging/            <- this repository
```

### 1.5 Long runs
Run training inside `tmux` so it survives a closed SSH session:
```bash
tmux new -s train
```
Then, inside the tmux session:
```bash
source .venv/bin/activate && corelog train -c configs/lab.yaml
```
Detach with `Ctrl-b d`; reattach with `tmux attach -t train`.

### 1.6 Pretrained VGG16 weights
`model.name=vgg16` with `vgg_weights=imagenet` downloads ~530 MB once into `~/.cache/torch/hub/checkpoints/`. If the lab machine has no internet access, copy that file there from another machine.

## 2. Development laptop (macOS / Linux, CPU only)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements/torch-cpu.txt -r requirements/base.txt && pip install -e .
pytest && corelog smoke-test
```
Do not train real models on the laptop.

## 3. Appendix: Windows with WSL2

The pipeline is plain Python with no OS-specific paths. If a Windows machine is ever used:
- **Recommended: WSL2 Ubuntu.** Install the NVIDIA Windows driver (it exposes CUDA inside WSL2), then follow §1 inside the Ubuntu shell. Keep data under the Linux filesystem (`/home/...`), not `/mnt/c/...`, because reading many small tiles through `/mnt/c` is very slow.
- **Native Windows** also works with PyTorch CUDA wheels: `py -3.12 -m venv .venv`, `.venv\Scripts\activate`, then the same `pip` commands. Set `train.num_workers: 0` if data-loader workers cause problems.

## 4. Reproducibility
- Direct dependencies are pinned in `requirements/base.txt`, and PyTorch is pinned in the commands above.
- Every training run writes `config.yaml` (the fully resolved configuration) and `environment.json` (package versions, CUDA, GPU, git commit and whether the working tree had uncommitted changes) into its run folder.
- Seeds are fixed (`project.seed`) and cuDNN is set to deterministic mode. Bit-identical GPU results are still not guaranteed across driver or GPU models.
