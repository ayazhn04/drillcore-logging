# Dependency files

| File | Purpose |
|---|---|
| `base.txt` | Exact pins for every direct dependency except PyTorch. |
| `torch-cpu.txt` | CPU-only PyTorch pins (laptop, CI, smoke tests). |
| `lock-laptop-macos-cpu.txt` | Full `pip freeze` of the development laptop (macOS arm64, Python 3.12.10). Kept as a record; it is not meant to be installed on Linux. |

On the lab machine, every run writes its own `environment.json` (package versions, CUDA, git commit) next to its outputs, so each result can be traced back to its exact environment.
