# Environment setup

The fresh 20-seed reproduction ran on **Windows / Python 3.9.21 / torch 2.7.0+cpu / NumPy 1.26.4**. `requirements.txt` pins the tested direct dependencies. A clean installation, other operating systems/Python versions, and CUDA numerical reproduction have not been certified by this validation.

## CPU evaluation

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python -m pip check
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -m tools.check_data_assets
python -m rl.reproduce_paper --smoke --workers 2
```

**Data prerequisite:** geographic inputs/derivatives are withheld from a clean checkout. Obtain authorized matching files and restore the config paths before executing the smoke/test/training commands. The preflight's missing-file failure is intentional; it does not fetch data. Code/model loading can be checked without scene files using the README's model-only command.

The CPU and optional CUDA 12.8 wheels are listed in the [official PyTorch 2.7.0 instructions](https://pytorch.org/get-started/previous-versions/#v270). Neither torchvision nor torchaudio is required. For GPU training, replace the CPU installation with:

```powershell
python -m pip install -r requirements-torch-cu128.txt
```

Hardware/driver compatibility must be checked separately. The combined paper-reproduction command uses CPU. Single-case evaluation supports `--device cuda` with one worker, but GPU reproduction is unvalidated.

## Local OpenMP caveat

The existing author environment required a **process-local workaround** for duplicate OpenMP runtimes:

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
```

This is **not automatically enabled**. It suppresses a duplicate-runtime safeguard and is not a general installation recommendation; prefer an environment with a single OpenMP runtime. Its use is explicitly recorded in the validation manifest.

## Development checks

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q tests
python -m rl.evaluate_checkpoint --help
python -m rl.reproduce_paper --help
python -m rl.train_masked_dqn --help
```

`activate_project.ps1` remains an optional relative-path helper. No manuscript/PDF packages are needed by the release runtime.
