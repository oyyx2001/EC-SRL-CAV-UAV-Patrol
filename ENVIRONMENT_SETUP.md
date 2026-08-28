# Project environment setup

The project uses a local Python virtual environment so its packages do not
modify the Anaconda base environment.  Python 3.13 is supported by the pinned
Windows wheels in the dependency files.

## Create and install

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --no-deps -r requirements-torch-cu128.txt
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`requirements-torch-cu128.txt` installs the official PyTorch CUDA 12.8 build
used by the NVIDIA RTX 5060.  On a machine without a compatible NVIDIA GPU,
replace that step with the CPU command from the official PyTorch installer.

## Activate in PowerShell

The helper also configures the project module paths and keeps the Matplotlib
cache inside the project directory.

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
. .\activate_project.ps1
```

The execution-policy change applies only to the current PowerShell process.

## Verify

```powershell
python -m pip check
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -m rl.train_masked_dqn --help
```

