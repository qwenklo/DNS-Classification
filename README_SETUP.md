# Setup Instructions

## Virtual Environment

A virtual environment has been created at `venv/` with all required packages installed.

## Running Your Scripts

To run your Python scripts with the virtual environment, use:

```bash
./venv/bin/python DNSAssignment.py
```

Or activate the virtual environment first:

```bash
source venv/bin/activate
python DNSAssignment.py
```

## Installed Packages

- PyTorch 2.9.1 (CPU version)
- NumPy
- scikit-learn
- tqdm
- torchsummary
- wandb
- pyyaml
- pandas

## Note

PyTorch is installed as CPU-only version to save disk space. If you need CUDA support, you can install it separately, but it requires significantly more disk space (~2GB+).

