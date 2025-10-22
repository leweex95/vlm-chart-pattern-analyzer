# Installation Notes

## PyTorch Installation (Windows)

If you encounter DLL errors with PyTorch on Windows, use version 2.3.0:

```bash
python -m pip uninstall torch torchvision -y
python -m pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cpu
```

## Dependencies

Install all dependencies:
```bash
pip install -r requirements.txt
```

Or use Poetry:
```bash
poetry install
```

## Running the Project

1. **Generate Charts** (requires MT5):
```bash
python generate_charts.py
```

2. **Test VLM Inference**:
```bash
python test_vlm.py
```

Note: First run will download the model (~5GB), which takes time.
