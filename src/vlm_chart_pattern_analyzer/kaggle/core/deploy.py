"""Deploy VLM benchmark to Kaggle kernel."""
import os
import json
import subprocess
import shutil
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def run(model_id, config="baseline", notebook="vlm-inference-benchmark.ipynb", kernel_path=".", gpu=None):
    """
    Deploy Kaggle notebook kernel for VLM inference benchmarking.
    
    Args:
        model_id: Hugging Face model ID (e.g., "Qwen/Qwen2-VL-2B-Instruct")
        config: Benchmark configuration (baseline, fp16, quantized-int8, quantized-int4, etc.)
        notebook: Path to notebook template
        kernel_path: Path to kernel configuration directory
        gpu: Enable GPU (True/False/None)
    """
    logging.info(f"Deploying VLM benchmark: {model_id} (config={config})")
    
    # Load notebook
    nb_path = Path(notebook)
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Update MODEL_ID and CONFIG in notebook environment
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            # Update MODEL_ID
            for i, line in enumerate(cell["source"]):
                if "MODEL_ID = os.environ.get" in line:
                    cell["source"][i] = f"MODEL_ID = os.environ.get('MODEL_ID', '{model_id}')\n"
                    break
            # Update CONFIG
            for i, line in enumerate(cell["source"]):
                if "CONFIG = os.environ.get" in line:
                    cell["source"][i] = f"CONFIG = os.environ.get('BENCHMARK_CONFIG', '{config}')\n"
                    break
    
    # Save updated notebook
    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    
    # Update kernel metadata
    metadata_path = Path(kernel_path) / "kernel-metadata.json"
    with open(metadata_path, "r", encoding="utf-8") as f:
        kernel_meta = json.load(f)
    if gpu is not None:
        kernel_meta["enable_gpu"] = str(gpu).lower()
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(kernel_meta, f, indent=2)
    
    # Push via Kaggle CLI
    kaggle_cmd = _get_kaggle_command()
    logging.info(f"Pushing kernel with command: {' '.join(kaggle_cmd)} kernels push -p {kernel_path}")
    
    result = subprocess.run(
        [*kaggle_cmd, "kernels", "push", "-p", str(kernel_path)],
        check=True, capture_output=True, text=True, encoding='utf-8'
    )
    
    if result.stdout:
        logging.debug(f"Kaggle push output: {result.stdout}")
    if result.stderr:
        logging.debug(f"Kaggle push stderr: {result.stderr}")
    
    logging.info("✓ Kernel deployed successfully")


def _get_kaggle_command():
    """
    Get the appropriate command to run Kaggle CLI.
    
    Returns:
        list: Command parts to execute kaggle CLI
    """
    # Check if poetry is available and we're in a poetry project
    def find_pyproject_toml():
        current = Path.cwd()
        while current != current.parent:  # Stop at root
            if (current / "pyproject.toml").exists():
                return True
            current = current.parent
        return False
    
    if shutil.which("poetry") and find_pyproject_toml():
        try:
            # Test if poetry can run the kaggle command
            result = subprocess.run(
                ["poetry", "run", "python", "-c", "import kaggle"],
                capture_output=True,
                check=False
            )
            if result.returncode == 0:
                return ["poetry", "run", "python", "-m", "kaggle.cli"]
        except Exception:
            pass
    
    # Fallback to direct python call
    return ["python", "-m", "kaggle.cli"]
