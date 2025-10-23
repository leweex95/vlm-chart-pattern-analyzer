"""Deploy VLM benchmark to Kaggle kernel."""
import os
import json
import subprocess
import shutil
import logging
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def run(model_id, config="baseline", notebook="vlm-inference-benchmark.ipynb", kernel_path=".", gpu=None, kernel_id=None):
    """
    Deploy Kaggle notebook kernel for VLM inference benchmarking.
    
    Args:
        model_id: Hugging Face model ID (e.g., "Qwen/Qwen2-VL-2B-Instruct")
        config: Benchmark configuration (baseline, fp16, quantized-int8, quantized-int4, etc.)
        notebook: Path to notebook template
        kernel_path: Path to kernel configuration directory
        gpu: Enable GPU (True/False/None)
        kernel_id: Kaggle kernel ID (e.g., "leventecsibi/vlm-chart-benchmark-baseline-qwen2-vl-2b")
    """
    logging.info(f"Deploying VLM benchmark: {model_id} (config={config})")
    
    # Create a temporary directory for this specific deployment
    # This ensures each kernel deployment is isolated and Kaggle treats them as separate kernels
    with tempfile.TemporaryDirectory(prefix=f"kaggle_{config}_") as temp_dir:
        temp_path = Path(temp_dir)
        
        # Copy kernel configuration to temp directory
        original_kernel_path = Path(kernel_path)
        for item in original_kernel_path.iterdir():
            if item.is_file():
                shutil.copy2(item, temp_path / item.name)
        
        # Load notebook
        nb_path = temp_path / Path(notebook).name
        if not nb_path.exists():
            # Copy from original location if not in kernel_path
            shutil.copy2(Path(notebook), nb_path)
        
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
        metadata_path = temp_path / "kernel-metadata.json"
        with open(metadata_path, "r", encoding="utf-8") as f:
            kernel_meta = json.load(f)
        
        # Update kernel ID if provided
        if kernel_id:
            kernel_meta["id"] = kernel_id
            # Update title to match
            model_name = model_id.split('/')[-1]
            kernel_meta["title"] = f"VLM Benchmark {config} {model_name}"
        
        if gpu is not None:
            kernel_meta["enable_gpu"] = str(gpu).lower()
        
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(kernel_meta, f, indent=2)
        
        # Push via Kaggle CLI from the temp directory
        kaggle_cmd = _get_kaggle_command()
        logging.info(f"Pushing kernel from temp directory: {temp_path}")
        logging.info(f"Kernel ID: {kernel_id}")
        
        result = subprocess.run(
            [*kaggle_cmd, "kernels", "push", "-p", str(temp_path)],
            capture_output=True, text=True, encoding='utf-8'
        )
        
        # Check for errors
        if result.returncode != 0:
            error_msg = f"Kaggle push failed (exit code {result.returncode})"
            if result.stderr:
                logging.error(f"stderr: {result.stderr}")
                error_msg += f"\nStderr: {result.stderr}"
            if result.stdout:
                logging.error(f"stdout: {result.stdout}")
                error_msg += f"\nStdout: {result.stdout}"
            raise RuntimeError(error_msg)
        
        # Parse the actual kernel ID from Kaggle's response
        actual_kernel_id = None
        if result.stdout:
            logging.debug(f"Kaggle push output: {result.stdout}")
            # Look for patterns like "Successfully pushed to username/kernel-slug"
            import re
            match = re.search(r'(?:pushed to|Kernel version \d+ for|at)\s+(\S+/[\w-]+)', result.stdout)
            if match:
                actual_kernel_id = match.group(1)
                logging.info(f"✓ Kernel deployed successfully: {actual_kernel_id}")
            else:
                logging.info("✓ Kernel deployed successfully")
        if result.stderr:
            logging.debug(f"Kaggle push stderr: {result.stderr}")
        
        return actual_kernel_id


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
