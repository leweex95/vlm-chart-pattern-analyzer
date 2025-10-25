"""Deploy VLM benchmark to Kaggle kernel."""
import os
import json
import subprocess
import shutil
import logging
import tempfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Model ID aliases for convenience
MODEL_ALIASES = {
    "smolvlm2-2.2b": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    "phi3-vision": "microsoft/Phi-3-vision-128k-instruct",
    "qwen2-vl-2b": "Qwen/Qwen2-VL-2B-Instruct",
}


def run(model_id, precision="fp32", tensorrt=False, notebook="vlm-inference-benchmark.ipynb", kernel_path=".", gpu=None, kernel_id=None, limit=3, hf_token=None):
    """
    Deploy Kaggle notebook kernel for VLM inference benchmarking.
    
    Args:
        model_id: Hugging Face model ID (e.g., "Qwen/Qwen2-VL-2B-Instruct") or alias (e.g., "smolvlm2-2.2b")
        precision: Model precision (fp32, fp16, int8, int4)
        tensorrt: Enable TensorRT optimization (True/False)
        notebook: Path to notebook template
        kernel_path: Path to kernel configuration directory
        gpu: Enable GPU (True/False/None)
        kernel_id: Kaggle kernel ID (e.g., "leventecsibi/vlm-chart-benchmark-baseline-qwen2-vl-2b")
        limit: Limit number of images to process (0 = all, default = 3)
        hf_token: HuggingFace token for private/gated models (or read from env var HUGGINGFACE_HUB_TOKEN)
    """

    print(f"Deploying benchmark kernel for model: {model_id}, precision: {precision}, tensorrt: {tensorrt}, limit: {limit}")

    # Resolve HF token from parameter or environment
    if hf_token is None:
        hf_token = os.environ.get('HUGGINGFACE_HUB_TOKEN', '')
    
    if hf_token:
        logging.info("Using HuggingFace token for authentication")
    else:
        logging.warning("No HuggingFace token provided. If model is gated/private, authentication will fail.")
    
    # Resolve model alias to full repository ID
    resolved_model_id = MODEL_ALIASES.get(model_id, model_id)
    if resolved_model_id != model_id:
        logging.info(f"Resolved model alias '{model_id}' to '{resolved_model_id}'")
    
    logging.info(f"Deploying VLM benchmark: {resolved_model_id} (precision={precision}, tensorrt={tensorrt}, limit={limit})")

    # If the user has downloaded the model locally into a `models/` folder, prefer that
    # to avoid remote HF access (useful when Kaggle can't access gated repos).
    try:
        local_models_dir = Path.cwd() / "models"
        local_candidate = None
        # candidate names to try (in order): raw model_id alias, repo name, repo with slash replaced
        candidates = [
            local_models_dir / model_id,
            local_models_dir / resolved_model_id.split('/')[-1],
            local_models_dir / resolved_model_id.replace('/', '_')
        ]
        for c in candidates:
            if c.exists():
                local_candidate = c
                break

        if local_candidate:
            logging.info(f"Found local model at {local_candidate}; will use local copy instead of HF Hub")
            # Use absolute path - notebook detects it and loads from disk
            resolved_model_id = str(local_candidate.resolve())
        else:
            logging.info("No local model found under ./models; will attempt to load from Hugging Face Hub")
    except Exception:
        # Non-fatal - continue using resolved_model_id from aliases
        logging.debug("Local model detection failed; continuing with resolved_model_id")

    # Create a temporary directory for this specific deployment
    # This ensures each kernel deployment is isolated and Kaggle treats them as separate kernels
    with tempfile.TemporaryDirectory(prefix=f"kaggle_{precision}_") as temp_dir:
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
        
        # Inject configuration as the first cell
        # This ensures MODEL_ID and other variables are available
        config_cell = {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# AUTO-INJECTED CONFIGURATION\n",
                f"CONFIG_MODEL_ID = r'{resolved_model_id}'\n",
                f"CONFIG_PRECISION = '{precision}'\n",
                f"CONFIG_TENSORRT = {str(tensorrt)}\n",
                # Notebook will write under a single `results` folder. The orchestrator
                # (`main.py`) will place that folder under data/results/kaggle_benchmark/{model}/{precision}.
                "CONFIG_OUTPUT_SUBDIR = 'results'\n",
                f"CONFIG_IMAGE_LIMIT = {limit}\n"
            ]
        }
        
        # Insert configuration cell at the beginning
        nb['cells'].insert(0, config_cell)
        
        # Inject HF token setup AFTER configuration, so config variables are available
        if hf_token:
            token_cell = {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# AUTO-INJECTED HUGGINGFACE TOKEN\n",
                    "import os\n",
                    f"os.environ['HUGGINGFACE_HUB_TOKEN'] = '{hf_token}'\n",
                    "# Also set for huggingface_hub library\n",
                    "from huggingface_hub import login\n",
                    f"login(token='{hf_token}', add_to_git_credential=False)\n"
                ]
            }
            nb['cells'].insert(1, token_cell)  # Insert after config cell
            logging.info("Injected HuggingFace token cell")
        
        logging.info(f"Notebook loaded: {len(nb['cells'])} cells (injected config + token cells)")
        
        # Save updated notebook
        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=2)
        
        # Update kernel metadata
        metadata_path = temp_path / "kernel-metadata.json"
        with open(metadata_path, "r", encoding="utf-8") as f:
            kernel_meta = json.load(f)
        
        # Keep the existing kernel ID from metadata - don't override it
        # This ensures Kaggle updates the existing kernel instead of creating new ones
        actual_kernel_id = kernel_meta.get("id", kernel_id or "leventecsibi/vlm-chart-benchmark")
        
        if gpu is not None:
            kernel_meta["enable_gpu"] = str(gpu).lower()
        
        # Set environment variables for the benchmark config
        if "environment_variables" not in kernel_meta:
            kernel_meta["environment_variables"] = {}
        
        # Always preserve CHART_IMAGES_DIR if it exists
        kernel_meta["environment_variables"]["MODEL_ID"] = model_id
        kernel_meta["environment_variables"]["BENCHMARK_PRECISION"] = precision
        kernel_meta["environment_variables"]["TENSORRT"] = str(tensorrt).lower()
        kernel_meta["environment_variables"]["IMAGE_LIMIT"] = str(limit)
        # Keep kernel metadata OUTPUT_SUBDIR aligned with the notebook's results folder
        kernel_meta["environment_variables"]["OUTPUT_SUBDIR"] = "results"
        logging.info(f"Set environment variables: MODEL_ID={model_id}, BENCHMARK_PRECISION={precision}, TENSORRT={str(tensorrt).lower()}, IMAGE_LIMIT={str(limit)}")
        
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(kernel_meta, f, indent=2)
        
        # Push via Kaggle CLI from the temp directory
        kaggle_cmd = _get_kaggle_command()
        logging.info(f"Pushing kernel from temp directory: {temp_path}")
        logging.info(f"Kernel ID: {kernel_id}")
        
        result = subprocess.run(
            [*kaggle_cmd, "kernels", "push", "-p", str(temp_path)],
            capture_output=True, text=True, encoding='utf-8', stdin=subprocess.DEVNULL, timeout=300
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
        
        logging.info(f"Kernel deployed successfully: {actual_kernel_id}")
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
