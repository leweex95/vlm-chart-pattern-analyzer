"""Download benchmark results from Kaggle kernel."""
import subprocess
from pathlib import Path
import logging
import shutil
import os

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")

KERNEL_ID = "leventecsibi/vlm-chart-benchmark"  # Update with your kernel ID


def run(dest="benchmark_results", kernel_id=None):
    """Download benchmark results from Kaggle kernel"""
    kernel_id = kernel_id or KERNEL_ID
    dest_path = Path(dest)
    dest_path.mkdir(parents=True, exist_ok=True)

    kaggle_cmd = _get_kaggle_command()

    logging.info(f"Downloading results from {kernel_id} to {dest_path}")
    
    # Download to a temp location first to avoid overwriting
    import tempfile
    with tempfile.TemporaryDirectory() as temp_download_dir:
        temp_path = Path(temp_download_dir)
        
        # Run Kaggle CLI to download to temp directory
        result = subprocess.run(
            [*kaggle_cmd, "kernels", "output", kernel_id, "-p", str(temp_path).replace("\\", "/"), "--force"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False  # Get bytes instead of text to handle encoding issues
        )

        # Decode outputs with error handling
        try:
            stdout_text = result.stdout.decode('utf-8')
        except UnicodeDecodeError:
            stdout_text = result.stdout.decode('latin1', errors='replace')
        
        try:
            stderr_text = result.stderr.decode('utf-8')
        except UnicodeDecodeError:
            stderr_text = result.stderr.decode('latin1', errors='replace')

        if result.returncode != 0:
            # Check if download actually succeeded despite return code
            if "Output file downloaded to" not in stdout_text:
                logging.warning("Kaggle CLI failed (non-zero exit code).")
            else:
                logging.info("Download succeeded despite non-zero return code")
        else:
            logging.info("Results downloaded successfully")
        
        # Now move the results from the temp directory to the destination
        # Kaggle downloads to kernel_id/ subdirectory, find and copy the CSV
        logging.info(f"Listing all files in temp directory {temp_path}:")
        for item in temp_path.rglob('*'):
            logging.info(f"  {item.relative_to(temp_path)}")
        
        for item in temp_path.rglob('benchmark_results.csv'):
            logging.info(f"Found CSV at: {item}")
            import shutil
            shutil.copy2(item, dest_path / 'benchmark_results.csv')
            logging.info(f"Copied CSV to: {dest_path / 'benchmark_results.csv'}")
        
        # Also copy any other result files (boot markers, diagnostics, etc.)
        for item in temp_path.rglob('*.txt'):
            logging.info(f"Found result file at: {item}")
            import shutil
            shutil.copy2(item, dest_path / item.name)
            logging.info(f"Copied result file to: {dest_path / item.name}")
        
        # Explicitly copy diagnostics file if it exists
        for item in temp_path.rglob('*diagnostics*'):
            logging.info(f"Found diagnostics-related file at: {item}")
            import shutil
            shutil.copy2(item, dest_path / item.name)
            logging.info(f"Copied diagnostics file to: {dest_path / item.name}")


def _get_kaggle_command():
    """
    Get the appropriate command to run Kaggle CLI.
    
    Returns:
        list: Command parts to execute kaggle CLI
    """
    # Check if poetry is available and we're in a poetry project
    if shutil.which("poetry") and Path("pyproject.toml").exists():
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
