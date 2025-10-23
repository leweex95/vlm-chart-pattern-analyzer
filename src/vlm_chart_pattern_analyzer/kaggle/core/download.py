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

    # Log files
    stdout_log = dest_path / "kaggle_cli_stdout.log"
    stderr_log = dest_path / "kaggle_cli_stderr.log"

    logging.info(f"Downloading results from {kernel_id} to {dest_path}")
    
    # Run Kaggle CLI
    result = subprocess.run(
        [*kaggle_cmd, "kernels", "output", kernel_id, "-p", str(dest_path).replace("\\", "/")],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8"
    )

    # Write all outputs to files
    stdout_log.write_text(result.stdout, encoding="utf-8")
    stderr_log.write_text(result.stderr, encoding="utf-8")

    if result.returncode != 0:
        # Check if download actually succeeded despite return code
        if "Output file downloaded to" not in stdout_log.read_text():
            logging.warning("Kaggle CLI failed (non-zero exit code). Check stdout/stderr log files.")
        else:
            logging.info("Download succeeded despite non-zero return code")
    else:
        logging.info("✓ Results downloaded successfully")


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
