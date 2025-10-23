"""Main orchestrator for VLM benchmarking pipeline on Kaggle."""
import argparse
import logging
from pathlib import Path
from vlm_chart_pattern_analyzer.kaggle.core import deploy, download
from vlm_chart_pattern_analyzer.kaggle.utils import poll_status

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def run_pipeline(model_id, config="baseline", notebook=None, kernel_path=None, gpu=True, dest="benchmark_results", kernel_id=None):
    """Run VLM benchmarking pipeline: deploy -> poll -> download"""
    
    cwd = Path(__file__).parent
    
    # Resolve paths
    if notebook is None:
        notebook = cwd / "config" / "vlm-inference-benchmark.ipynb"
    else:
        notebook = Path(notebook)
        if not notebook.is_absolute():
            notebook = cwd / notebook
    
    if kernel_path is None:
        kernel_path = cwd / "config"
    else:
        kernel_path = Path(kernel_path)
        if not kernel_path.is_absolute():
            kernel_path = cwd / kernel_path
    
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    
    logging.info("=" * 60)
    logging.info("VLM Benchmarking Pipeline")
    logging.info("=" * 60)
    logging.info(f"Model: {model_id}")
    logging.info(f"Config: {config}")
    logging.info(f"GPU: {gpu}")
    logging.info(f"Notebook: {notebook}")
    logging.info(f"Kernel path: {kernel_path}")
    logging.info("=" * 60)
    
    # Step 1: Deploy
    logging.info("\nStep 1: Deploying kernel...")
    try:
        deploy.run(model_id, config, str(notebook), str(kernel_path), gpu, kernel_id)
        logging.info("✓ Deployment completed")
    except Exception as e:
        logging.error(f"✗ Deployment failed: {str(e)}")
        raise
    
    # Step 2: Poll status
    logging.info("\nStep 2: Polling kernel status...")
    try:
        kernel_id = kernel_id or "leventecsibi/vlm-chart-benchmark"
        status = poll_status.run(kernel_id)
        logging.info(f"Kernel status: {status}")
        
        if status == "kernelworkerstatus.error":
            logging.error("✗ Kernel failed during benchmark")
            raise RuntimeError("Kaggle kernel failed during VLM benchmarking. Aborting pipeline.")
        
        logging.info("✓ Kernel execution completed successfully")
    except Exception as e:
        logging.error(f"✗ Status polling failed: {str(e)}")
        raise
    
    # Step 3: Download results
    logging.info("\nStep 3: Downloading results...")
    try:
        kernel_id = kernel_id or "leventecsibi/vlm-chart-benchmark"
        download.run(str(dest), kernel_id)
        logging.info(f"✓ Results downloaded to {dest}")
    except Exception as e:
        logging.error(f"✗ Download failed: {str(e)}")
        raise
    
    logging.info("\n" + "=" * 60)
    logging.info("Pipeline completed successfully!")
    logging.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="VLM Benchmarking Pipeline on Kaggle")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2-VL-2B-Instruct", help="Hugging Face model ID")
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "fp16", "int8"], help="Model precision")
    parser.add_argument("--notebook", type=str, default=None, help="Benchmark notebook path")
    parser.add_argument("--kernel_path", type=str, default=None, help="Kaggle kernel configuration directory")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU acceleration")
    parser.add_argument("--dest", type=str, default="benchmark_results", help="Output directory for results")
    parser.add_argument("--kernel_id", type=str, default="leventecsibi/vlm-chart-benchmark", help="Kaggle kernel ID")
    
    args = parser.parse_args()
    
    run_pipeline(
        model_id=args.model_id,
        precision=args.precision,
        notebook=args.notebook,
        kernel_path=args.kernel_path,
        gpu=args.gpu,
        dest=args.dest,
        kernel_id=args.kernel_id
    )


if __name__ == "__main__":
    main()
