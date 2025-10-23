#!/usr/bin/env python3
"""
Strategic VLM Benchmarking on Kaggle GPU.

Strategy:
- 3 VLM Models: Qwen2-VL-2B, LLaVA-1.6-Mistral-7B, Phi-3-Vision
- 8 Optimization Configs: baseline, fp16, quantized-int8, quantized-int4, compiled, flash-attention, tensorrt, tensorrt-int8
- 20 Chart Images per model
- Each deployment runs ONE config on ONE model
- Total: 8 configs × 3 models × 20 images = 480 total inferences

Why this approach:
✓ Each Kaggle deployment tests ONE optimization config only
✓ Avoids model reloading and GPU memory issues
✓ Clear isolation between different optimization strategies
✓ Easy to compare performance across configs
✓ Can run configs in parallel or sequentially
✓ Results are unambiguous (no config mixing)
"""
import argparse
import csv
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vlm_chart_pattern_analyzer.kaggle.main import run_pipeline
from vlm_chart_pattern_analyzer.metrics_collector import MetricsCollector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Model configurations
MODELS = {
    "smolvlm2-2.2b": {
        "model_id": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        "display_name": "SmolVLM2-2.2B",
    },
    "llava-1.6-mistral-7b": {
        "model_id": "llava-hf/llava-v1.6-mistral-7b-hf",
        "display_name": "LLaVA-1.6-Mistral-7B",
    },
    "phi-3-vision": {
        "model_id": "microsoft/Phi-3-vision-128k-instruct",
        "display_name": "Phi-3-Vision",
    },
}

# Optimization configurations
BENCHMARK_CONFIGS = {
    "baseline": {
        "precision": "fp32",
        "optimizations": [],
        "description": "Unoptimized baseline FP32",
    },
    "fp16": {
        "precision": "fp16",
        "optimizations": [],
        "description": "FP16 mixed precision",
    },
    "quantized-int8": {
        "precision": "int8",
        "optimizations": ["quantization"],
        "description": "INT8 quantization",
    },
    "quantized-int4": {
        "precision": "int4",
        "optimizations": ["quantization"],
        "description": "INT4 quantization (NF4)",
    },
    "compiled": {
        "precision": "fp16",
        "optimizations": ["torch_compile"],
        "description": "Torch compiled model",
    },
    "flash-attention": {
        "precision": "fp16",
        "optimizations": ["flash_attention"],
        "description": "Flash Attention v2",
    },
    "tensorrt": {
        "precision": "fp16",
        "optimizations": ["tensorrt"],
        "description": "TensorRT optimization",
    },
    "tensorrt-int8": {
        "precision": "int8",
        "optimizations": ["tensorrt", "quantization"],
        "description": "TensorRT with INT8 quantization",
    },
}


class KaggleBenchmarkOrchestrator:
    """Orchestrates multi-model benchmarking on Kaggle."""
    
    def __init__(
        self,
        images_dir: str = "data/images",
        results_dir: str = "data/results/kaggle_benchmark",
        kernel_id_prefix: str = "leventecsibi/vlm-benchmark",
        config: str = "baseline",
        gpu: bool = True,
        dry_run: bool = False,
        models: Dict = None,
    ):
        self.images_dir = Path(images_dir)
        self.results_dir = Path(results_dir)
        self.kernel_id_prefix = kernel_id_prefix
        self.config = config
        self.gpu = gpu
        self.dry_run = dry_run
        self.models = models or MODELS
        
        # Validate config
        if config not in BENCHMARK_CONFIGS:
            raise ValueError(f"Unknown config: {config}. Choose from: {list(BENCHMARK_CONFIGS.keys())}")
        
        self.config_settings = BENCHMARK_CONFIGS[config]
        
        # Ensure results directory exists
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics collector
        self.collector = MetricsCollector()
        
        # Track all deployments
        self.deployments: List[Dict] = []
    
    def get_images(self) -> List[Path]:
        """Get all chart images to process."""
        image_files = sorted(self.images_dir.glob("*.png"))
        if not image_files:
            raise FileNotFoundError(f"No images found in {self.images_dir}")
        logger.info(f"Found {len(image_files)} chart images")
        return image_files
    
    def log_deployment_plan(self):
        """Log the complete benchmarking plan."""
        images = self.get_images()
        num_models = len(self.models)
        # Each deployment runs ONE config (one precision + optimizations)
        total_inferences = len(images) * num_models
        
        logger.info("\n" + "=" * 70)
        logger.info("KAGGLE VLM BENCHMARKING PLAN")
        logger.info("=" * 70)
        model_names = ", ".join([info["display_name"] for info in self.models.values()])
        logger.info(f"Models: {len(self.models)} ({model_names})")
        logger.info(f"Config: {self.config} ({self.config_settings['precision']} + {', '.join(self.config_settings['optimizations']) if self.config_settings['optimizations'] else 'no optimizations'})")
        logger.info(f"Chart Images: {len(images)}")
        logger.info(f"Kaggle Deployments: {num_models} (one per model)")
        logger.info(f"Total Inferences: {total_inferences}")
        logger.info(f"Estimated GPU Time: {(total_inferences * 0.5 / 60):.1f} hours at ~0.5s/inference")
        logger.info("=" * 70)
        logger.info("\nDeployment Schedule:")
        
        for idx, (model_key, model_info) in enumerate(self.models.items(), 1):
            logger.info(f"\n  [{idx}] {model_info['display_name']}")
            logger.info(f"      Model ID: {model_info['model_id']}")
            logger.info(f"      Precision: {self.config_settings['precision']}")
            logger.info(f"      Optimizations: {', '.join(self.config_settings['optimizations']) if self.config_settings['optimizations'] else 'None'}")
            logger.info(f"      Images: {len(images)}")
            logger.info(f"      Total inferences: {len(images)}")
            kernel_id = f"{self.kernel_id_prefix}-{self.config}-{model_key}"
            logger.info(f"      Kernel ID: {kernel_id}")
        
        logger.info("\n" + "=" * 70)
    
    def deploy_model_benchmark(
        self,
        model_key: str,
        model_info: Dict,
        deployment_index: int,
        total_deployments: int,
    ) -> bool:
        """
        Deploy and run benchmark for a single model.
        
        This deploys to Kaggle and processes all 20 images with both FP32 and FP16.
        """
        model_id = model_info["model_id"]
        display_name = model_info["display_name"]
        
        logger.info("\n" + "=" * 70)
        logger.info(f"DEPLOYMENT [{deployment_index}/{total_deployments}] {display_name}")
        logger.info("=" * 70)
        logger.info(f"Model ID: {model_id}")
        
        # Create deployment-specific output directory
        deployment_results_dir = self.results_dir / model_key
        deployment_results_dir.mkdir(parents=True, exist_ok=True)
        
        # Use simple kernel ID without timestamp (Kaggle will version automatically)
        kernel_id = f"{self.kernel_id_prefix}-{self.config}-{model_key}"
        
        try:
            if self.dry_run:
                logger.info("[DRY RUN] Would deploy to Kaggle with:")
                logger.info(f"  - Model: {model_id}")
                logger.info(f"  - Kernel: {kernel_id}")
                logger.info(f"  - GPU: {self.gpu}")
                logger.info(f"  - Results directory: {deployment_results_dir}")
                
                # Create mock results for dry run
                self._create_mock_results(deployment_results_dir, model_key, display_name)
                deployment_success = True
            else:
                # Run the pipeline: deploy -> poll -> download
                logger.info(f"Starting pipeline deployment...")
                logger.info(f"  Config: {self.config}")
                logger.info(f"  GPU: {self.gpu}")
                logger.info(f"  Results directory: {deployment_results_dir}")
                
                # Note: run_pipeline will handle deploy -> poll -> download internally
                run_pipeline(
                    model_id=model_id,
                    config=self.config,
                    notebook=None,  # Use default
                    kernel_path=None,  # Use default
                    gpu=self.gpu,
                    dest=str(deployment_results_dir),
                    kernel_id=kernel_id,
                )
                
                deployment_success = True
                logger.info(f"✓ Deployment completed successfully")
        
        except Exception as e:
            logger.error(f"✗ Deployment failed: {str(e)}")
            deployment_success = False
        
        # Record deployment
        self.deployments.append({
            "model_key": model_key,
            "model_id": model_id,
            "display_name": display_name,
            "kernel_id": kernel_id,
            "status": "success" if deployment_success else "failed",
            "timestamp": datetime.now().isoformat(),
            "results_dir": str(deployment_results_dir),
        })
        
        return deployment_success
    
    def load_and_aggregate_results(self) -> bool:
        """Load results from all deployments and aggregate metrics."""
        logger.info("\n" + "=" * 70)
        logger.info("AGGREGATING RESULTS")
        logger.info("=" * 70)
        
        all_loaded = True
        
        for deployment in self.deployments:
            if deployment["status"] != "success":
                logger.warning(f"Skipping {deployment['display_name']} (failed deployment)")
                continue
            
            results_dir = Path(deployment["results_dir"])
            csv_file = results_dir / "benchmark_results.csv"
            
            if not csv_file.exists():
                logger.warning(f"No CSV found in {results_dir}")
                all_loaded = False
                continue
            
            try:
                logger.info(f"Loading {deployment['display_name']} results from {csv_file}")
                self.collector.add_metrics_from_csv(csv_file)
                logger.info(f"  ✓ Metrics loaded")
            except Exception as e:
                logger.error(f"  ✗ Error loading: {str(e)}")
                all_loaded = False
        
        return all_loaded
    
    def generate_reports(self):
        """Generate comprehensive benchmark reports."""
        logger.info("\n" + "=" * 70)
        logger.info("GENERATING REPORTS")
        logger.info("=" * 70)
        
        # Get summary statistics
        summary = self.collector.get_summary_stats()
        
        if not summary.get('num_inferences'):
            logger.warning("No results available yet - kernels still running on Kaggle")
            return
        
        logger.info(f"\nSummary Statistics:")
        logger.info(f"  Total inferences: {summary['num_inferences']}")
        logger.info(f"  Models: {', '.join(summary['models'])}")
        logger.info(f"  Precisions: {', '.join(summary['precisions'])}")
        
        if summary['latency_ms']:
            logger.info(f"\n  Latency (ms):")
            logger.info(f"    Min: {summary['latency_ms']['min']:.2f}")
            logger.info(f"    Max: {summary['latency_ms']['max']:.2f}")
            logger.info(f"    Mean: {summary['latency_ms']['mean']:.2f}")
            logger.info(f"    Median: {summary['latency_ms'].get('median', 0):.2f}")
            logger.info(f"    Stdev: {summary['latency_ms'].get('stdev', 0):.2f}")
        
        if summary['memory_mb']:
            logger.info(f"\n  Memory (MB):")
            logger.info(f"    Min: {summary['memory_mb']['min']:.2f}")
            logger.info(f"    Max: {summary['memory_mb']['max']:.2f}")
            logger.info(f"    Mean: {summary['memory_mb']['mean']:.2f}")
        
        if summary['throughput_tokens_per_sec']:
            logger.info(f"\n  Throughput (tokens/sec):")
            logger.info(f"    Min: {summary['throughput_tokens_per_sec']['min']:.2f}")
            logger.info(f"    Max: {summary['throughput_tokens_per_sec']['max']:.2f}")
            logger.info(f"    Mean: {summary['throughput_tokens_per_sec']['mean']:.2f}")
        
        # Model comparison
        logger.info(f"\nModel Comparison:")
        model_comp = self.collector.get_model_comparison()
        for model, stats in sorted(model_comp.items()):
            logger.info(f"  {model}:")
            logger.info(f"    Latency: {stats.get('avg_latency_ms', 0):.2f} ms")
            logger.info(f"    Memory: {stats.get('avg_memory_mb', 0):.2f} MB")
            logger.info(f"    Throughput: {stats.get('avg_throughput_tokens_per_sec', 0):.2f} tokens/sec")
        
        # Precision comparison
        logger.info(f"\nPrecision Comparison:")
        prec_comp = self.collector.get_precision_comparison()
        for precision, stats in sorted(prec_comp.items()):
            logger.info(f"  {precision}:")
            logger.info(f"    Latency: {stats.get('avg_latency_ms', 0):.2f} ms")
            logger.info(f"    Memory: {stats.get('avg_memory_mb', 0):.2f} MB")
            logger.info(f"    Throughput: {stats.get('avg_throughput_tokens_per_sec', 0):.2f} tokens/sec")
        
        # Export reports
        logger.info(f"\nExporting reports...")
        try:
            self.collector.export_to_json(self.results_dir / "benchmark_report.json")
            logger.info(f"  ✓ JSON report: {self.results_dir / 'benchmark_report.json'}")
        except Exception as e:
            logger.error(f"  ✗ JSON export failed: {str(e)}")
        
        try:
            self.collector.export_to_csv(self.results_dir / "benchmark_metrics.csv")
            logger.info(f"  ✓ CSV report: {self.results_dir / 'benchmark_metrics.csv'}")
        except Exception as e:
            logger.error(f"  ✗ CSV export failed: {str(e)}")
        
        try:
            self.collector.export_to_html_report(self.results_dir / "benchmark_report.html")
            logger.info(f"  ✓ HTML report: {self.results_dir / 'benchmark_report.html'}")
        except Exception as e:
            logger.error(f"  ✗ HTML export failed: {str(e)}")
        
        # Save deployment metadata
        deployment_log = self.results_dir / "deployments.json"
        with open(deployment_log, 'w') as f:
            json.dump(self.deployments, f, indent=2)
        logger.info(f"  ✓ Deployment log: {deployment_log}")
    
    def _create_mock_results(self, results_dir: Path, model_key: str, display_name: str):
        """Create mock results for dry run testing."""
        import random
        
        images_dir = self.images_dir
        image_files = sorted(images_dir.glob("*.png"))
        
        csv_path = results_dir / "benchmark_results.csv"
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "image_filename", "model_id", "precision", "device",
                    "latency_ms", "memory_used_mb", "tokens_generated",
                    "throughput_tokens_per_sec", "timestamp"
                ]
            )
            writer.writeheader()
            
            precision = self.config_settings['precision']
            for image_file in image_files:
                # Generate realistic mock data
                if model_key == "qwen2-vl-2b" or model_key == "smolvlm2-2.2b":
                    latency_base = 150 if precision in ["fp16", "int8", "int4"] else 250
                    memory_base = 100 if precision in ["fp16", "int8", "int4"] else 150
                elif "llava" in model_key:
                    latency_base = 400 if precision in ["fp16", "int8", "int4"] else 700
                    memory_base = 300 if precision in ["fp16", "int8", "int4"] else 500
                else:  # phi-3-vision
                    latency_base = 300 if precision in ["fp16", "int8", "int4"] else 550
                    memory_base = 200 if precision in ["fp16", "int8", "int4"] else 350
                
                latency = latency_base + random.uniform(-20, 20)
                memory = memory_base + random.uniform(-10, 10)
                tokens = random.randint(100, 200)
                throughput = tokens / (latency / 1000)
                
                writer.writerow({
                    "image_filename": image_file.name,
                    "model_id": self.models[model_key]["model_id"],
                    "precision": precision,
                    "device": "cuda",
                    "latency_ms": f"{latency:.2f}",
                    "memory_used_mb": f"{memory:.2f}",
                    "tokens_generated": tokens,
                    "throughput_tokens_per_sec": f"{throughput:.2f}",
                    "timestamp": datetime.now().isoformat(),
                })
    
    def run_benchmark(self):
        """Run the complete benchmark orchestration."""
        try:
            # Log the plan
            self.log_deployment_plan()
            
            # Run deployments
            successful_deployments = 0
            for idx, (model_key, model_info) in enumerate(self.models.items(), 1):
                success = self.deploy_model_benchmark(
                    model_key,
                    model_info,
                    deployment_index=idx,
                    total_deployments=len(self.models),
                )
                if success:
                    successful_deployments += 1
                
                # Add delay between deployments to avoid rate limiting
                if idx < len(self.models):
                    logger.info("\nWaiting 30 seconds before next deployment...")
                    time.sleep(30)
            
            logger.info(f"\n✓ Completed {successful_deployments}/{len(self.models)} deployments")
            
            # Aggregate and report results
            if successful_deployments > 0:
                self.load_and_aggregate_results()
                self.generate_reports()
            else:
                logger.error("No successful deployments to aggregate")
            
            logger.info("\n" + "=" * 70)
            logger.info("BENCHMARK COMPLETE")
            logger.info("=" * 70)
        
        except Exception as e:
            logger.error(f"Benchmark failed: {str(e)}", exc_info=True)
            raise


def main():
    parser = argparse.ArgumentParser(
        description="Strategic VLM Benchmarking on Kaggle GPU",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full benchmark on Kaggle
  poetry run python scripts/benchmark.py --mode kaggle

  # Dry run (generates mock results for testing)
  poetry run python scripts/benchmark.py --mode kaggle --dry-run

  # Specify custom Kaggle kernel ID
  poetry run python scripts/benchmark.py --mode kaggle --kernel-id-prefix "myusername/my-benchmark"
        """
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["kaggle"],
        default="kaggle",
        help="Benchmark mode (currently only Kaggle is supported)"
    )
    parser.add_argument(
        "--config",
        type=str,
        choices=list(BENCHMARK_CONFIGS.keys()),
        default="baseline",
        help="Benchmark configuration (baseline, fp16, sglang, tensorrt, quantized-int8, compiled, flash-attention)"
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default="data/images",
        help="Directory containing chart images"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="data/results/kaggle_benchmark",
        help="Output directory for benchmark results"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="Specific models to benchmark (e.g., qwen2-vl-2b smolvlm2-2.2b). If not specified, all models will be benchmarked."
    )
    parser.add_argument(
        "--kernel-id-prefix",
        type=str,
        default="leventecsibi/vlm-chart-benchmark",
        help="Kaggle kernel ID prefix (will append model key)"
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="Disable GPU acceleration on Kaggle"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run with mock results (no actual Kaggle deployment)"
    )
    
    args = parser.parse_args()
    
    # Filter models if specified
    models_to_benchmark = MODELS
    if args.models:
        models_to_benchmark = {k: v for k, v in MODELS.items() if k in args.models}
        if not models_to_benchmark:
            logger.error(f"No valid models found. Available: {', '.join(MODELS.keys())}")
            sys.exit(1)
    
    # Create orchestrator
    orchestrator = KaggleBenchmarkOrchestrator(
        images_dir=args.images_dir,
        results_dir=args.results_dir,
        kernel_id_prefix=args.kernel_id_prefix,
        config=args.config,
        gpu=not args.no_gpu,
        dry_run=args.dry_run,
        models=models_to_benchmark,
    )
    
    # Run benchmark
    orchestrator.run_benchmark()


if __name__ == "__main__":
    main()
