"""Metrics collection and analysis for VLM benchmarking."""
import json
import csv
import logging
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
import statistics


@dataclass
class BenchmarkMetrics:
    """Container for VLM benchmark metrics."""
    image_filename: str
    model_id: str
    precision: str
    device: str
    latency_ms: float  # Total inference time in milliseconds
    memory_used_mb: float  # Memory consumed during inference
    tokens_generated: int  # Total tokens generated in output
    throughput_tokens_per_sec: float  # Tokens generated per second
    timestamp: str
    optimizations: str = ""
    response: str = ""
    error: str = ""
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkMetrics":
        """Create metrics from dictionary."""
        return cls(**data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return asdict(self)


class MetricsCollector:
    """Collects and aggregates VLM benchmark metrics."""
    
    def __init__(self):
        """Initialize metrics collector."""
        self.metrics: List[BenchmarkMetrics] = []
    
    def add_metric(self, metric: BenchmarkMetrics) -> None:
        """Add a benchmark metric."""
        self.metrics.append(metric)
    
    def add_metrics_from_csv(self, csv_path: Path) -> None:
        """Load metrics from CSV file."""
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert numeric fields, handling empty strings
                try:
                    row["latency_ms"] = float(row["latency_ms"]) if row["latency_ms"] else 0.0
                    row["memory_used_mb"] = float(row["memory_used_mb"]) if row["memory_used_mb"] else 0.0
                    row["tokens_generated"] = int(row["tokens_generated"]) if row["tokens_generated"] else 0
                    row["throughput_tokens_per_sec"] = float(row["throughput_tokens_per_sec"]) if row["throughput_tokens_per_sec"] else 0.0
                except ValueError as e:
                    logging.warning(f"Skipping row with invalid numeric data: {e}, row: {row}")
                    continue
                row["response"] = row.get("response", "")
                row["optimizations"] = row.get("optimizations", "")
                row["error"] = row.get("error", "")
                self.metrics.append(BenchmarkMetrics.from_dict(row))
    
    def add_metrics_from_json(self, json_path: Path) -> None:
        """Load metrics from JSON file."""
        with open(json_path, "r") as f:
            data = json.load(f)
            for result in data.get("results", []):
                self.metrics.append(BenchmarkMetrics.from_dict(result))
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics across all metrics."""
        if not self.metrics:
            return {}
        
        latencies = [m.latency_ms for m in self.metrics]
        memory_usage = [m.memory_used_mb for m in self.metrics]
        throughputs = [m.throughput_tokens_per_sec for m in self.metrics]
        token_counts = [m.tokens_generated for m in self.metrics]
        
        return {
            "num_inferences": len(self.metrics),
            "models": list(set(m.model_id for m in self.metrics)),
            "precisions": list(set(m.precision for m in self.metrics)),
            "devices": list(set(m.device for m in self.metrics)),
            "latency_ms": {
                "min": min(latencies),
                "max": max(latencies),
                "mean": statistics.mean(latencies),
                "median": statistics.median(latencies),
                "stdev": statistics.stdev(latencies) if len(latencies) > 1 else 0,
            },
            "memory_mb": {
                "min": min(memory_usage),
                "max": max(memory_usage),
                "mean": statistics.mean(memory_usage),
                "median": statistics.median(memory_usage),
                "stdev": statistics.stdev(memory_usage) if len(memory_usage) > 1 else 0,
            },
            "throughput_tokens_per_sec": {
                "min": min(throughputs),
                "max": max(throughputs),
                "mean": statistics.mean(throughputs),
                "median": statistics.median(throughputs),
                "stdev": statistics.stdev(throughputs) if len(throughputs) > 1 else 0,
            },
            "tokens_generated": {
                "total": sum(token_counts),
                "mean": statistics.mean(token_counts),
                "median": statistics.median(token_counts),
            },
        }
    
    def get_model_comparison(self) -> Dict[str, Dict[str, Any]]:
        """Get comparison statistics by model."""
        model_metrics: Dict[str, List[BenchmarkMetrics]] = {}
        
        for metric in self.metrics:
            if metric.model_id not in model_metrics:
                model_metrics[metric.model_id] = []
            model_metrics[metric.model_id].append(metric)
        
        comparison = {}
        for model_id, metrics_list in model_metrics.items():
            latencies = [m.latency_ms for m in metrics_list]
            memory_usage = [m.memory_used_mb for m in metrics_list]
            throughputs = [m.throughput_tokens_per_sec for m in metrics_list]
            
            comparison[model_id] = {
                "num_inferences": len(metrics_list),
                "avg_latency_ms": round(statistics.mean(latencies), 2),
                "avg_memory_mb": round(statistics.mean(memory_usage), 2),
                "avg_throughput": round(statistics.mean(throughputs), 2),
                "min_latency_ms": round(min(latencies), 2),
                "max_latency_ms": round(max(latencies), 2),
            }
        
        return comparison
    
    def get_precision_comparison(self) -> Dict[str, Dict[str, Any]]:
        """Get comparison statistics by precision."""
        precision_metrics: Dict[str, List[BenchmarkMetrics]] = {}
        
        for metric in self.metrics:
            if metric.precision not in precision_metrics:
                precision_metrics[metric.precision] = []
            precision_metrics[metric.precision].append(metric)
        
        comparison = {}
        for precision, metrics_list in precision_metrics.items():
            latencies = [m.latency_ms for m in metrics_list]
            memory_usage = [m.memory_used_mb for m in metrics_list]
            throughputs = [m.throughput_tokens_per_sec for m in metrics_list]
            
            comparison[precision] = {
                "num_inferences": len(metrics_list),
                "avg_latency_ms": round(statistics.mean(latencies), 2),
                "avg_memory_mb": round(statistics.mean(memory_usage), 2),
                "avg_throughput": round(statistics.mean(throughputs), 2),
            }
        
        return comparison
    
    def export_to_csv(self, output_path: Path, append: bool = False) -> None:
        """Export metrics to CSV. If append=True, append to file instead of overwriting."""
        if not self.metrics:
            return
        mode = "a" if append and output_path.exists() else "w"
        # Always include all possible fields
        fieldnames = [
            "image_filename", "model_id", "precision", "optimizations", "device", "latency_ms", "memory_used_mb",
            "tokens_generated", "throughput_tokens_per_sec", "timestamp", "response", "error"
        ]
        with open(output_path, mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if mode == "w":
                writer.writeheader()
            for metric in self.metrics:
                writer.writerow(metric.to_dict())
    
    def export_to_json(self, output_path: Path) -> None:
        """Export metrics and analysis to JSON."""
        report = {
            "metrics": [m.to_dict() for m in self.metrics],
            "summary": self.get_summary_stats(),
            "model_comparison": self.get_model_comparison(),
            "precision_comparison": self.get_precision_comparison(),
        }
        
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
    
    def export_to_html_report(self, output_path: Path) -> None:
        """Export metrics as HTML report."""
        summary = self.get_summary_stats()
        model_comparison = self.get_model_comparison()
        precision_comparison = self.get_precision_comparison()
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>VLM Benchmark Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .section {{ margin: 30px 0; }}
    </style>
</head>
<body>
    <h1>VLM Benchmark Report</h1>
    
    <div class="section">
        <h2>Summary Statistics</h2>
        <p><strong>Total Inferences:</strong> {summary.get("num_inferences", 0)}</p>
        <p><strong>Models:</strong> {", ".join(summary.get("models", []))}</p>
        <p><strong>Precisions:</strong> {", ".join(summary.get("precisions", []))}</p>
        <p><strong>Devices:</strong> {", ".join(summary.get("devices", []))}</p>
    </div>
    
    <div class="section">
        <h2>Latency Statistics (ms)</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Min</td>
                <td>{summary.get("latency_ms", {}).get("min", 0):.2f}</td>
            </tr>
            <tr>
                <td>Max</td>
                <td>{summary.get("latency_ms", {}).get("max", 0):.2f}</td>
            </tr>
            <tr>
                <td>Mean</td>
                <td>{summary.get("latency_ms", {}).get("mean", 0):.2f}</td>
            </tr>
            <tr>
                <td>Median</td>
                <td>{summary.get("latency_ms", {}).get("median", 0):.2f}</td>
            </tr>
            <tr>
                <td>Std Dev</td>
                <td>{summary.get("latency_ms", {}).get("stdev", 0):.2f}</td>
            </tr>
        </table>
    </div>
    
    <div class="section">
        <h2>Memory Statistics (MB)</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Min</td>
                <td>{summary.get("memory_mb", {}).get("min", 0):.2f}</td>
            </tr>
            <tr>
                <td>Max</td>
                <td>{summary.get("memory_mb", {}).get("max", 0):.2f}</td>
            </tr>
            <tr>
                <td>Mean</td>
                <td>{summary.get("memory_mb", {}).get("mean", 0):.2f}</td>
            </tr>
        </table>
    </div>
    
    <div class="section">
        <h2>Throughput Statistics (tokens/sec)</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Min</td>
                <td>{summary.get("throughput_tokens_per_sec", {}).get("min", 0):.2f}</td>
            </tr>
            <tr>
                <td>Max</td>
                <td>{summary.get("throughput_tokens_per_sec", {}).get("max", 0):.2f}</td>
            </tr>
            <tr>
                <td>Mean</td>
                <td>{summary.get("throughput_tokens_per_sec", {}).get("mean", 0):.2f}</td>
            </tr>
        </table>
    </div>
    
    <div class="section">
        <h2>Model Comparison</h2>
        <table>
            <tr>
                <th>Model</th>
                <th>Avg Latency (ms)</th>
                <th>Avg Memory (MB)</th>
                <th>Avg Throughput (tokens/sec)</th>
            </tr>
"""
        
        for model, stats in model_comparison.items():
            html_content += f"""
            <tr>
                <td>{model}</td>
                <td>{stats.get("avg_latency_ms", 0):.2f}</td>
                <td>{stats.get("avg_memory_mb", 0):.2f}</td>
                <td>{stats.get("avg_throughput", 0):.2f}</td>
            </tr>
"""
        
        html_content += """
        </table>
    </div>
    
    <div class="section">
        <h2>Precision Comparison</h2>
        <table>
            <tr>
                <th>Precision</th>
                <th>Avg Latency (ms)</th>
                <th>Avg Memory (MB)</th>
                <th>Avg Throughput (tokens/sec)</th>
            </tr>
"""
        
        for precision, stats in precision_comparison.items():
            html_content += f"""
            <tr>
                <td>{precision}</td>
                <td>{stats.get("avg_latency_ms", 0):.2f}</td>
                <td>{stats.get("avg_memory_mb", 0):.2f}</td>
                <td>{stats.get("avg_throughput", 0):.2f}</td>
            </tr>
"""
        
        html_content += """
        </table>
    </div>
</body>
</html>
        """
        
        with open(output_path, "w") as f:
            f.write(html_content)
