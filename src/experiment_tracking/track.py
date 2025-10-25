import os
import mlflow

# Use environment variable if set, otherwise fall back to default
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow.mlflow.svc.cluster.local:5000")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("hello-world")

with mlflow.start_run():
    mlflow.log_param("param1", 42)
    mlflow.log_metric("metric1", 0.99)
    print("Hello, MLflow!")
    print(f"Tracking URI: {tracking_uri}")
