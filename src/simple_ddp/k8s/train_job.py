apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: simple-ddp
spec:
  pytorchReplicaSpecs:
    Master:
      replicas: 1
      restartPolicy: OnFailure
      template:
        spec:
          containers:
            - name: pytorch
              image: csibilevente14/simple-ddp:2025-10-25-2318
              imagePullPolicy: IfNotPresent
              command: ["torchrun", "--nproc_per_node=2", "train.py"]
    Worker:
      replicas: 1
      restartPolicy: OnFailure
      template:
        spec:
          containers:
            - name: pytorch
              image: csibilevente14/simple-ddp:2025-10-25-2318
              imagePullPolicy: IfNotPresent
              command: ["torchrun", "--nproc_per_node=2", "train.py"]
