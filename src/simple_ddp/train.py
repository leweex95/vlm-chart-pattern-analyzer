import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def main():
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    rank = dist.get_rank()
    torch.manual_seed(0)

    model = torch.nn.Linear(10, 1).to(rank)
    ddp_model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None)

    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)
    data = torch.randn(100, 10).to(rank)
    target = torch.randn(100, 1).to(rank)

    for _ in range(5):
        optimizer.zero_grad()
        output = ddp_model(data)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        if rank == 0:
            print(f"Loss: {loss.item():.4f}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
