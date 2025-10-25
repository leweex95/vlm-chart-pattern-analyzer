import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText
from torch.optim import AdamW
from torch.utils.data import DistributedSampler
from torch.cuda.amp import autocast, GradScaler

# ----- DDP setup -----
dist.init_process_group(backend='nccl')
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

# ----- Load model and processor -----
model_id = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
model = DDP(model, device_ids=[local_rank], output_device=local_rank)

# ----- Dataset and train/test split -----
dataset = load_dataset("json", data_files={"all": "dataset/dataset.json"}, split="all")
train_test = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_test["train"]
test_dataset = train_test["test"]

def preprocess_function(examples):
    encodings = processor(
        images=examples["image"],
        text=examples["text"],
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    encodings["labels"] = encodings["input_ids"].clone()
    return encodings

train_dataset = train_dataset.map(preprocess_function, batched=True, remove_columns=["image", "text"])
test_dataset = test_dataset.map(preprocess_function, batched=True, remove_columns=["image", "text"])

def collate_fn(batch):
    return {k: torch.stack([item[k] for item in batch]) for k in batch[0].keys()}

train_sampler = DistributedSampler(train_dataset)
train_loader = DataLoader(train_dataset, batch_size=4, sampler=train_sampler, collate_fn=collate_fn, drop_last=True)

# ----- Optimizer & scaler -----
optimizer = AdamW(model.parameters(), lr=5e-5)
scaler = GradScaler()

# ----- Training Loop with evaluation -----
model.train()
epochs = 3
grad_accum_steps = 8

for epoch in range(epochs):
    train_sampler.set_epoch(epoch)
    for step, batch in enumerate(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        with autocast(dtype=torch.bfloat16):
            outputs = model(**batch)
            loss = outputs.loss / grad_accum_steps
        scaler.scale(loss).backward()

        if (step + 1) % grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if dist.get_rank() == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item()*grad_accum_steps:.4f}")

    # ----- Evaluate on test set at the end of each epoch -----
    if dist.get_rank() == 0:
        model.eval()
        test_loader = DataLoader(test_dataset, batch_size=4, collate_fn=collate_fn)
        total_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                with autocast(dtype=torch.bfloat16):
                    outputs = model(**batch)
                    total_loss += outputs.loss.item()
                    n_batches += 1
        avg_loss = total_loss / n_batches
        print(f"Epoch {epoch} evaluation: Avg Loss = {avg_loss:.4f}")
        model.train()

# ----- Save model (only rank 0) -----
if dist.get_rank() == 0:
    model.module.save_pretrained("./finetuned_model")
    processor.save_pretrained("./finetuned_model")
