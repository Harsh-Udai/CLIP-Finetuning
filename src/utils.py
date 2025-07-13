import os
import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# === Logging + Saving ===
def save_model(model, path, accelerator, processor=None):
    if accelerator.is_main_process:
        os.makedirs(path, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(path)
        if processor:
            processor.save_pretrained(path)

def log_metrics(step, loss):
    print(f"Step {step} | Loss: {loss:.4f}")
