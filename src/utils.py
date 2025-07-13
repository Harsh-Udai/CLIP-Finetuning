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
# === Logging + Saving ===
def save_model(model, path, accelerator, processor=None):
    if accelerator.is_main_process:
        os.makedirs(path, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        
        # 1. Save base model (CLIP) using HuggingFace API
        unwrapped.save_pretrained(path)
        if processor:
            processor.save_pretrained(path)

        # 2. Save LoRA weights separately
        lora_weights = {
            k: v.cpu() for k, v in unwrapped.state_dict().items() if "lora_" in k
        }
        if lora_weights:
            torch.save(lora_weights, os.path.join(path, "lora_weights.pth"))
            print(f"[INFO] Saved {len(lora_weights)} LoRA weights to {os.path.join(path, 'lora_weights.pth')}")
        else:
            print("[WARN] No LoRA weights found to save.")


def log_metrics(step, loss):
    print(f"Step {step} | Loss: {loss:.4f}")
