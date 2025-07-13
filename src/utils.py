import os
import torch

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
