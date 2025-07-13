import os
import sys
# Add project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import yaml
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, get_cosine_schedule_with_warmup
from accelerate import Accelerator


from utils import save_model, log_metrics
from coco import get_dataloader

# === Config ===
def load_config(path="config/train_config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# === Training ===
def train():
    config = load_config()
    config["learning_rate"] = float(config["learning_rate"])
    config["weight_decay"] = float(config["weight_decay"])
    config["train_batch_size"] = int(config["train_batch_size"])

    accelerator = Accelerator(mixed_precision=config["mixed_precision"])
    device = accelerator.device

    # Load model and processor
    model = CLIPModel.from_pretrained(config["model_name"]).to(device)
    processor = CLIPProcessor.from_pretrained(config["model_name"])
    model.train()

    # Data
    train_loader = get_dataloader(
        image_dir=config["train_image_dir"],
        annotation_file=config["train_annotation_file"],
        processor=processor,
        batch_size=config["train_batch_size"],
        shuffle=True,
        max_samples=config.get("max_train_samples")
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]
    )

    # Scheduler
    total_steps = len(train_loader) * config["num_epochs"]
    warmup_steps = config.get("warmup_steps", int(0.05 * total_steps))

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Prepare everything
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )

    global_step = 0
    for epoch in range(config["num_epochs"]):
        total_loss = 0
        for step, batch in enumerate(train_loader):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"]
            )
            logits_image = outputs.logits_per_image
            logits_text = outputs.logits_per_text

            labels = torch.arange(len(logits_image), device=logits_image.device)
            loss = (
                torch.nn.functional.cross_entropy(logits_image, labels) +
                torch.nn.functional.cross_entropy(logits_text, labels)
            ) / 2

            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            scheduler.step()

            global_step += 1
            total_loss += loss.item()

            if global_step % config["logging_steps"] == 0:
                log_metrics(global_step, loss.item())

        print(f"Epoch {epoch+1} completed. Avg Loss: {total_loss / len(train_loader):.4f}")

        if config["checkpointing"] and (epoch + 1) % config["save_every"] == 0:
            save_model(model, os.path.join(config["output_dir"], f"checkpoint-epoch{epoch+1}"), accelerator)

    save_model(model, os.path.join(config["output_dir"], "final"), accelerator, processor)

if __name__ == "__main__":
    # train()
    