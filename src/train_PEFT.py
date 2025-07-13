import os
import sys
# Add project root to PYTHONPATH
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import yaml
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor, get_cosine_schedule_with_warmup
from accelerate import Accelerator
from peft import get_peft_model, LoraConfig, TaskType

from utils import save_model, log_metrics, set_seed
from coco import get_dataloader
from PEFT.injectLORA import inject_lora

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

    # === Seeding ===
    seed = config.get("seed", int(config["seed"]))

    # Set seed for reproducibilit
    accelerator = Accelerator(mixed_precision=config["mixed_precision"])
    device = accelerator.device

    # Load model and processor
    model = CLIPModel.from_pretrained(config["model_name"]).to(device)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj"]
    )
    # Freeze all base model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Inject LoRA into both towers
    inject_lora(model.text_model)
    inject_lora(model.vision_model)

    processor = CLIPProcessor.from_pretrained(config["model_name"])
    model.train()

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable_params} / {total_params} ({100*trainable_params/total_params:.2f}%)")

    # for name, param in model.named_parameters():
    #     if "lora_" in name:
    #         print(f"{name}: grad={param.grad is not None}, norm={param.data.norm():.4f}")


    # Data
    train_loader = get_dataloader(
        image_dir=config["train_image_dir"],
        annotation_file=config["train_annotation_file"],
        processor=processor,
        batch_size=config["train_batch_size"],
        shuffle=True,
        max_samples=config.get("max_train_samples"),
        seed=int(config["seed"])  # pass seed to the dataloader
    )


    def get_lora_params(model):
        return [p for n, p in model.named_parameters() if 'lora_' in n and p.requires_grad]

    # Optimizer
    optimizer = torch.optim.AdamW(
        get_lora_params(model),
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
    train()
    