from datasets import load_dataset
from transformers import CLIPProcessor
from torch.utils.data import DataLoader, Dataset
import torch
import sys 
import os

# === Dataset ===
class LocalCocoCaptionDataset(Dataset):
    def __init__(self, image_dir, annotation_file, processor, max_samples=None):
        self.image_dir = image_dir
        self.processor = processor

        import json
        with open(annotation_file, 'r') as f:
            data = json.load(f)

        self.annotations = data['annotations']
        self.image_id_to_filename = {
            img['id']: img['file_name']
            for img in data['images']
        }

        if max_samples:
            self.annotations = self.annotations[:max_samples]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        image_id = ann['image_id']
        caption = ann['caption']
        filename = self.image_id_to_filename[image_id]
        image_path = os.path.join(self.image_dir, filename)

        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(text=caption, images=image, return_tensors="pt", padding="max_length", truncation=True)

        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
        }

def get_dataloader(image_dir, annotation_file, processor, batch_size, shuffle=True, max_samples=None):
    dataset = LocalCocoCaptionDataset(image_dir, annotation_file, processor, max_samples=max_samples)

    def collate_fn(batch):
        return {
            "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
            "input_ids": torch.stack([x["input_ids"] for x in batch]),
            "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        }

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
