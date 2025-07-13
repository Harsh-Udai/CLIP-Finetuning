import os
import json
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from transformers import CLIPProcessor, CLIPModel
from pycocotools.coco import COCO

class CocoValDataset(Dataset):
    def __init__(self, image_dir, annotation_file):
        self.coco = COCO(annotation_file)
        self.image_dir = os.path.abspath(image_dir)
        self.image_ids = list(self.coco.imgs.keys())
        self.annotations = self.coco.loadAnns(self.coco.getAnnIds())
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # Build list of (caption, image_id)
        self.caption_texts = []
        self.caption_img_ids = []
        for ann in self.annotations:
            self.caption_texts.append(ann["caption"])
            self.caption_img_ids.append(ann["image_id"])

    def __len__(self):
        return len(self.image_ids)

    def load_image_by_id(self, img_id):
        file_name = self.coco.imgs[img_id]["file_name"]
        image_path = os.path.join(self.image_dir, file_name)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Missing image: {image_path}")
        return Image.open(image_path).convert("RGB")

    def get_all_captions(self):
        return self.caption_texts, self.caption_img_ids

def compute_clip_embeddings(model, processor, images=None, texts=None, batch_size=64, device="cuda"):
    all_embeddings = []
    model.eval()

    if images:
        for i in tqdm(range(0, len(images), batch_size), desc="Encoding images"):
            batch_imgs = [images[j] for j in range(i, min(i + batch_size, len(images)))]
            inputs = processor(images=batch_imgs, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                emb = model.get_image_features(**inputs)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                all_embeddings.append(emb)
        return torch.cat(all_embeddings)

    if texts:
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
            inputs = processor(text=texts[i:i+batch_size], return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                emb = model.get_text_features(**inputs)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                all_embeddings.append(emb)
        return torch.cat(all_embeddings)


def evaluate_clip_on_coco(image_dir, annotation_file, device="cuda"):
    print(f"Loading dataset from:\n - images: {image_dir}\n - annotations: {annotation_file}")
    dataset = CocoValDataset(image_dir, annotation_file)
    model = CLIPModel.from_pretrained("saved_model/final").to(device)
    processor = CLIPProcessor.from_pretrained("saved_model/final")

    # model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    # processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # 1. Encode all images
    print("Loading and encoding images...")
    image_ids = dataset.image_ids
    images = []
    for img_id in tqdm(image_ids, desc="Loading images"):
        images.append(dataset.load_image_by_id(img_id))
    image_embs = compute_clip_embeddings(model, processor, images=images, device=device)

    # 2. Encode all captions
    print("Encoding captions...")
    captions, cap_img_ids = dataset.get_all_captions()
    text_embs = compute_clip_embeddings(model, processor, texts=captions, device=device)

    # 3. Compute similarity matrices
    print("Computing similarity scores...")
    sims_i2t = image_embs @ text_embs.T  # [N_images, N_captions]
    sims_t2i = text_embs @ image_embs.T  # [N_captions, N_images]

    # 4. Build ground truth
    img_id_to_idx = {img_id: i for i, img_id in enumerate(image_ids)}
    gt_i2t = [[j for j, cap_id in enumerate(cap_img_ids) if cap_img_ids[j] == img_id] for img_id in image_ids]
    gt_t2i = [img_id_to_idx[img_id] for img_id in cap_img_ids]

    # 5. Evaluate I2T: rank captions for each image
    print("Evaluating Image-to-Text retrieval...")
    i2t_ranks = []
    for sims_row, gt_list in tqdm(zip(sims_i2t, gt_i2t), total=len(gt_i2t), desc="I2T ranking"):
        sorted_idx = sims_row.argsort(descending=True)
        first_correct = min([sorted_idx.tolist().index(gt) for gt in gt_list])
        i2t_ranks.append(first_correct)

    # 6. Evaluate T2I: rank images for each caption
    print("Evaluating Text-to-Image retrieval...")
    t2i_ranks = []
    for sims_row, gt in tqdm(zip(sims_t2i, gt_t2i), total=len(gt_t2i), desc="T2I ranking"):
        sorted_idx = sims_row.argsort(descending=True)
        rank = sorted_idx.tolist().index(gt)
        t2i_ranks.append(rank)

    # 7. Compute Recall@K
    def compute_recall(ranks, ks=(1, 5, 10)):
        return {f"R@{k}": sum([r < k for r in ranks]) / len(ranks) for k in ks}

    print("\n📊 Final Results:")
    print("🔁 Image-to-Text:", compute_recall(i2t_ranks))
    print("🔁 Text-to-Image:", compute_recall(t2i_ranks))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--val_image_dir", type=str, required=True)
    parser.add_argument("--val_annotation_file", type=str, required=True)
    args = parser.parse_args()

    evaluate_clip_on_coco(
        image_dir=args.val_image_dir,
        annotation_file=args.val_annotation_file,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
