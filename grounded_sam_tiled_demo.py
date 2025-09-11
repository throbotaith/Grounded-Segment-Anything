import argparse
import os
import sys
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image

sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "segment_anything"))

# Grounding DINO
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

# Segment Anything
from segment_anything import sam_model_registry, sam_hq_model_registry, SamPredictor

import cv2
import matplotlib.pyplot as plt
from torchvision.ops import nms


def build_gdino(model_config_path: str, model_checkpoint_path: str, bert_base_uncased_path: str, device: str):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    args.bert_base_uncased_path = bert_base_uncased_path
    model = build_model(args)
    ckpt = torch.load(model_checkpoint_path, map_location="cpu")
    _ = model.load_state_dict(clean_state_dict(ckpt["model"]), strict=False)
    model.eval()
    return model


def tile_coords(width: int, height: int, tile: int, overlap: int) -> List[Tuple[int, int, int, int]]:
    xs = list(range(0, max(width - tile, 0) + 1, max(tile - overlap, 1)))
    ys = list(range(0, max(height - tile, 0) + 1, max(tile - overlap, 1)))
    if not xs:
        xs = [0]
    if not ys:
        ys = [0]
    coords = []
    for y in ys:
        for x in xs:
            x1 = min(x + tile, width)
            y1 = min(y + tile, height)
            coords.append((x, y, x1, y1))
    return coords


@torch.no_grad()
def detect_on_tile(model, image_tensor, caption: str, box_threshold: float, text_threshold: float, device: str):
    caption = caption.lower().strip()
    if not caption.endswith("."):
        caption += "."

    image_tensor = image_tensor.to(device)
    outputs = model(image_tensor[None], captions=[caption])
    logits = outputs["pred_logits"].sigmoid().cpu()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4) in cxcywh normalized

    # filter by score threshold on any token
    filt = logits.max(dim=1)[0] > box_threshold
    logits_f = logits[filt]
    boxes_f = boxes[filt]

    # phrases and confidence per box
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    phrases = []
    confs = []
    for lg, _ in zip(logits_f, boxes_f):
        phrases.append(get_phrases_from_posmap(lg > text_threshold, tokenized, tokenlizer) + f"({str(lg.max().item())[:4]})")
        confs.append(lg.max().item())
    scores = torch.tensor(confs, dtype=torch.float32)
    return boxes_f, phrases, scores


def main():
    parser = argparse.ArgumentParser("Grounded-SAM Tiled Demo (4K-friendly)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--grounded_checkpoint", type=str, required=True)
    parser.add_argument("--input_image", type=str, required=True)
    parser.add_argument("--text_prompt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--box_threshold", type=float, default=0.3)
    parser.add_argument("--text_threshold", type=float, default=0.25)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--bert_base_uncased_path", type=str, default=None)

    # Tiling params
    parser.add_argument("--tile_size", type=int, default=1408, help="tile size in pixels")
    parser.add_argument("--tile_overlap", type=int, default=256, help="overlap between tiles in pixels")
    parser.add_argument("--nms_iou", type=float, default=0.5, help="NMS IoU threshold to merge tiles")

    # SAM params
    parser.add_argument("--sam_version", type=str, default="vit_h", help="vit_b / vit_l / vit_h")
    parser.add_argument("--sam_checkpoint", type=str, required=True)
    parser.add_argument("--use_sam_hq", action="store_true")
    parser.add_argument("--sam_hq_checkpoint", type=str, default=None)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load full-res image
    image_pil = Image.open(args.input_image).convert("RGB")
    W, H = image_pil.size

    # Build GroundingDINO once
    model = build_gdino(args.config, args.grounded_checkpoint, args.bert_base_uncased_path, device=args.device)

    # Simple transform without global resize (keep tile native size)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    all_boxes_xyxy = []
    all_phrases = []
    all_scores = []

    # Iterate tiles
    for (x0, y0, x1, y1) in tile_coords(W, H, args.tile_size, args.tile_overlap):
        tile = image_pil.crop((x0, y0, x1, y1))
        tile_w, tile_h = tile.size

        tile_tensor, _ = transform(tile, None)
        boxes_f, phrases, scores = detect_on_tile(
            model, tile_tensor, args.text_prompt, args.box_threshold, args.text_threshold, args.device
        )

        if boxes_f.numel() == 0:
            continue

        # Convert to absolute XYXY in tile coords then offset to global
        boxes_abs = boxes_f.clone()
        boxes_abs[:, 0] *= tile_w
        boxes_abs[:, 1] *= tile_h
        boxes_abs[:, 2] *= tile_w
        boxes_abs[:, 3] *= tile_h
        # cxcywh -> xyxy
        boxes_abs[:, :2] = boxes_abs[:, :2] - boxes_abs[:, 2:] / 2
        boxes_abs[:, 2:] = boxes_abs[:, :2] + boxes_abs[:, 2:]
        # offset
        boxes_abs[:, [0, 2]] += x0
        boxes_abs[:, [1, 3]] += y0

        all_boxes_xyxy.append(boxes_abs)
        all_phrases.extend(phrases)
        all_scores.append(scores)

    if not all_boxes_xyxy:
        print("No boxes detected on any tile.")
        # Still save the raw image
        image_pil.save(os.path.join(args.output_dir, "raw_image.jpg"))
        return

    boxes_xyxy = torch.cat(all_boxes_xyxy, dim=0)
    scores = torch.cat(all_scores, dim=0)

    # NMS to merge duplicates across overlaps
    keep = nms(boxes_xyxy, scores, args.nms_iou)
    boxes_xyxy = boxes_xyxy[keep]
    kept_phrases = [all_phrases[i] for i in keep.tolist()]

    # Prepare SAM on full image
    if args.use_sam_hq:
        sam = sam_hq_model_registry[args.sam_version](checkpoint=args.sam_hq_checkpoint)
    else:
        sam = sam_model_registry[args.sam_version](checkpoint=args.sam_checkpoint)
    sam.to(args.device)
    predictor = SamPredictor(sam)

    image = cv2.cvtColor(cv2.imread(args.input_image), cv2.COLOR_BGR2RGB)
    predictor.set_image(image)

    # transform boxes for SAM
    transformed_boxes = predictor.transform.apply_boxes_torch(boxes_xyxy, image.shape[:2]).to(args.device)
    masks, _, _ = predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )

    # Visualize
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for m in masks:
        h, w = m.shape[-2:]
        mask_image = m.reshape(h, w, 1).cpu().numpy() * np.array([[[(30/255, 144/255, 1.0, 0.6)]]])
        plt.imshow(mask_image)
    for box, label in zip(boxes_xyxy, kept_phrases):
        x0, y0, x1, y1 = box.tolist()
        plt.gca().add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor='green', facecolor=(0,0,0,0), lw=2))
        plt.gca().text(x0, y0, label)
    plt.axis('off')
    plt.savefig(os.path.join(args.output_dir, "grounded_sam_tiled_output.jpg"), bbox_inches='tight', dpi=300, pad_inches=0)


if __name__ == "__main__":
    main()

