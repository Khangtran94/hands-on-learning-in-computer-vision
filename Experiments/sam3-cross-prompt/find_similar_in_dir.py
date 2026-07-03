"""
Find visually similar objects across multiple images using a bbox exemplar.

Stacks the reference image (with user-drawn bbox) on top of each target image
vertically, then runs SAM3 on the composite. SAM3's ROI-aligned visual features
from the reference bbox guide detection across both halves.

Usage:
    uv run python find_similar_in_dir.py <reference.jpg> <target_dir/> \\
        [-o output_dir/] [--thresh 0.5]

Example:
>>>    uv run python find_similar_in_dir.py ref.jpg ./images/ -o ./results/

>>>    uv run python find_similar_in_dir.py ref.jpg ./img/ -o ./results/
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.backends.cuda

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
torch.inference_mode().__enter__()

import os.path as _p
_script_dir = _p.dirname(_p.abspath(__file__)) if __file__ else ""
sys.path = [p for p in sys.path if p and _p.abspath(p) != _script_dir]

from PIL import Image, ImageDraw
from sam3 import build_sam3_image_model
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.model.sam3_image_processor import Sam3Processor


RES = 1008
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
]
IMG_EXTS = {".jpg", ".jpeg", ".png"}


def _resize_for_display(img, max_w=1200, max_h=900):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        resized = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))),
                             interpolation=cv2.INTER_AREA)
        return resized, scale
    return img.copy(), 1.0


def draw_bbox_interactive(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: cannot load '{image_path}'", file=sys.stderr)
        sys.exit(1)

    display_img, scale = _resize_for_display(img)
    display = display_img.copy()
    orig = display_img.copy()
    x0 = y0 = x1 = y1 = -1
    drawing = False

    def _scale_point(pt):
        return max(int(round(pt / scale)), 0)

    def on_mouse(event, x, y, flags, param):
        nonlocal x0, y0, x1, y1, drawing, display
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            x0, y0 = x, y
            x1, y1 = x, y
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            x1, y1 = x, y
            display = orig.copy()
            cv2.rectangle(display, (x0, y0), (x1, y1), (0, 255, 0), 2)
            w, h = abs(x1 - x0), abs(y1 - y0)
            cv2.putText(display, f"({x0},{y0}) {w}x{h}", (x0, max(y0 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            x1, y1 = x, y
            display = orig.copy()
            cv2.rectangle(display, (x0, y0), (x1, y1), (0, 255, 0), 2)
            w, h = abs(x1 - x0), abs(y1 - y0)
            cv2.putText(display, f"({x0},{y0}) {w}x{h}", (x0, max(y0 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.namedWindow("Draw bbox around exemplar object", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Draw bbox around exemplar object", on_mouse)
    print("Click & drag to draw a bbox. Press SPACE to confirm, ESC to quit.")

    while True:
        cv2.imshow("Draw bbox around exemplar object", display)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord(" "), 13):
            break
        if key in (27, ord("q")):
            cv2.destroyAllWindows()
            sys.exit(0)

    cv2.destroyAllWindows()
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)
    return _scale_point(x0), _scale_point(y0), _scale_point(x1), _scale_point(y1)


def scale_bbox_to_res(bbox, orig_w, orig_h):
    x0, y0, x1, y1 = bbox
    sx = RES / orig_w
    sy = RES / orig_h
    return [x0 * sx, y0 * sy, x1 * sx, y1 * sy]


def bbox_xyxy_to_cxcywh(x0, y0, x1, y1):
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    w = x1 - x0
    h = y1 - y0
    return [cx, cy, w, h]


def render_overlay(pil_image, masks, boxes, scores):
    img_w, img_h = pil_image.size
    overlay = pil_image.copy().convert("RGBA")

    for i in range(len(masks)):
        color = COLORS[i % len(COLORS)]
        mask_np = masks[i].squeeze().cpu().numpy()
        pixels = np.zeros((img_h, img_w, 4), dtype=np.uint8)
        pixels[mask_np, 0] = color[0]
        pixels[mask_np, 1] = color[1]
        pixels[mask_np, 2] = color[2]
        pixels[mask_np, 3] = 120
        overlay = Image.alpha_composite(
            overlay, Image.fromarray(pixels, mode="RGBA")
        )
        draw = ImageDraw.Draw(overlay)
        draw.rectangle(boxes[i].tolist(), outline=color, width=3)
        tag = f"{i} {scores[i].item():.2f}"
        draw.text((boxes[i][0], boxes[i][1] - 14), tag, fill=color + (255,))

    return overlay


def main():
    parser = argparse.ArgumentParser(
        description="Find similar objects across images using a bbox exemplar"
    )
    parser.add_argument("reference", help="Reference image with the exemplar object")
    parser.add_argument("target_dir", help="Directory with target images to search")
    parser.add_argument("--output", "-o", default=None, help="Output directory")
    parser.add_argument("--thresh", type=float, default=0.5,
                        help="Confidence threshold (default: 0.5)")
    args = parser.parse_args()

    target_dir = Path(args.target_dir)
    if not target_dir.is_dir():
        print(f"Error: '{args.target_dir}' is not a directory", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output) if args.output else target_dir / "similar_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Phase A: draw bbox on reference ──
    print("=== Phase A: Draw bbox on reference image ===")
    ref_bbox_orig = draw_bbox_interactive(args.reference)
    x0, y0, x1, y1 = ref_bbox_orig
    print(f"  bbox: [{x0}, {y0}, {x1}, {y1}]")

    # Open reference as PIL
    ref_pil = Image.open(args.reference).convert("RGB")
    ref_w, ref_h = ref_pil.size

    # Scale bbox to 1008x1008 reference space
    ref_bbox_1008 = scale_bbox_to_res(ref_bbox_orig, ref_w, ref_h)
    ref_cxcywh_1008 = bbox_xyxy_to_cxcywh(*ref_bbox_1008)

    # ── Phase B: scan target directory ──
    target_paths = sorted(
        p for p in target_dir.iterdir()
        if p.suffix.lower() in IMG_EXTS and p.name != Path(args.reference).name
    )
    if not target_paths:
        print("No target images found in directory")
        sys.exit(0)

    print(f"\n=== Phase B: Searching {len(target_paths)} image(s) ===")

    # Load model once
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SAM3 on {device} ...", end=" ", flush=True)
    model = build_sam3_image_model(device=device)
    processor = Sam3Processor(model, confidence_threshold=args.thresh, device=device)
    print("done")

    # Pre-resize reference to 1008x1008
    ref_resized = ref_pil.resize((RES, RES), Image.Resampling.LANCZOS)

    all_results = {}

    for target_path in target_paths:
        print(f"\n  [{target_path.name}] ", end="", flush=True)

        target_pil = Image.open(target_path).convert("RGB")
        tgt_w, tgt_h = target_pil.size

        # Pre-resize target to 1008x1008
        tgt_resized = target_pil.resize((RES, RES), Image.Resampling.LANCZOS)

        # Create vertical composite: ref on top, target on bottom
        composite = Image.new("RGB", (RES, RES * 2))
        composite.paste(ref_resized, (0, 0))
        composite.paste(tgt_resized, (0, RES))

        # Normalize bbox for composite (1008 x 2016)
        cx, cy, w, h = ref_cxcywh_1008
        norm_box = [cx / RES, cy / (RES * 2), w / RES, h / (RES * 2)]

        # Run SAM3
        state = processor.set_image(composite)
        state = processor.add_geometric_prompt(state=state, box=norm_box, label=True)

        masks = state["masks"]
        scores = state["scores"]
        boxes = state["boxes"]  # [x0,y0,x1,y1] in composite pixel coords

        if len(masks) == 0:
            print("no matches")
            all_results[target_path.name] = {"count": 0, "objects": []}
            continue

        # Filter: keep only objects in bottom half (target image area)
        target_masks, target_scores, target_boxes = [], [], []
        for i in range(len(masks)):
            box = boxes[i]  # 1D tensor [x0, y0, x1, y1]
            center_y = (box[1].item() + box[3].item()) / 2.0
            if center_y > RES:
                target_scores.append(scores[i])
                # Crop mask to bottom half (target region) and resize to original target size
                mask_full = masks[i]  # shape: [1, 2016, 1008] boolean
                mask_target_half = mask_full[:, RES:, :]  # crop bottom half: [1, 1008, 1008]
                mask_resized = (
                    mask_target_half.float()
                    .unsqueeze(0)  # add batch for interpolate: [1, 1, 1008, 1008]
                    .to(device)
                )
                mask_resized = torch.nn.functional.interpolate(
                    mask_resized, size=(tgt_h, tgt_w), mode="nearest"
                ).squeeze(0)  # [1, tgt_h, tgt_w]
                target_masks.append(mask_resized.bool())
                # Map coords from composite space → original target space
                x0_c, y0_c = box[0].item(), box[1].item()
                x1_c, y1_c = box[2].item(), box[3].item()
                x0_t = round(x0_c * (tgt_w / RES), 1)
                y0_t = round((y0_c - RES) * (tgt_h / RES), 1)
                x1_t = round(x1_c * (tgt_w / RES), 1)
                y1_t = round((y1_c - RES) * (tgt_h / RES), 1)
                target_boxes.append(torch.tensor([x0_t, y0_t, x1_t, y1_t]))

        if len(target_masks) == 0:
            print("no matches in target region")
            all_results[target_path.name] = {"count": 0, "objects": []}
            continue

        print(f"found {len(target_masks)} match(es)")
        objects_info = []
        for i in range(len(target_masks)):
            score_val = target_scores[i].item()
            box_list = target_boxes[i].tolist()
            objects_info.append({"score": round(score_val, 3), "box": box_list})
            print(f"    [{i}] score={score_val:.3f} box={[round(v,1) for v in box_list]}")

        all_results[target_path.name] = {
            "count": len(target_masks),
            "objects": objects_info,
        }

        # Save overlay
        overlay = render_overlay(
            target_pil, target_masks, target_boxes, target_scores
        )
        out_path = out_dir / f"{target_path.stem}_similar.png"
        overlay.save(out_path)

    # ── Phase C: save metadata ──
    meta_path = out_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(all_results, f, indent=2)

    total_matches = sum(v["count"] for v in all_results.values())
    print(f"\n=== Done ===")
    print(f"  Total matches across {len(target_paths)} images: {total_matches}")
    print(f"  Results saved to: {out_dir}/")


if __name__ == "__main__":
    main()
