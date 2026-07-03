"""
Segment the object inside a user-drawn bounding box.

Opens the image → draw bbox (click & drag) → SAM3 segments the best object
inside it → saves overlay of mask + box + score.

Usage:
    uv run python segment_with_bbox.py <image_path> [-o <output.png>]
    
    uv run python segment_with_bbox.py ref.jpg -o result1.png
"""

import argparse
import sys

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

    cv2.namedWindow("Draw bbox around object", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Draw bbox around object", on_mouse)
    print("Click & drag to draw a bbox. Press SPACE to confirm, ESC to quit.")

    while True:
        cv2.imshow("Draw bbox around object", display)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord(" "), 13):
            break
        if key in (27, ord("q")):
            cv2.destroyAllWindows()
            sys.exit(0)

    cv2.destroyAllWindows()
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)
    return img, _scale_point(x0), _scale_point(y0), _scale_point(x1), _scale_point(y1)


def normalize_bbox(bbox_cxcywh, img_w, img_h):
    cx, cy, w, h = bbox_cxcywh
    return [cx / img_w, cy / img_h, w / img_w, h / img_h]


def main():
    parser = argparse.ArgumentParser(
        description="SAM3: segment an object by drawing a bbox"
    )
    parser.add_argument("image_path", help="Path to input image")
    parser.add_argument("--output", "-o", default=None, help="Output image path")
    parser.add_argument("--thresh", type=float, default=0.5,
                        help="Confidence threshold (default: 0.5)")
    args = parser.parse_args()

    # ── Draw bbox ──
    cv_img, x0, y0, x1, y1 = draw_bbox_interactive(args.image_path)
    print(f"Bbox: [{x0}, {y0}, {x1}, {y1}]")

    # ── Load model & run ──
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading SAM3 on {device} ...", end=" ", flush=True)
    model = build_sam3_image_model(device=device)
    processor = Sam3Processor(model, confidence_threshold=args.thresh, device=device)
    print("done")

    image = Image.open(args.image_path).convert("RGB")
    img_w, img_h = image.size

    state = processor.set_image(image)

    box_xywh = [x0, y0, x1 - x0, y1 - y0]
    box_cxcywh = box_xywh_to_cxcywh(torch.tensor(box_xywh).view(-1, 4))
    norm_box = normalize_bbox(box_cxcywh.flatten().tolist(), img_w, img_h)

    state = processor.add_geometric_prompt(state=state, box=norm_box, label=True)

    masks = state["masks"]
    scores = state["scores"]
    boxes = state["boxes"]

    if len(masks) == 0:
        print("No objects found")
        return

    best_idx = scores.argmax().item()
    best_mask = masks[best_idx].squeeze().cpu().numpy()
    best_score = scores[best_idx].item()
    best_box = boxes[best_idx].tolist()
    print(f"Best: score={best_score:.3f} box={[round(v, 1) for v in best_box]}")

    # ── Render ──
    overlay = image.copy().convert("RGBA")
    mask_pixels = np.zeros((img_h, img_w, 4), dtype=np.uint8)
    mask_pixels[best_mask, 1] = 255
    mask_pixels[best_mask, 3] = 120
    mask_overlay = Image.fromarray(mask_pixels, mode="RGBA")
    overlay = Image.alpha_composite(overlay, mask_overlay)

    draw = ImageDraw.Draw(overlay)
    draw.rectangle(best_box, outline="red", width=3)
    draw.text((best_box[0], best_box[1] - 12), f"{best_score:.2f}", fill="red")

    out_path = args.output or args.image_path.rsplit(".", 1)[0] + "_segmented.png"
    overlay.save(out_path)
    print(f"Saved result to {out_path}")


if __name__ == "__main__":
    main()
