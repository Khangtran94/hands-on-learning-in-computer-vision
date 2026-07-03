# SAM3-visual-cross-prompt

Interactive object segmentation tools built on [SAM3](https://github.com/facebookresearch/sam3). Draw a bounding box around an object — the model segments it, finds visually similar objects in the same image, or searches across a directory of images.

## Setup

Requires Python 3.11, a CUDA 12.8-capable GPU, and `uv`.

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install SAM3 in editable mode
git clone https://github.com/facebookresearch/sam3
cd sam3
uv pip install -e .

# Patch: Triton connected-components crashes on consumer GPUs →
# force CPU fallback unconditionally
sed -i 's/HAVE_TRITON_CCL = True/HAVE_TRITON_CCL = False/' \
  sam3/sam3/perflib/connected_components.py
```

### GPU performance notes

The scripts set these flags for efficient consumer-GPU inference:

```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
torch.inference_mode().__enter__()
```

## Scripts

### `segment_with_bbox.py`

Segment the best object inside a user-drawn bounding box.

```
uv run python segment_with_bbox.py image.jpg
```

Click and drag to draw a bbox, press SPACE to confirm. Saves an overlay of the mask, box, and confidence score.

### `segment_all_similar.py`

Find every object in the **same image** that looks like the exemplar inside the bbox.

```
uv run python segment_all_similar.py image.jpg
```

Each detection is drawn in a different color with index and score. Results saved to `*_all_similar.png`.

### `find_similar_in_dir.py`

Search a directory of images for objects visually similar to the exemplar.

```
uv run python find_similar_in_dir.py ref.jpg ./images/ -o ./results/
```

**How it works:** The reference and each target image are resized to 1008×1008 and stacked into a **1008×2016 vertical composite**. SAM3's `add_geometric_prompt` extracts ROI-aligned visual features (color, texture, shape) from the reference bbox via its `SequenceGeometryEncoder`. The decoder's global cross-attention compares these features against every patch token in the composite — objects in the bottom (target) half that visually match get masks and bounding boxes. Detections in the target region (center y > 1008) are cropped, resized back to original target dimensions, and saved as overlays + a `metadata.json` summary.

```
uv run python find_similar_in_dir.py ref.jpg ./images/ -o ./results/ --thresh 0.3
```

### `get_bbox.py`

Just print bounding box coordinates — no model inference.

```
uv run python get_bbox.py image.jpg
```

Prints `x0 y0 x1 y1` to stdout. Useful for piping into other tools.

## Limitations

- The cross-image search (`find_similar_in_dir.py`) uses a vertical composite approach that is conceptually correct but **not yet validated** on real multi-image datasets.
- Mixed aspect ratios are not handled — images are resized to 1008×1008, which distorts non-square inputs. Pad-to-square preprocessing is planned.
- All interactive scripts depend on OpenCV's `cv2.imshow`, so they need a display (won't work in a headless terminal without X forwarding).
