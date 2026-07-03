"""
Interactive bbox selector for images.

Click and drag to draw a bounding box. The coordinates [x0, y0, x1, y1]
are printed to stdout for use with segment_with_bbox.py.

Usage:
    uv run python get_bbox.py <image_path>

Controls:
    - Left-click & drag: draw bounding box
    - SPACE / ENTER: print coordinates and exit
    - ESC / q: quit without printing
"""

import argparse
import sys

import os.path as _p
_script_dir = _p.dirname(_p.abspath(__file__)) if __file__ else ""
sys.path = [p for p in sys.path if p and _p.abspath(p) != _script_dir]

import cv2

def main():
    parser = argparse.ArgumentParser(
        description="Draw a bounding box on an image and print its coordinates"
    )
    parser.add_argument("image_path", help="Path to input image")
    args = parser.parse_args()

    img = cv2.imread(args.image_path)
    if img is None:
        print(f"Error: could not load image '{args.image_path}'", file=sys.stderr)
        sys.exit(1)

    def _resize_for_display(img_, max_w=1200, max_h=900):
        h, w = img_.shape[:2]
        scale = min(max_w / w, max_h / h, 1.0)
        if scale < 1.0:
            resized = cv2.resize(img_, (int(round(w * scale)), int(round(h * scale))),
                                 interpolation=cv2.INTER_AREA)
            return resized, scale
        return img_.copy(), 1.0

    display_img, scale = _resize_for_display(img)
    display = display_img.copy()
    orig_display = display_img.copy()

    def _scale_point(pt):
        return max(int(round(pt / scale)), 0)

    x0 = y0 = x1 = y1 = -1
    drawing = False

    def on_mouse(event, x, y, flags, param):
        nonlocal x0, y0, x1, y1, drawing, display

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            x0, y0 = x, y
            x1, y1 = x, y

        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            x1, y1 = x, y
            display = orig_display.copy()
            cv2.rectangle(display, (x0, y0), (x1, y1), (0, 255, 0), 2)
            w = abs(x1 - x0)
            h = abs(y1 - y0)
            label = f"({x0},{y0})  {w}x{h}"
            cv2.putText(display, label, (x0, y0 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            x1, y1 = x, y
            display = orig_display.copy()
            cv2.rectangle(display, (x0, y0), (x1, y1), (0, 255, 0), 2)
            w = abs(x1 - x0)
            h = abs(y1 - y0)
            label = f"({x0},{y0})  {w}x{h}"
            cv2.putText(display, label, (x0, y0 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.namedWindow("Draw bbox", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Draw bbox", on_mouse)

    while True:
        cv2.imshow("Draw bbox", display)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord(" "), 13):  # SPACE or ENTER
            if x0 >= 0 and x1 >= 0:
                _x0, _x1 = min(x0, x1), max(x0, x1)
                _y0, _y1 = min(y0, y1), max(y0, y1)
                print(f"{_scale_point(_x0)} {_scale_point(_y0)} {_scale_point(_x1)} {_scale_point(_y1)}")
            break
        elif key in (27, ord("q")):  # ESC or q
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
