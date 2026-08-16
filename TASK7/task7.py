
import os
import sys
import glob
import cv2

from task6 import stitch_two_images

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_images_from_folder(folder, extensions=(".jpg", ".jpeg", ".png")):
    paths = []
    for ext in extensions:
        paths.extend(glob.glob(os.path.join(folder, f"*{ext}")))
    # Ignore any previous outputs so re-running doesn't stitch its own results
    paths = sorted(p for p in paths if not os.path.basename(p).startswith("output_"))

    if not paths:
        raise FileNotFoundError(f"No images found in folder: {folder}")

    images = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print(f"[WARN] Could not read {p}, skipping.")
            continue
        images.append(img)
        print(f"[INFO] Loaded {p}  shape={img.shape}")
    return images


def generate_panorama(images, method="sift", downscale_width=None):
    if len(images) < 2:
        raise ValueError("Need at least 2 images to build a panorama.")

    if downscale_width:
        resized = []
        for img in images:
            h, w = img.shape[:2]
            scale = downscale_width / w
            resized.append(cv2.resize(img, (downscale_width, int(h * scale))))
        images = resized

    panorama = images[0]
    for i in range(1, len(images)):
        print(f"[INFO] Stitching image {i + 1}/{len(images)} into panorama...")
        try:
            panorama = stitch_two_images(panorama, images[i], method=method)
        except Exception as e:
            print(f"[WARN] Failed to stitch image {i + 1}: {e}. Skipping it.")
            continue

    return panorama


def build_panorama_from_source(source, method="sift", downscale_width=900):
    if isinstance(source, str) and os.path.isdir(source):
        images = load_images_from_folder(source)
    else:
        paths = source if isinstance(source, list) else [source]
        images = []
        for p in paths:
            img = cv2.imread(p)
            if img is None:
                print(f"[WARN] Could not read {p}, skipping.")
                continue
            images.append(img)
            print(f"[INFO] Loaded {p}  shape={img.shape}")

    return generate_panorama(images, method=method, downscale_width=downscale_width)


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        # Command-line usage: python task7.py <folder_or_files...>
        source = sys.argv[1] if len(sys.argv) == 2 else sys.argv[1:]
    else:
        # No arguments (e.g. hitting "Run" in PyCharm) -> auto-detect images
        # sitting in the same folder as this script.
        source = SCRIPT_DIR
        print(f"[INFO] No arguments given — auto-detecting images in: {SCRIPT_DIR}")

    pano = build_panorama_from_source(source, method="sift", downscale_width=900)

    out_path = os.path.join(SCRIPT_DIR, "output_task7_panorama.jpg")
    cv2.imwrite(out_path, pano)
    print(f"[DONE] Panorama saved to {out_path}  final shape={pano.shape}")