
import cv2
import numpy as np
from task5 import detect_and_describe, match_keypoints, estimate_homography


def _feather_weight_map(mask):

    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    if dist.max() > 0:
        dist = dist / dist.max()
    return dist


def blend_images(canvas1, canvas2):

    gray1 = cv2.cvtColor(canvas1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(canvas2, cv2.COLOR_BGR2GRAY)

    mask1 = (gray1 > 0).astype(np.uint8)
    mask2 = (gray2 > 0).astype(np.uint8)

    w1 = _feather_weight_map(mask1)
    w2 = _feather_weight_map(mask2)

    # Where only one image has data, force its weight to 1 so we don't
    # darken regions that aren't actually overlapping.
    only1 = (mask1 == 1) & (mask2 == 0)
    only2 = (mask2 == 1) & (mask1 == 0)
    w1[only1] = 1.0
    w2[only1] = 0.0
    w1[only2] = 0.0
    w2[only2] = 1.0

    total = w1 + w2
    total[total == 0] = 1.0  # avoid divide-by-zero outside both images
    w1n = (w1 / total)[..., None]
    w2n = (w2 / total)[..., None]

    blended = (canvas1.astype(np.float32) * w1n +
               canvas2.astype(np.float32) * w2n)
    return np.clip(blended, 0, 255).astype(np.uint8)


def crop_black_borders(image):
    """Crop away the empty black borders left over after warping."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    coords = cv2.findNonZero(gray)
    x, y, w, h = cv2.boundingRect(coords)
    return image[y:y + h, x:x + w]


def stitch_two_images(image1, image2, method="sift"):

    kp1, desc1 = detect_and_describe(image1, method)
    kp2, desc2 = detect_and_describe(image2, method)
    matches = match_keypoints(desc1, desc2, method)
    H, mask = estimate_homography(kp1, kp2, matches)

    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]

    # Figure out the canvas size needed to fit both images after warping,
    # by warping the corners of image2 and image1 together.
    corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
    corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
    warped_corners2 = cv2.perspectiveTransform(corners2, H)
    all_corners = np.concatenate((corners1, warped_corners2), axis=0)

    [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)

    # Translation so all coordinates become positive
    translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float64)
    canvas_size = (x_max - x_min, y_max - y_min)

    # Warp image2 with homography (translated into the shared canvas)
    canvas2 = cv2.warpPerspective(image2, translation @ H, canvas_size)

    # Place image1 with only the translation (no perspective change)
    canvas1 = cv2.warpPerspective(image1, translation, canvas_size)

    blended = blend_images(canvas1, canvas2)
    result = crop_black_borders(blended)
    return result


if __name__ == "__main__":
    img1 = cv2.imread("graf1.png")
    img2 = cv2.imread("graf3.png")

    if img1 is None or img2 is None:
        raise FileNotFoundError("Could not load images/graf_pair/graf1.png or graf3.png")

    stitched = stitch_two_images(img1, img2, method="sift")
    cv2.imwrite("output_task6_stitched_blended.jpg", stitched)
    print("[DONE] Saved output_task6_stitched_blended.jpg", stitched.shape)