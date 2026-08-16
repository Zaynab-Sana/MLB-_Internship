
import cv2
import numpy as np


def detect_and_describe(image, method="sift"):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if method == "sift":
        detector = cv2.SIFT_create()
    elif method == "orb":
        detector = cv2.ORB_create(nfeatures=5000)
    else:
        raise ValueError("method must be 'sift' or 'orb'")

    keypoints, descriptors = detector.detectAndCompute(gray, None)
    return keypoints, descriptors


def match_keypoints(desc1, desc2, method="sift", ratio=0.75):
    if method == "sift":
        norm = cv2.NORM_L2
    else:  # orb -> binary descriptor
        norm = cv2.NORM_HAMMING

    bf = cv2.BFMatcher(norm)
    raw_matches = bf.knnMatch(desc1, desc2, k=2)

    good_matches = []
    for pair in raw_matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < ratio * n.distance:
                good_matches.append(m)

    return good_matches


def estimate_homography(kp1, kp2, matches, reproj_thresh=4.0):
    if len(matches) < 4:
        raise ValueError("Need at least 4 matches to estimate a homography.")

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # H maps points from image2 -> image1
    H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, reproj_thresh)
    return H, mask


def align_images(image1, image2, method="sift", ratio=0.75, reproj_thresh=4.0):

    kp1, desc1 = detect_and_describe(image1, method)
    kp2, desc2 = detect_and_describe(image2, method)

    matches = match_keypoints(desc1, desc2, method, ratio)
    print(f"[INFO] {len(matches)} good matches found after ratio test.")

    H, mask = estimate_homography(kp1, kp2, matches, reproj_thresh)
    inliers = int(mask.sum())
    print(f"[INFO] {inliers}/{len(matches)} matches kept as RANSAC inliers.")
    print("[INFO] Estimated Homography matrix:\n", H)

    # Warp image2 onto image1's plane (canvas sized to fit both images)
    h1, w1 = image1.shape[:2]
    h2, w2 = image2.shape[:2]
    canvas_w, canvas_h = w1 + w2, max(h1, h2)
    aligned_image2 = cv2.warpPerspective(image2, H, (canvas_w, canvas_h))

    # Match visualization (inliers only)
    matches_mask = mask.ravel().tolist()
    match_vis = cv2.drawMatches(
        image1, kp1, image2, kp2, matches, None,
        matchColor=(0, 255, 0),
        singlePointColor=None,
        matchesMask=matches_mask,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    return aligned_image2, H, match_vis


if __name__ == "__main__":
    img1 = cv2.imread("graf1.png")
    img2 = cv2.imread("graf3.png")

    if img1 is None or img2 is None:
        raise FileNotFoundError("Could not load images/graf_pair/graf1.png or graf3.png")

    aligned, H, match_vis = align_images(img1, img2, method="sift")

    cv2.imwrite("output_task5_matches.jpg", match_vis)
    cv2.imwrite("output_task5_aligned.jpg", aligned)
    print("[DONE] Saved output_task5_matches.jpg and output_task5_aligned.jpg")