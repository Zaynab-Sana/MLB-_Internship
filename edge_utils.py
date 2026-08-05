

import numpy as np


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------
def to_uint8(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0, 255).astype(np.uint8)


def manual_convolve2d(img2d: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Manual 2D convolution (zero padding) for a single-channel image,
    implemented as explicit shifted-array multiply-adds - kh*kw taps,
    each a full-image vectorized operation, so even a 9x9 kernel stays
    fast regardless of image size.
    """
    img_f = img2d.astype(np.float64)
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(img_f, ((ph, ph), (pw, pw)), mode="constant")
    h, w = img2d.shape
    out = np.zeros((h, w), dtype=np.float64)
    for ki in range(kh):
        for kj in range(kw):
            weight = float(kernel[ki, kj])
            if weight == 0:
                continue
            out += weight * padded[ki : ki + h, kj : kj + w]
    return out


# ---------------------------------------------------------------------------
# Gaussian kernel (parameterized by sigma) - used by LoG, DoG, Canny
# ---------------------------------------------------------------------------
def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """
    Builds a size x size Gaussian kernel from the 2D Gaussian formula
    G(x,y) = exp(-(x^2+y^2) / (2*sigma^2)), normalized so it sums to 1.
    `size` must be odd.
    """
    ax = np.arange(-(size // 2), size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel


def gaussian_blur_manual(img2d: np.ndarray, size: int, sigma: float) -> np.ndarray:
    kernel = gaussian_kernel(size, sigma)
    return manual_convolve2d(img2d, kernel)


# ---------------------------------------------------------------------------
# Sobel gradient
# ---------------------------------------------------------------------------
SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)


def sobel_gradients(img2d: np.ndarray):
    """Returns (Gx, Gy, magnitude, direction_radians)."""
    gx = manual_convolve2d(img2d, SOBEL_X)
    gy = manual_convolve2d(img2d, SOBEL_Y)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    direction = np.arctan2(gy, gx)  # radians, range [-pi, pi]
    return gx, gy, magnitude, direction


# ---------------------------------------------------------------------------
# Laplacian
# ---------------------------------------------------------------------------
LAPLACIAN_KERNEL = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)


def laplacian_manual(img2d: np.ndarray) -> np.ndarray:
    return manual_convolve2d(img2d, LAPLACIAN_KERNEL)


# ---------------------------------------------------------------------------
# Gaussian noise (manual, NumPy only) - for testing LoG on noisy images
# ---------------------------------------------------------------------------
def add_gaussian_noise(img2d: np.ndarray, noise_percentage: float) -> np.ndarray:
    noise_level = (noise_percentage / 100.0) * 255.0
    noise = np.random.normal(0, noise_level, img2d.shape)
    return to_uint8(img2d.astype(np.float64) + noise)


# ---------------------------------------------------------------------------
# Canny pipeline stages
# ---------------------------------------------------------------------------
def non_max_suppression(magnitude: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """
    Thins edges by keeping only local maxima along the gradient direction.
    Direction is quantized into 4 sectors: 0, 45, 90, 135 degrees.
    """
    h, w = magnitude.shape
    out = np.zeros((h, w), dtype=np.float64)
    angle = np.degrees(direction) % 180  # fold negative angles into [0, 180)

    for i in range(1, h - 1):
        for j in range(1, w - 1):
            a = angle[i, j]
            m = magnitude[i, j]

            if (0 <= a < 22.5) or (157.5 <= a <= 180):
                n1, n2 = magnitude[i, j - 1], magnitude[i, j + 1]          # horizontal
            elif 22.5 <= a < 67.5:
                n1, n2 = magnitude[i - 1, j + 1], magnitude[i + 1, j - 1]  # diagonal /
            elif 67.5 <= a < 112.5:
                n1, n2 = magnitude[i - 1, j], magnitude[i + 1, j]          # vertical
            else:  # 112.5 <= a < 157.5
                n1, n2 = magnitude[i - 1, j - 1], magnitude[i + 1, j + 1]  # diagonal \

            out[i, j] = m if (m >= n1 and m >= n2) else 0
    return out


def double_threshold(img2d: np.ndarray, low_ratio: float = 0.05, high_ratio: float = 0.15):
    """
    Classifies pixels into strong edges, weak edges, and non-edges, using
    two thresholds relative to the image's own maximum gradient value.
    Returns (result, strong_value, weak_value): `result` contains
    `strong_value` for strong edges, `weak_value` for weak edges, 0 elsewhere.
    """
    max_val = img2d.max()
    high_thresh = max_val * high_ratio
    low_thresh = max_val * low_ratio

    strong_value = 255
    weak_value = 75

    result = np.zeros_like(img2d, dtype=np.uint8)
    strong_i, strong_j = np.where(img2d >= high_thresh)
    weak_i, weak_j = np.where((img2d >= low_thresh) & (img2d < high_thresh))

    result[strong_i, strong_j] = strong_value
    result[weak_i, weak_j] = weak_value
    return result, strong_value, weak_value


def hysteresis(img2d: np.ndarray, weak_value: int = 75, strong_value: int = 255) -> np.ndarray:
    """
    Manual edge tracking by hysteresis: any weak-edge pixel connected
    (8-neighborhood), directly or via a chain of other weak pixels, to a
    strong-edge pixel is promoted to a strong edge. Implemented as an
    iterative stack-based flood fill - no scipy.ndimage.label, fully manual.
    """
    h, w = img2d.shape
    out = img2d.copy()
    visited = np.zeros((h, w), dtype=bool)

    stack = list(zip(*np.where(out == strong_value)))

    while stack:
        i, j = stack.pop()
        if visited[i, j]:
            continue
        visited[i, j] = True

        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                ni, nj = i + di, j + dj
                if 0 <= ni < h and 0 <= nj < w and out[ni, nj] == weak_value:
                    out[ni, nj] = strong_value
                    stack.append((ni, nj))

    out[out == weak_value] = 0  # weak pixels never connected to a strong edge are dropped
    return out


def canny_manual(img2d: np.ndarray, sigma: float = 1.4, kernel_size: int = 5,
                  low_ratio: float = 0.05, high_ratio: float = 0.15) -> np.ndarray:
    """Full manual Canny pipeline: blur -> gradient -> NMS -> double threshold -> hysteresis."""
    blurred = gaussian_blur_manual(img2d, kernel_size, sigma)
    _, _, magnitude, direction = sobel_gradients(blurred)
    suppressed = non_max_suppression(magnitude, direction)
    thresholded, strong_val, weak_val = double_threshold(suppressed, low_ratio, high_ratio)
    edges = hysteresis(thresholded, weak_val, strong_val)
    return edges