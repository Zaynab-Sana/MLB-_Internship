"""
dataset.py
----------
Everything related to loading the TrashNet dataset:

1. Expected folder layout
2. Image preprocessing (resize, tensor conversion, normalization)
3. Data augmentation (training set only)
4. Reproducible 70/15/15 train/validation/test split
5. PyTorch DataLoaders

--------------------------------------------------------------------
WHERE TO GET TRASHNET
--------------------------------------------------------------------
TrashNet was created by Gary Thung and Mindy Yang (Stanford). The most
common way to get it:

    Option A (GitHub, original):
        git clone https://github.com/garythung/trashnet
        # images are inside trashnet/data/dataset-resized/

    Option B (Kaggle mirror, easier to browse/download):
        https://www.kaggle.com/datasets/feyzazkefe/trashnet
        (search "trashnet" on Kaggle if this exact slug changes)

After downloading, arrange the images like this so this script can
find them (this is the standard "ImageFolder" layout PyTorch expects
-- one subfolder per class, class name = folder name):

    dataset/
    ├── cardboard/
    │   ├── cardboard1.jpg
    │   ├── cardboard2.jpg
    │   └── ...
    ├── glass/
    ├── metal/
    ├── paper/
    ├── plastic/
    └── trash/

That's it -- no CSV, no annotation file needed. The folder name IS the
label. This is exactly what torchvision.datasets.ImageFolder expects,
which is why we use it below instead of writing a custom parser.
--------------------------------------------------------------------
"""

import os
from collections import Counter

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from src.utils import CLASS_NAMES, project_root

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------
IMAGE_SIZE = 128          # Resize every image to 128x128 pixels.
# Standard ImageNet mean/std values. Even though we are NOT using a
# pretrained ImageNet model, these are well-tested "center the pixel
# values around 0, scale to roughly unit variance" numbers that work
# fine for training any CNN from scratch on natural images.
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]


def get_train_transforms():
    """
    Preprocessing + augmentation applied ONLY to training images.

    Why each step exists:
    - Resize(128, 128):       CNNs need a fixed input size because the
                               fully-connected layer at the end expects
                               a fixed number of input features.
    - RandomHorizontalFlip:   A plastic bottle photographed facing left
                               or right is still a plastic bottle. This
                               teaches the model that orientation doesn't
                               matter, using free extra "views" of the data.
    - RandomRotation(15):     Photos in the real world won't always be
                               perfectly upright; small rotations make
                               the model robust to camera angle.
    - ColorJitter:            Varies brightness/contrast slightly so the
                               model doesn't overfit to one lighting setup.
    - ToTensor():              Converts a PIL image (H x W x C, values
                               0-255) into a PyTorch tensor (C x H x W,
                               values 0-1). PyTorch layers expect tensors,
                               not raw images.
    - Normalize(mean, std):   Rescales each channel to roughly zero mean
                               and unit variance. This helps the network
                               train faster and more stably, because the
                               optimizer isn't dealing with wildly
                               different value ranges.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])


def get_eval_transforms():
    """
    Preprocessing applied to validation, test, and single-image prediction.

    IMPORTANT: No random augmentation here. Validation and test data must
    stay exactly the same every time we evaluate, otherwise our accuracy
    numbers would be noisy and not comparable between epochs or between
    experiments. Augmentation is a training-time trick only, it should
    never touch the data we use to judge how good the model really is.
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])


class TransformedSubset(torch.utils.data.Dataset):
    """
    A thin wrapper that lets us apply DIFFERENT transforms to the train
    split vs the val/test splits, even though they all come from the
    same underlying ImageFolder dataset.

    Why this is needed: torch.utils.data.random_split() gives back
    "Subset" objects that all share the same underlying dataset object
    (and therefore the same transform). But we want augmentation only
    on the training subset. This wrapper re-applies the correct
    transform to the raw (untransformed) image for each split.
    """

    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, label = self.subset[idx]
        # `image` here is a PIL Image because the base ImageFolder
        # dataset was created with transform=None (see below).
        if self.transform:
            image = self.transform(image)
        return image, label


def load_datasets(dataset_dir: str = None, seed: int = 42):
    """
    Load TrashNet from `dataset_dir` and split it into train/val/test
    subsets (70% / 15% / 15%), with a fixed random seed so the split
    is identical every time this function runs.

    Returns three Dataset objects: train_dataset, val_dataset, test_dataset
    """
    if dataset_dir is None:
        dataset_dir = os.path.join(project_root(), "dataset")

    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(
            f"Dataset folder not found at '{dataset_dir}'.\n"
            f"Download TrashNet and place it there using this layout:\n"
            f"  dataset/cardboard/*.jpg\n"
            f"  dataset/glass/*.jpg\n"
            f"  dataset/metal/*.jpg\n"
            f"  dataset/paper/*.jpg\n"
            f"  dataset/plastic/*.jpg\n"
            f"  dataset/trash/*.jpg\n"
            f"See the top of src/dataset.py for download links."
        )

    # transform=None here on purpose: we apply the real transforms later,
    # per-split, via TransformedSubset. This is what avoids leaking
    # augmentation into the validation/test data.
    full_dataset = datasets.ImageFolder(root=dataset_dir, transform=None)

    detected_classes = full_dataset.classes
    if detected_classes != CLASS_NAMES:
        print(
            f"NOTE: folder class order detected as {detected_classes}, "
            f"which differs from the expected {CLASS_NAMES}. "
            f"This is fine as long as you always use this same "
            f"dataset folder — ImageFolder's class order is just "
            f"alphabetical and is saved inside the checkpoint."
        )

    total_size = len(full_dataset)
    train_size = int(0.70 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size  # remainder avoids rounding loss

    # A fixed generator seed guarantees the SAME split every run.
    # Because the split happens on indices, and each index is used in
    # exactly one of train/val/test, there is no overlap (no data leakage)
    # between the three sets.
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset, test_subset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    train_dataset = TransformedSubset(train_subset, get_train_transforms())
    val_dataset = TransformedSubset(val_subset, get_eval_transforms())
    test_dataset = TransformedSubset(test_subset, get_eval_transforms())

    print(f"Total images found: {total_size}")
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    return train_dataset, val_dataset, test_dataset, full_dataset.classes


def get_dataloaders(dataset_dir: str = None, batch_size: int = 32,
                     num_workers: int = 2, seed: int = 42):
    """
    Convenience function: load datasets AND wrap them in DataLoaders.

    - batch_size: how many images are processed together in one forward/
      backward pass. Larger batches train faster (more parallelism) but
      use more memory and can generalize slightly differently. 32 is a
      safe default for a 128x128 image CNN on a normal laptop.
    - shuffle=True for training: reshuffling the order every epoch stops
      the model from memorizing the order of images and improves
      generalization. Validation/test use shuffle=False because order
      doesn't matter for evaluation and keeping it fixed makes debugging
      easier (e.g. confusion matrix rows line up with the same images
      each run).
    - num_workers: number of background CPU processes that load and
      preprocess images in parallel while the GPU/CPU trains on the
      previous batch. 2 is a safe default; set to 0 if you hit
      multiprocessing errors on Windows.
    """
    train_dataset, val_dataset, test_dataset, classes = load_datasets(dataset_dir, seed)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )

    return train_loader, val_loader, test_loader, classes


def explore_dataset(dataset_dir: str = None, save_dir: str = None):
    """
    Simple exploratory analysis, meant to be run once before training:
    - counts images per class
    - reports whether the dataset is balanced or imbalanced
    - saves a bar chart of class distribution
    - saves a grid of sample images, one per class

    This corresponds to "Day 1, Step 1: Dataset Download and Exploration"
    in the project plan.
    """
    if dataset_dir is None:
        dataset_dir = os.path.join(project_root(), "dataset")
    if save_dir is None:
        save_dir = os.path.join(project_root(), "outputs", "graphs")
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(
            f"Dataset folder not found at '{dataset_dir}'. "
            f"See the docstring at the top of dataset.py for setup instructions."
        )

    dataset = datasets.ImageFolder(root=dataset_dir, transform=None)
    labels = [label for _, label in dataset.samples]
    counts = Counter(labels)
    class_names = dataset.classes

    print("Class distribution:")
    for idx, name in enumerate(class_names):
        print(f"  {name:12s}: {counts.get(idx, 0)} images")

    values = [counts.get(i, 0) for i in range(len(class_names))]
    max_count, min_count = max(values), min(values)
    # A common rule of thumb: if the largest class has more than ~1.5x
    # the images of the smallest class, treat the dataset as imbalanced.
    imbalance_ratio = max_count / max(min_count, 1)
    if imbalance_ratio > 1.5:
        print(f"Dataset appears IMBALANCED (ratio {imbalance_ratio:.2f}x between "
              f"largest and smallest class). Consider this when reading per-class "
              f"metrics later -- a class with few images is naturally harder to learn.")
    else:
        print(f"Dataset appears roughly BALANCED (ratio {imbalance_ratio:.2f}x).")

    # --- Bar chart of class distribution ---
    plt.figure(figsize=(8, 5))
    plt.bar(class_names, values, color="teal")
    plt.title("TrashNet: Number of Images per Class")
    plt.xlabel("Class")
    plt.ylabel("Number of Images")
    plt.tight_layout()
    dist_path = os.path.join(save_dir, "class_distribution.png")
    plt.savefig(dist_path, dpi=150)
    plt.close()
    print(f"Class distribution chart saved to: {dist_path}")

    # --- Sample image grid, one image per class ---
    fig, axes = plt.subplots(1, len(class_names), figsize=(3 * len(class_names), 3))
    for i, class_name in enumerate(class_names):
        # Find the first sample belonging to this class.
        for path, label in dataset.samples:
            if label == i:
                from PIL import Image
                img = Image.open(path).convert("RGB")
                axes[i].imshow(img)
                axes[i].set_title(class_name)
                axes[i].axis("off")
                break
    plt.tight_layout()
    samples_path = os.path.join(save_dir, "sample_images.png")
    plt.savefig(samples_path, dpi=150)
    plt.close()
    print(f"Sample images grid saved to: {samples_path}")


if __name__ == "__main__":
    # Running `python src/dataset.py` directly performs the Day-1
    # exploration step: prints class counts and saves example figures.
    explore_dataset()
