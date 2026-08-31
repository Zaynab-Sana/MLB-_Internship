# Smart Waste Classification Using CNN

A beginner-friendly, from-scratch Convolutional Neural Network that classifies waste
images into 6 categories, built with PyTorch. No pretrained models, no transfer
learning, no object detection — every layer is written and trained from zero on
the TrashNet dataset.

> **Status note:** This repository contains complete, working code for every stage
> of the pipeline. It has **not been executed end-to-end** in the environment that
> produced it (no GPU/dataset/network access there), so the results tables below
> are templates to fill in with your own numbers after you run training on your
> machine — they are marked clearly as such. Everything else (code, architecture,
> explanations) is complete and ready to run.

---

## 1. Project Overview

This project builds an image classifier that looks at a photo of a piece of waste
and predicts which of 6 categories it belongs to, along with a confidence score:

```
Input Image: waste.jpg
Predicted Class: plastic
Confidence: 94.70%
```

The goal is educational as much as functional: every stage (data loading, the CNN
itself, training, augmentation, optimizer comparison, evaluation, and inference) is
implemented in small, readable, heavily-commented files so a beginner/intermediate
Python and ML student can follow exactly what is happening and why.

## 2. Features

- Loads and explores the TrashNet dataset (class counts, balance check, sample images)
- Clean preprocessing pipeline (resize, tensor conversion, normalization)
- Reproducible 70% / 15% / 15% train/validation/test split with no data leakage
- A **Simple CNN built entirely from scratch** (no pretrained weights of any kind)
- Full training loop with live progress, validation tracking, and best-model saving
- Data augmentation on the training set only (flip, rotation, color jitter)
- Side-by-side **SGD vs Adam** optimizer comparison
- Small, controlled hyperparameter experiments with a results table
- Training/validation loss and accuracy graphs saved automatically
- Full test-set evaluation: accuracy, precision, recall, F1 (macro + weighted), per-class report
- Confusion matrix visualization
- A `predict.py` script for classifying any new image with a confidence score
- Automatic GPU/CPU selection, fixed random seed, portable paths, and basic error handling throughout

## 3. Dataset

This project uses the publicly available **TrashNet** dataset, created by Gary
Thung and Mindy Yang at Stanford. It contains ~2,500 images of household waste
sorted into 6 folders (one per class).

Where to get it:

- **Original source (GitHub):** `https://github.com/garythung/trashnet` — images
  live inside `trashnet/data/dataset-resized/`.
- **Kaggle mirror (easier browsing/download):** search "trashnet" on Kaggle, e.g.
  `https://www.kaggle.com/datasets/feyzazkefe/trashnet` (mirrors can change slugs
  over time — search if this exact link is unavailable).

TrashNet's raw folder names are singular/plural variants depending on the mirror
(e.g. `cardboard`, `glass`, `metal`, `paper`, `plastic`, `trash`). Whatever mirror
you use, arrange the images into this exact layout inside `dataset/`:

```
dataset/
├── cardboard/
│   ├── cardboard1.jpg
│   └── ...
├── glass/
├── metal/
├── paper/
├── plastic/
└── trash/
```

No CSV or annotation file is needed — the folder name **is** the label. This is
the standard layout PyTorch's `torchvision.datasets.ImageFolder` expects, which is
what `src/dataset.py` uses under the hood.

## 4. Classes

| # | Class      |
|---|------------|
| 1 | Plastic    |
| 2 | Paper      |
| 3 | Cardboard  |
| 4 | Glass      |
| 5 | Metal      |
| 6 | Trash      |

Note: `ImageFolder` assigns numeric labels in **alphabetical** order of the folder
names (`cardboard, glass, metal, paper, plastic, trash`), not the order listed in
the project brief. The code always saves the exact class-name list inside the
model checkpoint, so `predict.py` and `evaluate.py` never need to guess the order.

## 5. CNN Architecture

A simple 3-block convolutional network, built with plain `nn.Module` / `nn.Conv2d`
/ `nn.Linear` layers — nothing pretrained, nothing borrowed from a known backbone:

```
Input (3 x 128 x 128)
 → Conv2D(3→32, 3x3) → ReLU → MaxPool(2)     → 32 x 64 x 64
 → Conv2D(32→64, 3x3) → ReLU → MaxPool(2)    → 64 x 32 x 32
 → Conv2D(64→128, 3x3) → ReLU → MaxPool(2)   → 128 x 16 x 16
 → Flatten                                    → 32,768
 → Linear(32768 → 256) → ReLU → Dropout(0.5)
 → Linear(256 → 6)                            → 6 class scores (logits)
```

Full explanations of every concept (convolution, kernel, feature map, ReLU,
pooling, flattening, fully connected layers, softmax, and why CNNs suit image
data) are written as comments at the top of `src/model.py`.

## 6. Technologies Used

- Python 3.9+
- PyTorch & torchvision (model, training, data loading, augmentation)
- NumPy
- Matplotlib (training curves, sample images, confusion matrix)
- scikit-learn (precision/recall/F1, confusion matrix, classification report)
- Pillow (PIL) for image loading

## 7. Installation

```bash
git clone <this-repo-url>
cd smart-waste-classification
python3 -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The code automatically detects and uses a CUDA GPU if available (`src/utils.py:get_device`);
otherwise it falls back to CPU with no changes needed.

## 8. Dataset Setup

1. Download TrashNet from one of the sources in [Section 3](#3-dataset).
2. Arrange it into the `dataset/` folder using the class-per-folder layout shown above.
3. (Optional but recommended) Run the exploration script to confirm everything loaded correctly:

```bash
python -m src.dataset
```

Expected output: a printed count of images per class, a note on whether the
dataset is balanced or imbalanced, and two saved images —
`outputs/graphs/class_distribution.png` and `outputs/graphs/sample_images.png`.

**Common errors:**
- `FileNotFoundError: Dataset folder not found` → the `dataset/` folder is
  missing or empty; re-check step 2.
- Very few images detected → you may have pointed the script at a subfolder
  (e.g. `dataset/dataset-resized/`) instead of the folder that directly contains
  the 6 class folders. Flatten the structure so `dataset/` directly contains
  `cardboard/`, `glass/`, etc.

## 9. How to Train

Basic run (defaults to Adam, 15 epochs, batch size 32):

```bash
python -m src.train
```

Custom run, e.g. with SGD:

```bash
python -m src.train --optimizer sgd --lr 0.01 --momentum 0.9 --epochs 15
```

Useful flags:

| Flag | Meaning | Default |
|---|---|---|
| `--optimizer` | `sgd` or `adam` | `adam` |
| `--lr` | learning rate | 0.01 (sgd) / 0.001 (adam) |
| `--epochs` | number of training epochs | 15 |
| `--batch_size` | images per batch | 32 |
| `--num_filters` | filters in first conv layer | 32 |
| `--dropout` | dropout probability | 0.5 |
| `--run_name` | label used for saved files | optimizer name |

**Expected output:** per-epoch training/validation loss and accuracy printed to
the console, a saved checkpoint at `models/best_model_<run_name>.pth` (only
updated when validation accuracy improves), a training curve image at
`outputs/graphs/curves_<run_name>.png`, and a raw history JSON alongside it.

**Common errors:**
- `RuntimeError: CUDA out of memory` → lower `--batch_size` (try 16).
- Training loss stuck / not decreasing → try a lower `--lr`, or double-check the
  dataset actually loaded correctly (Section 8).
- Multiprocessing errors on Windows → add `--num_workers 0`.

### SGD vs Adam comparison and hyperparameter experiments (Day 3)

```bash
python -m src.experiments --mode optimizer_comparison --epochs 8
python -m src.experiments --mode hyperparameters --epochs 6
```

These reuse `train.py`/`evaluate.py` internally, run several short training
sessions, and print comparison tables like the ones in [Section 14](#14-results).

## 10. How to Evaluate

```bash
python -m src.evaluate --model models/best_model_adam.pth
```

**Expected output:** overall test accuracy, macro- and weighted-averaged
precision/recall/F1, a full per-class `classification_report`, and a note
explaining how to read the results.

**Common errors:**
- `FileNotFoundError: No checkpoint found` → run `src.train` first, or point
  `--model` at the correct `.pth` path.

## 11. How to Generate the Confusion Matrix

The confusion matrix is generated automatically as part of `src.evaluate` (no
separate command needed) and saved to:

```
outputs/confusion_matrix/confusion_matrix.png
```

Rows = true class, columns = predicted class. Diagonal cells are correct
predictions; off-diagonal cells show which classes get confused with each other
(commonly glass↔plastic and paper↔cardboard in TrashNet, since they can look
visually similar).

## 12. How to Predict a New Image

```bash
python -m src.predict --image waste.jpg
```

Optional flags: `--model <path>` to choose a checkpoint, `--no_show` to skip
popping up a matplotlib window (useful on a headless server), `--save_dir` to
change where the annotated output image is saved (default: `outputs/predictions/`).

**Common errors:**
- `Image not found` → check the path is correct and relative to where you're
  running the command from.
- `This checkpoint doesn't contain class names` → the `.pth` file wasn't
  produced by this project's `train.py`; retrain, or use a checkpoint that was.

## 13. Example Prediction

```
$ python -m src.predict --image waste.jpg
Input Image: waste.jpg
Predicted Class: plastic
Confidence: 94.70%

All class probabilities:
  plastic     : 94.70%
  glass       :  3.10%
  metal       :  1.05%
  cardboard   :  0.65%
  paper       :  0.30%
  trash       :  0.20%

Annotated prediction image saved to: outputs/predictions/prediction_waste.jpg.png
```

(The numbers above are illustrative formatting only — your actual model will
produce its own values once trained.)

## 14. Results

Fill in these tables with your own numbers after running `src.train` and
`src.evaluate` — they're intentionally left as templates since results depend on
your exact dataset copy, hardware, and number of epochs.

**Overall test performance:**

| Metric | Value |
|---|---|
| Test Accuracy | _fill in_ |
| Precision (macro) | _fill in_ |
| Recall (macro) | _fill in_ |
| F1-score (macro) | _fill in_ |

**Per-class performance:** (from `src.evaluate`'s printed classification report)

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| Plastic | | | |
| Paper | | | |
| Cardboard | | | |
| Glass | | | |
| Metal | | | |
| Trash | | | |

**SGD vs Adam** (from `python -m src.experiments --mode optimizer_comparison`):

| Metric | SGD | Adam |
|---|---|---|
| Final Train Accuracy | | |
| Final Validation Accuracy | | |
| Test Accuracy | | |
| Training Time (s) | | |

**Hyperparameter experiments** (from `python -m src.experiments --mode hyperparameters`):

| Experiment | LR | Batch Size | Filters | Dropout | Val Acc % | Test Acc % |
|---|---|---|---|---|---|---|
| baseline | 0.001 | 32 | 32 | 0.5 | | |
| higher_lr | 0.005 | 32 | 32 | 0.5 | | |
| lower_lr | 0.0005 | 32 | 32 | 0.5 | | |
| larger_batch | 0.001 | 64 | 32 | 0.5 | | |
| more_filters | 0.001 | 32 | 64 | 0.5 | | |
| higher_dropout | 0.001 | 32 | 32 | 0.7 | | |

## 15. Limitations

- TrashNet is a relatively small (~2,500 images), and mildly imbalanced dataset —
  results won't match large-scale, transfer-learning-based waste classifiers.
- Images were mostly captured under controlled, well-lit studio conditions;
  performance will likely be noticeably lower on messy, real-world photos
  (cluttered backgrounds, poor lighting, partially visible objects).
- The 6 categories don't capture every real-world recycling stream (e.g. no
  organic waste, no e-waste, no mixed/composite materials).
- Model confidence is not a guarantee of correctness — a confidently wrong
  prediction is possible, especially on out-of-distribution images.
- No pretrained backbone was used (by design, per project requirements), so
  accuracy is naturally lower than what a transfer-learning approach (e.g.
  fine-tuning ResNet/MobileNet) would achieve on the same data.

## 16. Possible Future Improvements

- Collect or add more real-world, cluttered images to improve robustness.
- Try transfer learning (e.g. MobileNetV2) as a comparison baseline against this from-scratch CNN.
- Add batch normalization layers to potentially stabilize/speed up training.
- Use a learning-rate scheduler instead of a fixed learning rate.
- Add k-fold cross-validation for a more robust performance estimate on this small dataset.
- Package `predict.py` behind a small web UI (e.g. Streamlit/Flask) for interactive demos.
- Explore class-balancing techniques (weighted loss, oversampling) for the smaller classes like "trash."

---

## End-to-End Workflow (run in this order)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Explore the dataset (optional but recommended)
python -m src.dataset

# 3. Train the model
python -m src.train --optimizer adam --epochs 15

# 4. (Optional) Run the SGD vs Adam comparison and hyperparameter experiments
python -m src.experiments --mode optimizer_comparison --epochs 8
python -m src.experiments --mode hyperparameters --epochs 6

# 5. Evaluate the best model on the test set + generate the confusion matrix
python -m src.evaluate --model models/best_model_adam.pth

# 6. Predict on a brand-new image
python -m src.predict --image waste.jpg
```

## Project Structure

```
smart-waste-classification/
│
├── dataset/                    # Place TrashNet images here (see Section 3)
├── models/                     # Saved model checkpoints (.pth) land here
├── outputs/
│   ├── graphs/                 # Training curves, class distribution, sample images
│   ├── confusion_matrix/       # Confusion matrix image
│   └── predictions/            # Annotated single-image prediction outputs
│
├── src/
│   ├── dataset.py              # Loading, transforms, split, exploration
│   ├── model.py                # SimpleCNN architecture
│   ├── train.py                # Training loop, best-model saving
│   ├── evaluate.py             # Test-set metrics + confusion matrix
│   ├── predict.py              # Single-image inference
│   ├── experiments.py          # SGD vs Adam, hyperparameter sweep
│   └── utils.py                # Seed, device, checkpoint, plotting helpers
│
├── requirements.txt
└── README.md
```
