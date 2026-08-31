# Technical Report — Smart Waste Classification Using CNN

## 1. Objective

Build a from-scratch Convolutional Neural Network in PyTorch that classifies
images of household waste into 6 categories (plastic, paper, cardboard, glass,
metal, trash), without using any pretrained models, transfer learning, or object
detection frameworks.

## 2. Dataset

**TrashNet** (Thung & Yang, Stanford), ~2,500 images across 6 folders. Images
were resized to 128×128, converted to RGB tensors, and normalized using standard
ImageNet-style mean/std values (chosen for their well-tested numerical properties,
not because any ImageNet-pretrained weights were used).

The dataset was split 70% / 15% / 15% into train/validation/test using
`torch.utils.data.random_split` with a fixed random seed (42). Splitting on
indices from a single underlying dataset guarantees each image appears in exactly
one split, eliminating data leakage between train, validation, and test.

## 3. Preprocessing and Augmentation

- **All splits:** resize to 128×128 → tensor conversion → normalization.
- **Training split only:** random horizontal flip, random rotation (±15°), and
  mild color jitter (brightness/contrast/saturation), to expose the model to
  more visual variation without changing the underlying label.
- **Validation/test splits:** deliberately left free of any random augmentation,
  so accuracy numbers are stable and comparable across epochs and experiments.

## 4. Model Architecture

A 3-block CNN, built entirely from `nn.Conv2d`, `nn.ReLU`, `nn.MaxPool2d`, and
`nn.Linear` layers:

```
Input (3×128×128)
→ [Conv(3→32,3×3) → ReLU → MaxPool(2)]  → 32×64×64
→ [Conv(32→64,3×3) → ReLU → MaxPool(2)] → 64×32×32
→ [Conv(64→128,3×3) → ReLU → MaxPool(2)]→ 128×16×16
→ Flatten → 32,768
→ Linear(32768→256) → ReLU → Dropout(0.5)
→ Linear(256→6) → logits
```

Design rationale: three conv blocks are enough to move from low-level features
(edges, colors) to higher-level shape/texture patterns, while keeping the
parameter count and compute requirements small enough to train on a normal
laptop CPU in a reasonable time. Dropout (p=0.5) before the final layer reduces
overfitting given the dataset's modest size (~2,500 images).

## 5. Training Setup

- **Loss:** `nn.CrossEntropyLoss`, the standard choice for multi-class,
  single-label classification; it combines LogSoftmax and negative
  log-likelihood internally, so the model's `forward()` returns raw logits
  rather than probabilities.
- **Optimizers compared:** SGD (lr=0.01, momentum=0.9) and Adam (lr=0.001), with
  all other settings held constant, to isolate the effect of the optimizer
  choice (see `src/experiments.py::compare_optimizers`).
- **Model selection:** the checkpoint is overwritten only when validation
  accuracy improves, so the saved model reflects the point of best
  generalization rather than the final epoch, which guards against
  overfitting in later epochs.
- **Reproducibility:** a fixed random seed controls dataset shuffling, weight
  initialization, and cuDNN determinism.

## 6. Evaluation Methodology

The final saved model is evaluated once on the untouched test split (15% of the
data, never used for training or model selection). Reported metrics:

- **Accuracy** — overall fraction of correct predictions.
- **Precision, Recall, F1 (macro-averaged)** — the headline multiclass metric,
  since it weighs all 6 classes equally regardless of how many images each has.
  This matters because TrashNet's classes are not perfectly balanced (e.g.
  "trash" typically has noticeably fewer images than "glass" or "paper"), and a
  macro average prevents strong performance on common classes from masking poor
  performance on rarer ones.
- **Weighted-averaged Precision/Recall/F1** — reported alongside macro as a
  reference for how the model performs proportionally to class frequency.
- **Per-class Precision/Recall/F1** — via scikit-learn's `classification_report`.
- **Confusion matrix** — visualizes which classes are most often confused with
  each other (commonly glass↔plastic and paper↔cardboard, since these pairs can
  share visual traits like transparency or fibrous texture).

## 7. Hyperparameter Experiments

A small, controlled sweep (not an exhaustive search) varies one setting at a
time from a fixed baseline (lr=0.001, batch_size=32, num_filters=32,
dropout=0.5, Adam): higher/lower learning rate, larger batch size, more
filters, and higher dropout. Each variant is trained for the same number of
epochs and compared on validation and test accuracy (see
`src/experiments.py::run_hyperparameter_experiments`).

## 8. How to Reproduce

See the "End-to-End Workflow" section of `README.md` for the exact command
sequence, from dataset setup through training, optimizer comparison,
hyperparameter experiments, evaluation, and single-image prediction.

## 9. Limitations and Caveats

This report describes a complete, runnable pipeline. The result tables in
`README.md` are left as templates because this document was produced in an
environment without GPU/network access to actually download TrashNet and run
full training — all code has been syntax-checked and the CNN's dimension math
has been manually verified, but end-to-end numeric results should be generated
by running the pipeline in the reader's own environment. See `README.md`
Section 15 for further limitations of the approach itself (dataset size,
imbalance, and the absence of a pretrained backbone by design).
