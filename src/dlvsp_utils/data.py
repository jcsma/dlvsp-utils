"""dlvsp_utils.data

Utilities for dataset inspection, class selection and imbalance profiles.

Author: Juan Carlos San Miguel <juancarlos.sanmiguel@uam.es>
Utility functions for the practical assignments of the
"Deep Learning for Visual Signal Processing" course (IPCVAI, UAM).

The helpers in this module are intended for interactive notebooks. They
provide small, composable building blocks for working with common
PyTorch datasets and their wrappers without having to re‑implement the
same patterns in every notebook.

Public functions
- ``_get_targets_any_dataset``: extract label arrays from torchvision
    datasets, ``torch.utils.data.Subset`` objects and simple wrappers that
    return ``(x, y)``.
- ``select_classes_dataset``: keep only a subset of class names and
    remap labels to a compact range ``[0..K-1]``.
- ``count_samples_dataset``: return sample counts per class as a plain
    Python list.
- ``define_imbalance_profile`` and ``exponential_decay_samples``: create
    simple long‑tailed or uniform‑minority class distributions in a
    reproducible way.
- ``apply_imbalance``: build an imbalanced ``Subset`` according to a
    target profile.
- ``inspect_dataset_classes``: quick text summary of class presence.

All utilities rely only on ``numpy`` and ``torch`` (with ``matplotlib``
used lazily inside plotting helpers) so they remain lightweight and
easy to reuse across different exercises.
"""
from typing import List, Tuple, Optional
import numpy as np
import torch
from torch.utils.data import Subset, Dataset

# ============================================================
# Utility: extract labels from "any" dataset
# ============================================================
def _get_targets_any_dataset(dataset):
    """
    Return labels for a dataset as a NumPy array.

    Why this function exists:
    - Many torchvision datasets (e.g., CIFAR-10) expose labels via dataset.targets.
    - But after we apply wrappers (e.g., RemapLabels) or create torch.utils.data.Subset,
      the resulting object may NOT have a .targets attribute.
    - This function unifies label extraction so downstream code can be generic.

    Supported cases:
    1) torchvision datasets (have .targets)
    2) torch.utils.data.Subset (has .dataset + .indices)
    3) wrapper datasets (fallback: read labels via __getitem__)
    """

    # -----------------------
    # Case 1: dataset is a Subset
    # -----------------------
    # A Subset stores:
    #   - dataset.dataset  -> the underlying original dataset
    #   - dataset.indices  -> the subset indices in the original dataset
    if isinstance(dataset, Subset):
        base = dataset.dataset        # underlying dataset
        idxs = dataset.indices        # indices selected in the subset

        # If the base dataset has .targets, we can extract labels fast
        if hasattr(base, "targets"):
            base_targets = np.asarray(base.targets)
            return base_targets[idxs]

        # Fallback: slow path (calls __getitem__ for each index)
        return np.array([base[i][1] for i in idxs], dtype=int)

    # -----------------------
    # Case 2: standard dataset with .targets
    # -----------------------
    if hasattr(dataset, "targets"):
        return np.asarray(dataset.targets)

    # -----------------------
    # Case 3: generic dataset fallback
    # -----------------------
    # Works for wrappers like RemapLabels: dataset[i] returns (x, y).
    return np.array([dataset[i][1] for i in range(len(dataset))], dtype=int)


# ============================================================
# Utility: create an imbalanced training subset
# ============================================================
def apply_imbalance(full_dataset, samples_per_class, seed=0):
    """
    Create an imbalanced Subset of `full_dataset` by keeping a specified number
    of samples per class.

    Parameters
    ----------
    full_dataset : torch.utils.data.Dataset
        Any PyTorch dataset that returns (x, y) and whose labels are integers.
        It can be:
        - CIFAR-10
        - a Subset
        - a wrapper dataset (e.g. RemapLabels)
    samples_per_class : list[int]
        Desired number of samples to keep for each class.
        Length defines num_classes = len(samples_per_class).
        Example: for 4 classes, samples_per_class could be [2000, 2000, 200, 200].
    seed : int
        Random seed used to select which samples are kept (reproducible).

    Returns
    -------
    torch.utils.data.Subset
        A Subset of `full_dataset` that follows the requested class distribution.
    """

    # Create a reproducible random number generator
    rng = np.random.default_rng(seed)

    # Extract labels from the dataset in a robust way
    targets = _get_targets_any_dataset(full_dataset)

    # Number of classes is determined by the length of samples_per_class
    num_classes = len(samples_per_class)

    selected_indices = []  # indices we will keep

    # For each class, choose the requested number of indices
    for c in range(num_classes):
        # Find indices in the dataset belonging to class c
        class_indices = np.where(targets == c)[0]

        # If a class is missing (possible in extreme filtering), skip it
        if len(class_indices) == 0:
            continue

        # Keep at most the number of available samples
        n_keep = int(min(samples_per_class[c], len(class_indices)))

        # Randomly pick which samples to keep, without replacement
        chosen = rng.choice(class_indices, size=n_keep, replace=False)
        selected_indices.extend(chosen.tolist())

    # Shuffle final subset indices so the Subset is not ordered by class
    rng.shuffle(selected_indices)

    # Return a Subset that will index into the original dataset
    return Subset(full_dataset, selected_indices)


# ============================================================
# Utility: select a subset of class names and remap labels
# ============================================================
def select_classes_dataset(source_dataset, selected_class_names):
    """
    Filter a dataset to keep only specified classes AND remap labels to [0, K-1].

    Why we remap labels:
    - If you select CIFAR-10 classes like ['cat', 'dog'], their original labels are [3, 5].
    - Most training code (and nn.CrossEntropyLoss) expects labels in [0..num_classes-1].
    - We remap:
        original label 3 -> new label 0
        original label 5 -> new label 1

    Parameters
    ----------
    source_dataset : torchvision dataset (e.g., CIFAR-10)
        Must have:
        - source_dataset.classes : list of class names
        - source_dataset.targets : list of original labels
    selected_class_names : list[str]
        Class names to keep (order defines the new label mapping).

    Returns
    -------
    selected_dataset : torch.utils.data.Dataset
        Dataset that contains only the selected classes, with remapped labels.
    class_names : list[str]
        The same list as selected_class_names (returned for convenience).
    """

    # Map original class name -> original class index
    all_class_names = source_dataset.classes
    class_to_idx = {cls: i for i, cls in enumerate(all_class_names)}

    # Convert requested class names into original numeric labels
    selected_class_indices = [class_to_idx[c] for c in selected_class_names]

    # Create a mask of which samples belong to the selected classes
    targets = torch.as_tensor(source_dataset.targets)
    mask = torch.isin(targets, torch.tensor(selected_class_indices))
    indices = mask.nonzero(as_tuple=True)[0].tolist()

    # Subset: keep only those samples
    subset = Subset(source_dataset, indices)

    # Build original_label -> new_label mapping (contiguous)
    # Example: selected_class_indices = [3, 5] -> mapping {3:0, 5:1}
    old_to_new = {old: new for new, old in enumerate(selected_class_indices)}

    # Wrapper dataset that applies the label remapping on the fly
    class RemapLabels(Dataset):
        def __init__(self, subset, mapping):
            self.subset = subset      # the Subset with selected samples
            self.mapping = mapping    # old->new label mapping

        def __len__(self):
            return len(self.subset)

        def __getitem__(self, idx):
            x, y = self.subset[idx]        # y is original label (e.g., 3 or 5)
            return x, self.mapping[y]      # return remapped label (e.g., 0 or 1)

    selected_dataset = RemapLabels(subset, old_to_new)
    return selected_dataset, selected_class_names


# ============================================================
# Utility: count samples per class (works for datasets + wrappers)
# ============================================================
def count_samples_dataset(dataset, num_classes: int = None):
    """
    Count the number of samples per class.

    Works for:
    - torchvision datasets with .targets
    - Subset objects
    - wrapper datasets (fallback via __getitem__)

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Dataset to analyze.
    num_classes : int, optional
        If provided, ensures the count vector has this length.
        If None, it is inferred from max(label)+1.

    Returns
    -------
    list[int]
        counts[c] = number of samples labeled as class c.
    """

    # Extract targets in a robust way
    targets = torch.as_tensor(_get_targets_any_dataset(dataset))

    # Infer number of classes if not given
    if num_classes is None:
        num_classes = int(targets.max().item()) + 1

    # bincount counts how many times each class index appears
    counts = torch.bincount(targets, minlength=num_classes)
    return counts.tolist()


# ============================================================
# Utility: exponential decay profile (for long-tailed imbalance)
# ============================================================
def exponential_decay_samples(start, end, num_classes, factor=None):
    """
    Create a long-tailed class distribution using exponential decay.

    start: number of samples in the head (majority) class
    end:   minimum number of samples in the tail (minority) classes
    num_classes: number of classes
    factor: optional decay rate; if None, computed so that class (num_classes-1) ~ end

    Returns
    -------
    list[int] length num_classes
    """

    # If factor is not provided, compute it so that the last class reaches ~end
    if factor is None:
        factor = -np.log(end / start) / (num_classes - 1)

    indices = np.arange(num_classes)
    samples = start * np.exp(-factor * indices)

    # Enforce a minimum of 'end' samples
    samples = np.maximum(samples, end).astype(int)

    return samples.tolist()


# ============================================================
# Utility: deterministic imbalance profile from (type, severity, student_id)
# ============================================================
def define_imbalance_profile(num_classes, max_count, student_id,
                            imbalance_type="long_tail", severity="medium"):
    """
    Define how many training samples to keep per class (samples_per_class)
    in a deterministic way based on student_id.

    This ensures:
    - Every student has a different imbalance (to prevent copying)
    - Your results are reproducible (same student_id -> same imbalance)

    Returns
    -------
    samples_per_class : list[int]
        Desired samples per class.
    minority_classes : list[int]
        A few minority class indices (for reporting).
    """

    # Map severity to a minimum ratio of samples in the smallest classes
    ratio_map = {"verylow": 0.5, "low": 0.2, "medium": 0.1, "high": 0.05, "veryhigh": 0.01,"ultrahigh": 0.001}
    if severity not in ratio_map:
        raise ValueError(f"severity must be one of {list(ratio_map.keys())}")
    min_ratio = ratio_map[severity]

    # Deterministic RNG seed from student_id
    try:
        seed = int(student_id) % (2**32 - 1)
    except Exception:
        seed = 0
    rng = np.random.default_rng(seed)

    if imbalance_type == "long_tail":
        # Exponential decay from max_count down to max_count * min_ratio
        end_value = max(1, int(max_count * min_ratio))
        samples = exponential_decay_samples(max_count, end_value, num_classes)

        # For reporting: identify a few smallest classes (tail)
        minority_classes = list(np.argsort(samples)[:min(3, num_classes)])
        return samples, minority_classes

    elif imbalance_type == "uniform_minority":
        # Choose a few minority classes deterministically
        n_minority = min(3, num_classes)
        minority_classes = rng.choice(np.arange(num_classes), size=n_minority, replace=False).tolist()

        # Assign low sample count to those minority classes
        min_count = max(1, int(max_count * min_ratio))
        samples = [max_count] * num_classes
        for c in minority_classes:
            samples[c] = min_count

        return samples, minority_classes

    else:
        raise ValueError('imbalance_type must be "long_tail" or "uniform_minority"')

# ============================================================
# Utility: inspect class presence & counts
# ============================================================
def inspect_dataset_classes(dataset, class_names=None, header=None):
    """
    Print which class indices are present in `dataset` and how many samples each has.

    If `header` is provided, it will be printed before the class listing
    (useful to label TRAIN/TEST blocks, e.g. header="\nTRAIN:").

    If class_names is provided, prints both:
      Class 0 (cat): 500 samples
    """
    targets = _get_targets_any_dataset(dataset)
    uniq, cnt = np.unique(targets, return_counts=True)

    if header is not None:
        print(header)

    print("Classes present:")
    for c, n in zip(uniq, cnt):
        name = ""
        if class_names is not None and int(c) < len(class_names):
            name = f" ({class_names[int(c)]})"
        print(f"  Class {int(c)}{name}: {int(n)} samples")

    return uniq.tolist(), cnt.tolist()
