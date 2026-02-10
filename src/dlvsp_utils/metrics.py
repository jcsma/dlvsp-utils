"""dlvsp_utils.metrics

Utilities for computing accuracy and per-class performance metrics.

Author: Juan Carlos San Miguel <juancarlos.sanmiguel@uam.es>
Utility functions for the practical assignments of the
"Deep Learning for Visual Signal Processing" course (IPCVAI, UAM).

The functions in this module are designed for interactive analysis in
notebooks. They return plain Python/numpy types and also provide human
friendly printing helpers used extensively in the course notebooks.

Public functions
- ``calculate_accuracy``: run model evaluation over a DataLoader and
  return accuracy plus raw labels and predictions.
- ``compute_accuracy_stats``: numeric summaries (overall, macro,
  per-class, and counts) without printing.
- ``compute_accuracy_per_class``: convenience wrapper returning a
  dictionary class->accuracy.
- ``print_accuracy_report``: nicely formatted textual report for
  interactive inspection.

All functions accept lists, numpy arrays or torch tensors for labels
and predictions when appropriate.
"""
from typing import List, Tuple, Dict, Optional
import numpy as np
import torch


def calculate_accuracy(loader, model, device: Optional[torch.device] = None) -> Tuple[float, List[int], List[int]]:
    """Evaluate ``model`` on ``loader`` and return accuracy and raw outputs.

    Parameters
    ----------
    loader : torch.utils.data.DataLoader
        DataLoader yielding batches of ``(images, labels)``.
    model : torch.nn.Module
        Model to evaluate. The function sets the model to eval mode.
    device : torch.device, optional
        Device used for evaluation. If None, CUDA is selected when available.

    Returns
    -------
    (accuracy_percent, labels_list, preds_list)
        - ``accuracy_percent``: float in [0, 100]
        - ``labels_list``: list of int (ground-truth)
        - ``preds_list``: list of int (predicted labels)

    Notes
    -----
    This helper disables gradients and moves data to the chosen device
    for efficient evaluation. It is intended for notebook use where the
    returned label/pred lists are useful for further analysis.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.view(-1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return 100 * correct / max(1, total), all_labels, all_preds


def compute_accuracy_stats(labels, preds, num_classes: Optional[int] = None) -> Tuple[float, float, Dict[int, float], List[int]]:
    """Compute numeric accuracy summaries from labels and preds.

    Parameters
    ----------
    labels, preds : array-like
        Sequences of integer class labels and predicted labels.
    num_classes : int, optional
        If provided, guarantees the returned class_counts length.

    Returns
    -------
    overall_acc, macro_acc, per_class_acc, class_counts
        - ``overall_acc``: micro accuracy in [0..1]
        - ``macro_acc``: mean of per-class accuracies (ignores empty classes)
        - ``per_class_acc``: dict class->accuracy (0..1 or nan)
        - ``class_counts``: list[int] with number of samples per class
    """
    labels = np.asarray(labels, dtype=int)
    preds = np.asarray(preds, dtype=int)
    if num_classes is None:
        num_classes = int(max(labels.max(initial=0), preds.max(initial=0))) + 1

    per_class_acc = {}
    class_counts = np.zeros(num_classes, dtype=int)

    correct_total = (preds == labels).sum()
    overall_acc = correct_total / max(1, len(labels))

    for c in range(num_classes):
        idx = (labels == c)
        n = int(idx.sum())
        class_counts[c] = n
        if n == 0:
            per_class_acc[c] = float('nan')
        else:
            per_class_acc[c] = float((preds[idx] == labels[idx]).mean())

    present_accs = [a for a in per_class_acc.values() if not np.isnan(a)]
    macro_acc = float(np.mean(present_accs)) if present_accs else float('nan')

    return overall_acc, macro_acc, per_class_acc, class_counts.tolist()


def compute_accuracy_per_class(train_labels, train_preds):
    """Return a mapping from class index to accuracy value.

    Accepts lists, numpy arrays or torch tensors. If ``train_preds`` has
    shape (N, C) the function converts it to predicted indices with
    ``argmax``.
    """
    train_labels = torch.as_tensor(train_labels)
    train_preds = torch.as_tensor(train_preds)
    if train_preds.ndim > 1:
        train_preds = torch.argmax(train_preds, dim=1)

    classes = torch.unique(train_labels)
    accuracy_per_class = {}
    for c in classes:
        idxs = (train_labels == c)
        if idxs.sum() == 0:
            accuracy_per_class[int(c)] = float('nan')
            continue
        class_acc = (train_preds[idxs] == train_labels[idxs]).float().mean()
        accuracy_per_class[int(c)] = class_acc.item()
    return accuracy_per_class


def print_accuracy_report(labels, preds, class_names: List[str], header: str = "Report", samples_per_class: Optional[List[int]] = None):
    """Print a human-friendly accuracy report useful in notebooks.

    Parameters
    ----------
    labels, preds : array-like
        Ground-truth and predicted labels.
    class_names : list[str]
        Human-readable names for each class (order defines index->name).
    header : str
        Header string printed before the table.
    samples_per_class : list[int], optional
        Optional sample counts to display instead of inferred counts.
    """
    num_classes = len(class_names)
    overall_acc, macro_acc, per_class_acc, class_counts = compute_accuracy_stats(labels, preds, num_classes=num_classes)

    print(header)
    print(f"Overall accuracy (micro, all samples): {overall_acc*100:.2f}%")
    print(f"Mean per-class accuracy (macro):        {macro_acc*100:.2f}%\n")

    print("Per-class accuracy:")
    accs = []
    for c in range(num_classes):
        name = class_names[c] if c < len(class_names) else f"Class {c}"
        acc = per_class_acc.get(c, float('nan'))
        if np.isnan(acc):
            print(f"{name}: N/A (no samples)")
            continue
        accs.append(acc)
        if samples_per_class is not None:
            ns = samples_per_class[c]
            nt = int(np.sum(samples_per_class))
            print(f"{name}: {acc*100:.2f}% (Samples: {ns}/{nt})")
        else:
            print(f"{name}: {acc*100:.2f}% (Samples: {class_counts[c]})")

    if accs:
        print(f"\nStd of per-class accuracies:            {np.std(accs)*100:.2f}%")
