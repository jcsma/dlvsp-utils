"""dlvsp_utils.viz

Small visualization helpers intended for interactive notebooks. These
functions are not full-featured plotting utilities but provide common
convenience operations used in the course.

Author: Juan Carlos San Miguel <juancarlos.sanmiguel@uam.es>
Utility functions for the practical assignments of the
"Deep Learning for Visual Signal Processing" course (IPCVAI, UAM).

- ``plot_class_distribution``: bar plot of class counts
- ``show_images``: quick grid of images from a dataset (with simple
    un-normalization support)

The functions use Matplotlib; import it lazily inside functions so the
module can be imported in environments without GUI backends.
"""
from typing import List
import numpy as np
import torch
from torch.utils.data import Dataset


def plot_class_distribution(sampled_counts: List[int], class_labels: List[str]):
    """Plot a bar chart showing how many samples belong to each class.

    Parameters
    ----------
    sampled_counts : list[int]
        Number of samples per class.
    class_labels : list[str]
        Labels to show on the x-axis.

    Notes
    -----
    The function imports Matplotlib lazily to avoid requiring plotting
    libraries during import-time in CI or headless contexts.
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.bar(class_labels, sampled_counts)
    plt.xlabel("Classes")
    plt.ylabel("Number of samples")
    plt.title("Distribution of Classes in the Dataset")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def show_images(dataset: Dataset, class_labels: List[str], n_show: int = 8, batch_size: int = 64, mean: float = 0.5, std: float = 0.5):
    """Display a small grid of images with their labels.

    Parameters
    ----------
    dataset : Dataset
        Any dataset that yields ``(image_tensor, label)`` from ``__getitem__``.
    class_labels : list[str]
        List of class names for titles.
    n_show : int
        Number of images to display (default 8).
    batch_size : int
        Batch size used only for random sampling.
    mean, std : float
        Values used to un-normalize image tensors for display. If your
        transforms used a tuple for channel-wise means/stds, either
        pass a scalar that approximates them or pre-process images.

    The function uses a DataLoader to sample images and constructs a
    Matplotlib figure. It clips values to [0, 1] before displaying.
    """
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    images, labels = next(iter(loader))

    images = images[:n_show]
    labels = labels[:n_show]

    fig, axes = plt.subplots(1, n_show, figsize=(2.5 * n_show, 2.5))
    if n_show == 1:
        axes = [axes]

    for img, y, ax in zip(images, labels, axes):
        img = img.permute(1, 2, 0).cpu().numpy()
        img = img * std + mean
        img = np.clip(img, 0.0, 1.0)
        ax.imshow(img)
        ax.set_title(class_labels[int(y)])
        ax.axis("off")

    plt.tight_layout()
    plt.show()
