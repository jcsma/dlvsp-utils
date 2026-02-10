"""Top-level exports for the ``dlvsp_utils`` package.

This module re-exports the most commonly used helpers from the
``data``, ``metrics`` and ``viz`` submodules so interactive users can
simply ``from dlvsp_utils import select_classes_dataset`` in notebooks.

The detailed implementations and additional helpers remain available
under their submodules (``dlvsp_utils.data``, ``dlvsp_utils.metrics``
and ``dlvsp_utils.viz``) for users that prefer explicit imports.
"""

from .data import (
    select_classes_dataset,
    _get_targets_any_dataset,
    count_samples_dataset,
    inspect_dataset_classes,
    apply_imbalance,
    exponential_decay_samples,
    define_imbalance_profile,
)

from .viz import (
    plot_class_distribution,
    show_images,
)

from .metrics import (
    calculate_accuracy,
    compute_accuracy_stats,
    compute_accuracy_per_class,
    print_accuracy_report,
)

__all__ = [
    "select_classes_dataset",
    "_get_targets_any_dataset",
    "count_samples_dataset",
    "inspect_dataset_classes",
    "apply_imbalance",
    "exponential_decay_samples",
    "define_imbalance_profile",
    "plot_class_distribution",
    "show_images",
    "calculate_accuracy",
    "compute_accuracy_stats",
    "compute_accuracy_per_class",
    "print_accuracy_report",
]
