# dlvsp-utils

Utility functions for the practical assignments of the
"Deep Learning for Visual Signal Processing" course (IPCVAI, UAM).

Author: Juan Carlos San Miguel (<juancarlos.sanmiguel@uam.es>)

This small utility package is used across the course notebooks and
contains dataset helpers, simple metrics and plotting utilities.

Install from GitHub (after you push the repo):

```bash
pip install git+https://github.com/jcsma/dlvsp-utils.git
```

Usage in notebooks:

```python
from dlvsp_utils.data import select_classes_dataset, inspect_dataset_classes
from dlvsp_utils.metrics import calculate_accuracy, print_accuracy_report

train_ds, class_names = select_classes_dataset(train_full, ['cat','dog'])
inspect_dataset_classes(train_ds, class_names=class_names, header="\nTRAIN:")
```

Files included in this scaffold:

- `src/dlvsp_utils/data.py`  (dataset helpers)
- `src/dlvsp_utils/metrics.py` (accuracy and reporting)
- `src/dlvsp_utils/viz.py` (visualization helpers)
- `pyproject.toml`

Related links:

- [Universidad Autónoma de Madrid (UAM)](https://www.uam.es)
- [Escuela Politécnica Superior @ UAM](https://www.uam.es/EPS)
- [Interuniversity Master in Computer Vision and Artificial Intelligence (IPCVAI)](https://ipcvai.eu)