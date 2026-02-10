<img width="3840" height="3840" alt="image" src="https://github.com/user-attachments/assets/bfb26a8e-48f0-4584-bffe-7e759ae15d77" /># dlvsp-utils

Utility functions for the practical assignments of the courses:

- **Deep Learning for Image and Video Processing: Learning Strategies and Applications**  
  (MUDLAVai, Universidad Autónoma de Madrid)  
  https://www.uam.es/uam/master-universitario-aprendizaje-profundo-tratamiento-senales-audio-video

- **Deep Learning for Visual Signal Processing**  
  (IPCVAI, Universidad Autónoma de Madrid)  
  https://ipcvai.eu

**Author:** Juan Carlos San Miguel  
📧 juancarlos.sanmiguel@uam.es
[![LinkedIn](https://img.shields.io/badge/-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/)
https://www.linkedin.com/in/jcsanmiguel/

---

## Overview

`dlvsp-utils` is a lightweight utility package used across the course notebooks.  
It provides reusable helpers for:

- dataset manipulation and inspection  
- accuracy computation and per-class reporting  
- simple visualization utilities for model analysis  

The goal is to reduce boilerplate code and keep the focus on **learning strategies and experimental analysis**.

---

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/jcsma/dlvsp-utils.git
```

---

## Usage in notebooks:

```python
from torchvision import datasets, transforms
from dlvsp_utils.data import select_classes_dataset, inspect_dataset_classes
from dlvsp_utils.metrics import calculate_accuracy, print_accuracy_report

train_full = datasets.CIFAR10(root="./data", train=True, download=True, transform=none)
train_ds, class_names = select_classes_dataset(train_full, ['cat','dog'])
inspect_dataset_classes(train_ds, class_names=class_names, header="\nTRAIN:")
```

---

## Package structure

The repository contains the following modules:
* `src/dlvsp_utils/data.py`
Dataset utilities (class selection, inspection, sampling helpers)
* `src/dlvsp_utils/metrics.py`
Accuracy computation and per-class performance reporting
* `src/dlvsp_utils/viz.py`
Visualization helpers for analysis and debugging
* `pyproject.toml`
Package configuration and dependencies

---

## Related links:

- [Universidad Autónoma de Madrid (UAM)](https://www.uam.es)
- [Escuela Politécnica Superior @ UAM](https://www.uam.es/EPS)
- [Erasmus Mundus Master in Computer Vision and Artificial Intelligence (IPCVai)](https://ipcvai.eu)
- [Master in Deep Learning for Audio and Visual Artificial Intelligence (MUDLAVai)](https://www.uam.es/uam/master-universitario-aprendizaje-profundo-tratamiento-senales-audio-video)
