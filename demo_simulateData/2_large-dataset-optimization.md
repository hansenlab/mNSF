# Large Dataset Optimization Techniques for mNSF

**Authors:** Yi Wang, Kasper Hansen, and the mNSF Team  
**Date:** March 2025

## Overview

As spatial transcriptomics technologies advance, datasets are becoming increasingly large, with some experiments containing tens of thousands of spots and thousands of genes across multiple samples. Running mNSF on such datasets can be challenging due to memory limitations and computational complexity. This tutorial provides practical strategies for optimizing mNSF for large datasets, with particular emphasis on:

1. Efficient data chunking strategies
2. Optimal selection of induced points
3. Memory profiling and management
4. Parallel processing implementation
5. Pre-processing techniques for dimensionality reduction

```python
import mNSF
from mNSF import large_dataset_optimication as ldo
from mNSF import process_multiSample
from mNSF.NSF import preprocess, misc, visualize
from mNSF import training_multiSample
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from os import path
import time
import psutil
import random
```

## 1. Optimizing Induced Points

Induced points are a critical optimization in mNSF that reduce the computational complexity of Gaussian processes. Selecting the right number and distribution of induced points is essential for balancing accuracy and performance.

### 1.1 Number of Induced Points

The number of induced points directly impacts both computational efficiency and model accuracy:

```python
## Example
# Generate spatial coordinates in a 10x10 grid
x = np.random.uniform(0, 10, 3000)
y = np.random.uniform(0, 10, 3000)
X = np.column_stack((x, y))
X_df = pd.DataFrame(X, columns=['x', 'y'])
ldo.induced_points_analysis(X_df, percentages=[0.5, 0.7, 0.85, 1])
#    percentage  n_induced  time_seconds  memory_delta_GB  matrix_size_GB
# 0        0.50       1500     16.160952         0.067062        0.033528
# 1        0.70       2100     22.770177         0.026817        0.046939
# 2        0.85       2550     27.504231         0.020119        0.056997
# 3        1.00       3000     32.378558         0.020119        0.067055

```
<img src="induced_points_analysis.png" alt="Alt text" width="80%">

### 1.2 Strategic Selection of Induced Points

Rather than random selection, strategically choosing induced points can improve model accuracy:

```python

# example
ldo.visualize_induced_points(X_df, percentage=0.15)
```
<img src="visualize_induced_points.png" alt="Alt text" width="80%">
