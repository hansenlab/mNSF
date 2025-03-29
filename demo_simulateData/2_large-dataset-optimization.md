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

## 1. Understanding Computational Challenges

The computational complexity of mNSF increases significantly with dataset size:

- Memory usage scales with the number of spots, genes, and factors
- Computation time increases with the number of spots due to Gaussian Process calculations
- Multiple samples compound these challenges

Before diving into optimization techniques, let's understand the main bottlenecks:

```python
# For estimating memory requirements, see large_dataset_optimization.py
```

## 2. Memory Management Techniques

As the size of your dataset grows, memory management becomes increasingly important. Here are strategies to minimize memory usage during mNSF analysis:

### 2.1 TensorFlow Memory Optimization

```python
# For TensorFlow memory optimization, see large_dataset_optimization.py
```

### 2.1 Memory Monitoring

It's important to monitor memory usage during mNSF runs, especially for large datasets. Here's a wrapper that can help track memory usage during model training:

```python
# For memory monitoring functions, see large_dataset_optimization.py
```

## 3. Optimizing Induced Points

Induced points are a critical optimization in mNSF that reduce the computational complexity of Gaussian processes. Selecting the right number and distribution of induced points is essential for balancing accuracy and performance.

### 3.1 Number of Induced Points

The number of induced points directly impacts both computational efficiency and model accuracy:

```python
# For induced points analysis functions, see large_dataset_optimization.py
```
<img src="induced_points_analysis.png" alt="Alt text" width="80%">

### 3.2 Strategic Selection of Induced Points

Rather than random selection, strategically choosing induced points can improve model accuracy:

```python
# For strategic induced points selection functions, see large_dataset_optimization.py
```
<img src="visualize_induced_points.png" alt="Alt text" width="80%">

## 4. Additional Resources

All functions discussed in this document have been moved to a Python module for easier access. To use these optimization techniques, import the module:

```python
import large_dataset_optimization as ldo

# Estimate memory for your dataset
memory_estimate = ldo.estimate_memory(50000, 5000, 15, 3)
print(memory_estimate)

# Configure TensorFlow for optimal memory usage
ldo.configure_tensorflow_memory()

# Monitor memory during training
list_fit, memory_stats = ldo.memory_monitored_training(
    list_fit, pp, list_Dtrain, list_D, 
    num_epochs=500, nsample=1, nchunk=1
)

# Analyze and visualize different induced point strategies
X_df = pd.DataFrame({'x': x_coords, 'y': y_coords})
ldo.visualize_induced_points(X_df, percentage=0.15)
```

For more information on the mNSF algorithm and implementation details, please refer to the main documentation.
