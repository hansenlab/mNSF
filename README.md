# mNSF
This GitHub repository is associated with our paper available on bioRxiv: [https://www.biorxiv.org/content/10.1101/2024.07.01.599554v1](https://www.biorxiv.org/content/10.1101/2024.07.01.599554v1)

All code to analyze the data and generate figures is available at [https://github.com/hansenlab/mNSF_paper](https://github.com/hansenlab/mNSF_paper)

Tutorial for using mNSF on example datasets are publicly available at [https://github.com/hansenlab/mNSF/blob/main/tutorial/mnsf-tutorial-dlpfc.md](https://github.com/hansenlab/mNSF/blob/main/tutorial/mnsf-tutorial-dlpfc.md) (use DLPFC data as example) and [https://github.com/hansenlab/mNSF/blob/main/tutorial/mnsf-tutorial-mouse.md](https://github.com/hansenlab/mNSF/blob/main/tutorial/mnsf-tutorial-mouse.md) (use mouse saggital section data as example).


### Demo notebooks of mNSF using simulated data

1. **[Getting Started with mNSF](1_getting-started-mnsf(2).md)**
   - Basic end-to-end workflow using synthetic data
   - Data preparation, model training, and result visualization
   - Parameter selection guidance

2. **[Large Dataset Optimization](2_large-dataset-optimization(1).md)**
   - Techniques for handling large-scale spatial transcriptomics data
   - Optimizing induced points selection
   - Memory and computational efficiency strategies

3. **[Effect of Parameter Selection](3_effect_of_parameter_selection.md)**
   - In-depth analysis of how different parameters affect mNSF performance
   - Interactive visualizations of parameter interactions
   - Practical guidelines for parameter tuning

4. **[Moran's I Tutorial](4_Morans_I_tutorial.md)**
   - Understanding spatial autocorrelation in mNSF factors
   - Calculating and interpreting Moran's I statistics
   - Visualizing different levels of spatial structure

5. **[Number of Factors Selection](5_number_of_factors_selection(1).md)**
   - Statistical approaches for determining optimal factor count
   - Using deviance explained and elbow methods
   - Practical guidelines for different dataset complexities

6. **[Factor Interpretability Guide](6_factor_interpretability(1).md)**
   - Connecting mNSF factors to biological meaning
   - Techniques for visualizing gene-factor relationships
   - Step-by-step interpretation framework

## Installation

You can install everything from the PyPI repository using `pip install -e .` but Tensorflow will most likely not install. A safer way would be to use conda to setup most of the packages then use pip to install.

### Install using pip

1. Git clone and activate your environment of choice.
2. [Install tensorflow](https://www.tensorflow.org/install).
3. `pip install -e .`

### Install using conda/mamba

1. Git clone this repo `git clone https://github.com/hansenlab/mNSF/` and enter `cd mNSF`.
2. Install `conda`. I recommend this distribution: https://github.com/conda-forge/miniforge. Do not install the full `anaconda`, it's highly bloated.
3. Create a new environment and install using

```sh
conda env create -n mnsf -f environment.yml
conda activate mnsf
```
The package should be available right away.

3. Install tensorflow.

<details>
  <summary>CPU only</summary>
  
    
    conda install tensorflow
    
</details>

<details>
  <summary>GPU</summary>
    If you have a GPU and is operating in a Linux system, you can in the `mnsf` environment.
  
    
    conda install tensorflow-gpu
    
</details>

## Development

This package is managed by `twine`. Assuming `twine` is installed in your python version, you build the distribution by
```
python setup.py sdist
```
inside the repository directory, and then you upload to PyPI by
```
twine upload dist/*
```
(requires an account on PyPI)

##
