# mNSF
This GitHub repository is associated with our paper available on bioRxiv: [https://www.biorxiv.org/content/10.1101/2024.07.01.599554v1](https://www.biorxiv.org/content/10.1101/2024.07.01.599554v1)

All code to analyze the data and generate figures is available at [https://github.com/hansenlab/mNSF_paper](https://github.com/hansenlab/mNSF_paper)

 Tutorials for using mNSF are publicly available at [https://github.com/hansenlab/mNSF/blob/main/tutorial/mnsf-tutorial-dlpfc.md](https://github.com/hansenlab/mNSF/blob/main/tutorial/mnsf-tutorial-dlpfc.md) (use DLPFC data as example) and [https://github.com/hansenlab/mNSF/blob/main/tutorial/mnsf-tutorial-mouse.md](https://github.com/hansenlab/mNSF/blob/main/tutorial/mnsf-tutorial-mouse.md) (use mouse saggital section data as example).


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
