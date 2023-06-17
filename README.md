# mNSF

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
    ```sh
    conda install tensorflow
    ```
</details>

<details>
  <summary>GPU</summary>
    If you have a GPU and is operating in a Linux system, you can in the `mnsf` environment.
    ```sh
    conda install tensorflow-gpu
    ```
</details>
