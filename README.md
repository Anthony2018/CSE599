# CSE 490/599 G1 Introduction to Deep Learning


## What the heck is this codebase? ##

During this class you'll be building out your own neural network framework. We'll take care of some of the more boring parts like loading data, stringing things together, etc. so you can focus on the important parts.
We will be implementing everything using [Numpy](https://docs.scipy.org/doc/numpy/user/quickstart.html) and [Numba](https://numba.pydata.org/numba-doc/dev/user/5minguide.html) in a [Conda](https://docs.conda.io/en/latest/) environment.
If you are not familiar with them, take a few minutes to learn the ins and outs. PyTorch uses a Numpy-like interface, so it will be good to know for the other parts of the homework as well.

## Setup Conda and the codebase ##
First install Miniconda (https://docs.conda.io/en/latest/miniconda.html). Then run the following commands.

```bash
conda create -n dl-class python=3.6.9
git clone https://gitlab.com/danielgordon10/dl-class-2019a.git
cd dl-class-2019a
conda deactivate
conda env update -n dl-class -f environment.yml
conda activate dl-class
```
