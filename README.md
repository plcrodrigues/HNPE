# HNPE: Hierarchical Neural Posterior Estimation

## What you will find here

This repository contains code related to the method we have proposed in our
[NeurIPS 2021](https://openreview.net/forum?id=E8BxwYR8op) paper 

> P Rodrigues, T Moreau, G Louppe, A Gramfort "*HNPE: Leveraging Global Parameters for Neural Posterior Estimation*". Proc. Advances in Neural Information Processing Systems (NeurIPS) 34, 2021

We have included basic code for reproducing the two numerical illustrations
presented in the paper:

- A toy model for which we can derive all analytic properties of the posterior
distribution
- The Jansen-Rit neural mass model that we have used to relate physiological
parameters to real EEG signals

Most of our code is based on the wonderful [`sbi` package](https://github.com/mackelab/sbi).

## Setup

Clone the GitHub repository on your local computer, and install the package by
running:

```
pip install -e .
```

All dependencies with pinned versions are listed in `requirements.txt` if you
want to reproduce the exact environment we used to run the original
computations.

To ensure that your code finds the right scripts, run:

```
python -c 'import hnpe'
```

Note that you might want to create a *conda environment* before doing all these
installations, e.g.:

```bash
conda create -n hnpe_env python=3.7
```

## Usage


### Example 1 - Toy Model

To obtain an approximation of the posterior distribution of our toy model when
only one observation is available and no noise is added to it, you should go to
`Ex1-ToyModel` and enter in your terminal 

```bash
python inference.py
```

This creates an approximation of the posterior distribution and stores its
parameters in `/results`

To check the results, you can simply enter

```bash
python inference.py --viz
```

Now if you would like to see what happens to the posterior distribution when
*N = 10* extra observations are available, you should enter

```bash
python inference.py --nextra 10
```

To include noise in the observations, you should run, for instance,

```bash
python inference.py --nextra 10 --noise 0.05
```

### Example 2 - Jansen & Rit Neural Mass Model

The way of doing things for `Ex2-JRNMM` is exactly the same, except for a few
choices of input parameters. 

Please do

```bash
python inference.py --help
```

for a list of all options available for each example.

>  **Important**: To run the examples in `Ex2-JRNMM` you will need to make sure
that the R code in https://github.com/massimilianotamborrino/sdbmpABC runs on
your computer! This is a C++ implementation of the Jansen-Rit neural mass model
compiled for R and which we bind to python.
