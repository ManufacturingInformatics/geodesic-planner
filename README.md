# Generating Continuous Paths On Learned Constraint Manifolds Using Policy Search

[![CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](http://creativecommons.org/licenses/by/4.0/])

<p align="center">
  <img src="assets/paper-model-2.png" width="500">
</p>

This repo is to act as a supplement to the work provided in our paper titled ***"Generating Continuous Paths On Learned Constraint Manifolds Using Policy Search"***. It is not designed to be production-level code, rather to inform readers as to our methodology.

**UPDATE (Date here):** Our work was accepted to IEEE/RSJ IROS 2024!

## Usage

### Prerequisites

We provide an ```environment.yml``` file that includes all of our packages. This has been tested in Python 3.11.5 on a system with the following specs:
```shell
OS: Ubuntu 20.04 LTS
Python Version: 3.10.10
GPU: NVIDIA GeForce RTX 3080
CPU: Intel Core i9-10920X 12C/24T
RAM: 64 GB
```

We recommend using a virtual environment such as virtualenv or Anaconda. We used Anaconda during our testing and will refer to the instructions as follows. To install and configure the environment, run:

```shell
conda env create -f environment.yml
```

Our implementation is built upon two other implementations of proximal policy optimisation (PPO) and latent space manifolds. Our PPO implementation is based on that by [Nikhil Barhate](https://github.com/nikhilbarhate99/PPO-PyTorch) and our latent space manifold implementation is adapted from the work in the paper ['Learning Riemannian Manifolds for Geodesic Motion Skills'](https://doi.org/10.15607/RSS.2021.XVII.082) by Beik-Mohammadi et al. and their implementation provided [here](https://github.com/boschresearch/GeodesicMotionSkills).

### Single Script Runfile

### Dockerfile Implementation
