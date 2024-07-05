# Generating Continuous Paths On Learned Constraint Manifolds Using Policy Search

[![CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](http://creativecommons.org/licenses/by/4.0/])

<p align="center">
  <img src="assets/paper-model-2.png" width="500">
</p>

This repo is to act as a supplement to the work provided in our paper titled ***"Generating Continuous Paths On Learned Constraint Manifolds Using Policy Search"***. It is not designed to be production-level code, rather to inform readers as to our methodology.

This will continued to be updated as it is an ongoing piece of research.  

**UPDATE (Date here):** Our paper was accepted to IEEE/RSJ IROS 2024!

## Usage

### Prerequisites

We provide an ```environment.yml``` file that includes all of our packages. This has been tested in Python 3.11.5 on a system with the following specs:

```shell
OS: Ubuntu 22.04 LTS
Python Version: 3.10.10
GPU: NVIDIA GeForce RTX 3080
CPU: Intel Core i9-10920X 12C/24T
RAM: 64 GB
```

We recommend using a virtual environment such as virtualenv or Anaconda. We used Anaconda during our testing and will refer to the instructions as follows. To install and configure the environment, run:

```shell
conda env create -f environment.yml
```

Our implementation is built upon two other implementations of proximal policy optimisation (PPO) and latent space manifolds. Our PPO implementation is adapted from that by [Nikhil Barhate](https://github.com/nikhilbarhate99/PPO-PyTorch) and our latent space manifold implementation is adapted from the work in the paper ['Learning Riemannian Manifolds for Geodesic Motion Skills'](https://doi.org/10.15607/RSS.2021.XVII.082) by Beik-Mohammadi et al. and their implementation provided [here](https://github.com/boschresearch/GeodesicMotionSkills).

### Training the Constraint Manifold

For training the manifold planner, we provide two sets of options. The first is a predefined shell script that allows some granularity with selecting options for training. You can run this script from the home directory of this repository. First give the script executable privileges on Ubuntu:

```shell
sudo chmod +x train_manifold.sh
```

Then run the script:

```shell
./train_manifold.sh -m <mode> -b <batch size> -d <num datapoints>
```

The `mode` parameter dictates whether the model is just training, testing or both. Batch size is the size of the update batches for PyTorch and the number of datapoints that are used to train the VAE manifold.

If you want to alter more of the training parameters, you can directly call the Python script inside the `manifold-learner` folder with the following parameters:

```shell
python3 main.py --mode <mode> --n_samples <num samples> \
    --epochs_kl <KL epochs> --epochs <epochs> --rbf_epochs <RBF epochs> \
    --lr <learning rate> --latent_max <latent max> \
    --batch_size <batch_size> --num_datapoints <num datapoints>
```

For more documentation regarding these variables, please find the variables within the code for their explanation. Further explanation can be found in the work by [Beik-Mohammedi et al.](https://github.com/boschresearch/GeodesicMotionSkills)

### Training the Geodesic Planner

Similar to the manifold learner, our manifold planner contains a single script file that can be used to run the planner with the default settings. First make the shell script executable:

```shell
sudo chmod +x train_geodesic.sh
```

Then you can run the script:

```shell
./train_geodesic.sh -e <episodes> -p <path> -r <rep>
```

Where `episodes` is the number of learning epochs to undergo for the policy optimisation, `path` is the save path for the model weights and `rep` determines the repitition number for this training cycle.

To run the default Python script, run the following command in the `geodesic-learner` folder:

```shell
python3 train_geodesic.py --episodes <episodes> --path <path> --rep <repetitions>
```

When the training has finished, the results can be plotted as shown below which is available in the paper. It shows the learned manifold with the sub-manifold constraint applied, then the learned shortest geodesic.

<p align="center">
  <img src="assets/paper-image.png" width="750">
</p>
