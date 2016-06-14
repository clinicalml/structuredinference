# structuredInf
Structured Inference Networks for Deep Kalman Filters 

## Goal
The goal of this package is to present a black box algorithm inference algorithm to learn models of time-series data. 
Learning is performed using a recognition network.

## Model
The figure below describes a simple model of time-series data. The observations x1....xT 

*Generative Model* The latent variables z1...zT and the observations x1...xT describe the generative process for the data. The figure depicts a simple state space model for time-varying data. 

<img src=https://raw.githubusercontent.com/clinicalml/structuredinference/master/images/dkf.png?token=AA5BDGHzqw3UohY9Kkd-Wj2JXCWCye7cks5XZ4y8wA%3D%3D =20x20 alt="DKF" width="500" height="400"/>

*Inference Model* The black-box q(z1..zT|x1...xT) represents the inference network. There are several supported inference networks within this package. See below for details on how to use different ones. 

### I want to learn a model of time series data. Should I use this?
The success or failure of this method is hard to predict but the hope is that the package is easy enough to work with
that the decision to use or not may be made quickly. Typically, (1) if you have enough training data (2)
if you would like to have a method for fast posterior inference at train 
and test time and (3) if your generative model has gaussian latent variables, this method would be a good fit. 


## Installation

# Requirements
This package has the following requirements. 

python2.7

[Theano](https://github.com/Theano/Theano)
Used for automatic differentiations

[theanomodels] (https://github.com/clinicalml/theanomodels) 
Wrapper around theano that takes care of bookkeeping, saving/loading models etc. Clone the github repository
and add it to your PATH environment variable so that it is accessible by this package. 

[pykalman] (https://pykalman.github.io/) 
For running baseline UKFs/KFs

An NVIDIA GPU w/ atleast 6G of memory is recommended.

# Setup
Once the requirements have been met, clone this repository and its ready to run. 

## Folder Structure
expt-synthetic, expt-polyphonic: Contains code and instructions for reproducing results. 
model: Contains the code for both the generative model and the inference networks.


## Dataset

# Format 
* The datasets within the model
* Create a file to load datasets into memory (see theanomodels/datasets/load.py for an example of how to load the dataset). This typically involves defining a train/validate split and masks for sequential data
* Create a training script to use dkf/models.py to create a DKF object to train/evaluate

# Instructions for running on my own data


## References: 
```
@article{krishnan2015deep,
  title={Deep Kalman Filters},
  author={Krishnan, Rahul G and Shalit, Uri and Sontag, David},
  journal={arXiv preprint arXiv:1511.05121},
  year={2015}
}
```


## Installation
* Follow the instructions to install theanomodels.
* Synthetic experiments in expt-synthetic
* Polyphonic experiments in expt-polyphonic

Use the .sh files provided to create and train the models used in order to obtain the reported numbers

## Model configuration
* The settings in the configuration file reflect the settings used in the paper. To use the code in your own application, 
you may modify the model parameters to suit your needs. See parse_args_dkf.py for different options on the same. 
