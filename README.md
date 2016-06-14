# structuredInf
Structured Inference Networks for Deep Kalman Filters 

## Goal
The goal of this package is to present a black box algorithm inference algorithm to learn models of time-series data. 
Learning is performed using a recognition network.

## Model
The figure below describes a simple model of time-series data.
Typically, (1) if you have enough training data (2)
if you would like to have a method for fast posterior inference at train 
and test time and (3) if your generative model has gaussian latent variables, this method would be a good fit
to learn your model. 

<img src=https://raw.githubusercontent.com/clinicalml/structuredinference/master/images/dkf.png?token=AA5BDBHi5YtrrTw5HTk_tNt6F97hc1Bqks5XaKQ9wA%3D%3D =20x20 alt="DKF" width="500" height="400"/>

The code optimizes the variational lower bound.

*Generative Model* The latent variables z1...zT and the observations x1...xT describe the generative process for the data. The figure depicts a simple state space model for time-varying data. 

*Inference Model* The black-box q(z1..zT|x1...xT) represents the inference network. There are several supported inference networks within this package. See below for details on how to use different ones. 

## Installation

### Requirements
This package has the following requirements:

python2.7

[Theano](https://github.com/Theano/Theano)
Used for automatic differentiations

[theanomodels] (https://github.com/clinicalml/theanomodels) 
Wrapper around theano that takes care of bookkeeping, saving/loading models etc. Clone the github repository
and add it to your PATH environment variable so that it is accessible by this package. 

[pykalman] (https://pykalman.github.io/) 
For running baseline UKFs/KFs

An NVIDIA GPU w/ atleast 6G of memory is recommended.

Once the requirements have been met, clone this repository and its ready to run. 

### Folder Structure
The following folders contain code to reproduct the result:
* expt-synthetic, expt-polyphonic: Contains code and instructions for reproducing results. 
* baselines/: Contains to run some of the baseline algorithms on the synthetic data
* ipynb/: Ipython notebooks for evaluation and building plots

The main files of interest are:
* parse_args_dkf.py: Contains the list of arguments that the model expects to be present. Looking through it is useful to understand the different knobs available to tune the model. 
* stinfmodel/dkf.py: Contains the code to construct the inference and generative model. The code is commented and should be readable.
* stinfmodel/evaluate_dkf.py: Contains code to evaluate the Deep Kalman Filter's performance during learning.
* stinfmodel/learning.py: Code for performing stochastic gradient ascent in the Evidence Lower Bound. 


## Dataset

### Format 
* The datasets within the model
* Create a file to load datasets into memory (see theanomodels/datasets/load.py for an example of how to load the dataset). This typically involves defining a train/validate split and masks for sequential data
* Create a training script to use dkf/models.py to create a DKF object to train/evaluate

### Running on different datasets


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
