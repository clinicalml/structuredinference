# structuredInf
Structured Inference Networks for Deep Kalman Filters 

## Goal

![alt tag](https://raw.githubusercontent.com/clinicalml/structuredinference/master/images/dkf.png?token=AA5BDGHzqw3UohY9Kkd-Wj2JXCWCye7cks5XZ4y8wA%3D%3D)

## Implementation

## Dataset
* The datasets within the model
* Create a file to load datasets into memory (see theanomodels/datasets/load.py for an example of how to load the dataset). This typically involves defining a train/validate split and masks for sequential data
* Create a training script to use dkf/models.py to create a DKF object to train/evaluate


## References: 
```
@article{krishnan2015deep,
  title={Deep Kalman Filters},
  author={Krishnan, Rahul G and Shalit, Uri and Sontag, David},
  journal={arXiv preprint arXiv:1511.05121},
  year={2015}
}
```

## Requirements
python2.7

[Theano](https://github.com/Theano/Theano)

[theanomodels] (https://github.com/clinicalml/theanomodels) Wrapper around theano to take care of model bookkeeping

[pykalman] (https://pykalman.github.io/) For running baseline UKFs/KFs

NVIDIA GPU w/ atleast 6G of Memory

## Installation
* Follow the instructions to install theanomodels.
* Synthetic experiments in expt-synthetic
* Polyphonic experiments in expt-polyphonic

Use the .sh files provided to create and train the models used in order to obtain the reported numbers

## Model configuration
* The settings in the configuration file reflect the settings used in the paper. To use the code in your own application, 
you may modify the model parameters to suit your needs. See parse_args_dkf.py for different options on the same. 
