# structuredInf
Structured Inference Networks for Deep Kalman Filters 

The repository contains code to train models from the workshop paper:

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

[theanomodels] (https://github.com/clinicalml/theanomodels)

[pykalman] (https://pykalman.github.io/)

NVIDIA GPU w/ atleast 6G of Memory

## Installation
* Follow the instructions to install theanomodels.
* Synthetic experiments in expt-synthetic
* Polyphonic experiments in expt-polyphonic

Use the .sh files provided to create and train the models used in order to obtain the reported numbers

## Running on your own datasets
* Create a file to load datasets into memory (see theanomodels/datasets/load.py for an example). This typically involves defining a train/validate split and masks for sequential data
* Create a training script to use dkf/models.py to create a DKF object to train/evaluate
