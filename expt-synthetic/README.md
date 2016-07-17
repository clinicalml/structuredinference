## README for expt-synthetic
Author: Rahul G. Krishnan (rahul@cs.nyu.edu)

## Reproducing results from the paper
This folder contains the scripts to reproduce the synthetic experiments. 
Steps:
    * Run "python create_expt.py". This will yield a list of settings to run. Run them sequentially/parallel 
    * Run the ipython notebook in structuredinference/ipynb/synthetic to obtain the desired plots 

## Synthetic Datasets

The synthetic datasets are located in the theanomodels repository. See theanomodels/datasets/synthp.py 
* The file contains the initial mean/covariance for the transition distribution, the fixed covariance for the emission
distribution as well as the emission and transition functions.
* The format of
* theanomodels/datasets/synthpTheano.py contains the theano implementations of the emission and transition distributions in the generative model 
