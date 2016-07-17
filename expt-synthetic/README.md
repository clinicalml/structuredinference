## README for expt-synthetic
Author: Rahul G. Krishnan (rahul@cs.nyu.edu)

## Reproducing results from the paper
This folder contains the scripts to reproduce the synthetic experiments. 

Steps:

    * Run "python create_expt.py". This will yield a list of settings to run. Run them sequentially/parallel 
    * Run the ipython notebook in structuredinference/ipynb/synthetic to obtain the desired plots 

## Synthetic Datasets

The synthetic datasets are located in the theanomodels repository. See theanomodels/datasets/synthp.py 
* The file defines a dictionary, one entry for every synthetic dataset
* Each dataset's parameters are contained within sub-dictionaries. This includes the initial mean/covariance for the transition distribution, the fixed covariance for the emission distribution as well as the emission and transition functions.
* At model creation, this dictionary is embedded into the DKF and use to create the fixed parameters for the generative model
for which inference will be performed. 
* theanomodels/datasets/synthpTheano.py contains the theano implementations of the emission and transition distributions in the generative model 
