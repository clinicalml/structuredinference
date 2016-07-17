## README for expt-synthetic
Author: Rahul G. Krishnan (rahul@cs.nyu.edu)

## Reproducing results from the paper
This folder contains the scripts to reproduce the synthetic experiments. 

Steps:

```
    Run "python create_expt.py". This will yield a list of settings to run. Run them sequentially/in parallel 
    Run the ipython notebook in structuredinference/ipynb/synthetic to obtain the desired plots 
```
## Synthetic Datasets

The synthetic datasets are located in the theanomodels repository. See theanomodels/datasets/synthp.py 
* The file defines a dictionary, one entry for every synthetic dataset
* Each dataset's parameters are contained within sub-dictionaries. This includes the initial mean/covariance for the transition distribution, the fixed covariance for the emission distribution as well as the emission and transition functions. (params_synthetic['synthetic9'] is a dict with keys such as trans_fxn, emis_fxn, init_cov etc. trans_fxn, emis_fxn are pointers to functions)
* At model creation, this dictionary is embedded into the DKF and use to create the transition and emission function for the generative model. The DKF directly uses the theano implementations of the transition and emission functions in (theanomodels/datasets/synthpTheano.py)
