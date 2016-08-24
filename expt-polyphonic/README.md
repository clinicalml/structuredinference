## Modeling Polyphonic Music with a Deep Kalman Filter
Author: Rahul G. Krishnan (rahul@cs.nyu.edu)

## Reproducing Numbers in the Paper
```
Run [dataset]_expts.sh to run the experiments on the polyphonic datasets
```

## Model Details
DKF
Standard Deep Kalman Filter. The default inference algorithm is set to be ST-R (see paper for more details) although
this can be modified through a variety of knobs primarily the -inference_model and the -var_model hyperparameters. 

DKF NADE
Use a nade to model the data rather than a distribution that treats dimensions of the data independantly. (can be used with the -usenade flag)

DKF AUG 
Augmented DKF. 
The emission distribution of the generative model is parameterized as p(x_t|x_{t-1}, z_t) (toggled with the -previnp flag)
and the transition distribution is parameterized as p(zt|z{t-1}) (can be activated with the -etype conditional flag)
