import numpy as np

def loadDataset():
    dataset = {}
    Ntrain, Nvalid, Ntest = 10000,500,500
    T , dim_observations  = 100,40
    dataset['train']      = (np.random.randn(Ntrain,T, dim_observations)>0)*1.
    dataset['mask_train'] = np.ones((Ntrain,T))
    dataset['valid']      = (np.random.randn(Ntest,T,dim_observations)>0)*1.
    dataset['mask_valid'] = np.ones((Nvalid,T))
    dataset['test']       = (np.random.randn(Ntest,T,dim_observations)>0)*1.
    dataset['mask_test']  = np.ones((Ntest,T))
    dataset['dim_observations'] = dim_observations 
    return dataset
