
from theano import config
import numpy as np
import time
"""
Functions for evaluating a DKF object
"""
def infer(dkf, dataset, indicators, actions, mask):
    """ Posterior Inference using recognition network 
    Returns: z,mu,logcov (each a 3D tensor) Remember to multiply each by the mask of the dataset before
    using the latent variables
    """
    dkf.resetDataset(dataset, indicators, actions, mask, quiet=True)
    assert len(dataset.shape)==3,'Expecting 3D tensor for data' 
    assert dataset.shape[2]==dkf.params['dim_observations'],'Data dim. not matching'
    return dkf.posterior_inference(idx=np.arange(dataset.shape[0]))

def evaluateBound(dkf, dataset, indicators, actions, mask, batch_size,S=2, normalization = 'frame', additional={}):
    """ Evaluate ELBO """
    bound = 0
    start_time = time.time()
    N = dataset.shape[0]
    tsbn_bound = 0
    dkf.resetDataset(dataset, indicators, actions, mask)
    for bnum,st_idx in enumerate(range(0,N,batch_size)):
        end_idx = min(st_idx+batch_size, N)
        idx_data= np.arange(st_idx,end_idx)
        maxS = S
        bound_sum, tsbn_bound_sum = 0, 0
        for s in range(S):
            if s>0 and s%500==0:
                dkf._p('Done '+str(s))
            batch_vec= dkf.evaluate(idx=idx_data)
            M = mask[idx_data]
            if np.any(np.isnan(batch_vec)) or np.any(np.isinf(batch_vec)):
                dkf._p('NaN detected during evaluation. Ignoring this sample')
                maxS -=1
                continue
            else:
                tsbn_bound_sum+=(batch_vec/M.sum(1,keepdims=True)).sum()
                bound_sum+=batch_vec.sum()
        tsbn_bound += tsbn_bound_sum/float(max(maxS*N,1.))
        bound  += bound_sum/float(max(maxS,1.))
    bound /= float(mask.sum())
    end_time   = time.time()
    dkf._p(('(Evaluate) Validation Bound: %.4f [Took %.4f seconds], TSBN Bound: %.4f')%(bound,end_time-start_time,tsbn_bound))
    additional['tsbn_bound'] = tsbn_bound
    return bound

def impSamplingNLL(dkf, dataset, mask, batch_size, S = 2, normalization = 'frame'):
    """ Importance sampling based log likelihood """
    ll = 0
    start_time = time.time()
    N = dataset.shape[0]
    dkf.resetDataset(dataset,indicators,actions,mask)
    for bnum,st_idx in enumerate(range(0,N,batch_size)):
        end_idx = min(st_idx+batch_size, N)
        idx_data= np.arange(st_idx,end_idx)
        maxS = S
        lllist = []
        for s in range(S):
            if s>0 and s%500==0:
                dkf._p('Done '+str(s))
            batch_vec = dkf.likelihood(idx=idx_data)
            if np.any(np.isnan(batch_vec)) or np.any(np.isinf(batch_vec)):
                dkf._p('NaN detected during evaluation. Ignoring this sample')
                maxS -=1
                continue
            else:
                lllist.append(batch_vec)
        ll  += dkf.meanSumExp(np.concatenate(lllist,axis=1), axis=1).sum()
    if normalization=='frame':
        ll /= float(mask.sum())
    elif normalization=='sequence':
        ll /= float(N)
    else:
        assert False,'Invalid normalization specified'
    end_time   = time.time()
    dkf._p(('(Evaluate w/ Imp. Sampling) Validation LL: %.4f [Took %.4f seconds]')%(ll,end_time-start_time))
    return ll

def sampleGaussian(dkf,mu,logcov):
        return mu + np.random.randn(*mu.shape)*np.exp(0.5*logcov)
