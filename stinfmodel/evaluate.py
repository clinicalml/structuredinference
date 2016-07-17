
from theano import config
import numpy as np
import time
"""
Functions for evaluating a DKF object
"""
def infer(dkf, dataset):
    """ Posterior Inference using recognition network 
    Returns: z,mu,logcov (each a 3D tensor) Remember to multiply each by the mask of the dataset before
    using the latent variables
    """
    assert len(dataset.shape)==3,'Expecting 3D tensor for data' 
    assert dataset.shape[2]==dkf.params['dim_observations'],'Data dim. not matching'
    eps  = np.random.randn(dataset.shape[0], 
                           dataset.shape[1], 
                           dkf.params['dim_stochastic']).astype(config.floatX)
    return dkf.posterior_inference(X=dataset.astype(config.floatX), eps=eps)

def evaluateBound(dkf, dataset, mask, batch_size,S=2, normalization = 'frame', additional={}):
    """ Evaluate ELBO """
    bound = 0
    start_time = time.time()
    N = dataset.shape[0]
    
    tsbn_bound  = 0
    for bnum,st_idx in enumerate(range(0,N,batch_size)):
        end_idx = min(st_idx+batch_size, N)
        X       = dataset[st_idx:end_idx,:,:].astype(config.floatX)
        M       = mask[st_idx:end_idx,:].astype(config.floatX)
        U       = None
        
        #Reduce the dimensionality of the tensors based on the maximum size of the mask
        maxT    = int(np.max(M.sum(1)))
        X       = X[:,:maxT,:]
        M       = M[:,:maxT]
        eps     = np.random.randn(X.shape[0],maxT,dkf.params['dim_stochastic']).astype(config.floatX)
        maxS = S
        bound_sum, tsbn_bound_sum = 0, 0
        for s in range(S):
            if s>0 and s%500==0:
                dkf._p('Done '+str(s))
            eps     = np.random.randn(X.shape[0],maxT,dkf.params['dim_stochastic']).astype(config.floatX)
            batch_vec= dkf.evaluate(X=X, M=M, eps=eps)
            if np.any(np.isnan(batch_vec)) or np.any(np.isinf(batch_vec)):
                dkf._p('NaN detected during evaluation. Ignoring this sample')
                maxS -=1
                continue
            else:
                tsbn_bound_sum+=(batch_vec/M.sum(1,keepdims=True)).sum()
                bound_sum+=batch_vec.sum()
        tsbn_bound += tsbn_bound_sum/float(max(maxS*N,1.))
        bound  += bound_sum/float(max(maxS,1.))
    if normalization=='frame':
        bound /= float(mask.sum())
    elif normalization=='sequence':
        bound /= float(N)
    else:
        assert False,'Invalid normalization specified'
    end_time   = time.time()
    dkf._p(('(Evaluate) Validation Bound: %.4f [Took %.4f seconds], TSBN Bound: %.4f')%(bound,end_time-start_time,tsbn_bound))
    additional['tsbn_bound'] = tsbn_bound
    return bound


def impSamplingNLL(dkf, dataset, mask, batch_size, S = 2, normalization = 'frame'):
    """ Importance sampling based log likelihood """
    ll = 0
    start_time = time.time()
    N = dataset.shape[0]
    for bnum,st_idx in enumerate(range(0,N,batch_size)):
        end_idx = min(st_idx+batch_size, N)
        X       = dataset[st_idx:end_idx,:,:].astype(config.floatX)
        M       = mask[st_idx:end_idx,:].astype(config.floatX)
        U       = None
        maxT    = int(np.max(M.sum(1)))
        X       = X[:,:maxT,:]
        M       = M[:,:maxT]
        eps     = np.random.randn(X.shape[0],maxT,dkf.params['dim_stochastic']).astype(config.floatX)
        maxS = S
        lllist = []
        for s in range(S):
            if s>0 and s%500==0:
                dkf._p('Done '+str(s))
            eps      = np.random.randn(X.shape[0],maxT,
                                       dkf.params['dim_stochastic']).astype(config.floatX)
            batch_vec = dkf.likelihood(X=X, M=M, eps=eps)
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

def sample(dkf, nsamples=100, T=10, additional = {}):
    """
                                  Sample from Generative Model
    """
    assert T>1, 'Sample atleast 2 timesteps'
    #Initial sample
    z      = np.random.randn(nsamples,1,dkf.params['dim_stochastic']).astype(config.floatX)
    all_zs = [np.copy(z)]
    additional['mu']     = []
    additional['logcov'] = []
    for t in range(T-1):
        mu,logcov = dkf.transition_fxn(z)
        z           = dkf.sampleGaussian(mu,logcov).astype(config.floatX)
        all_zs.append(np.copy(z))
        additional['mu'].append(np.copy(mu))
        additional['logcov'].append(np.copy(logcov))
    zvec = np.concatenate(all_zs,axis=1)
    additional['mu']     = np.concatenate(additional['mu'], axis=1)
    additional['logcov'] = np.concatenate(additional['logcov'], axis=1)
    return dkf.emission_fxn(zvec), zvec
    
