"""
Functions for learning with a DKF object
"""
import evaluate as DKF_evaluate
import numpy as np
from utils.misc import saveHDF5
import time
from theano import config

def learn(dkf, dataset, mask, epoch_start=0, epoch_end=1000, 
          batch_size=200, shuffle=False,
          savefreq=None, savefile = None, 
          dataset_eval = None, mask_eval = None, 
          replicate_K = None,
          normalization = 'frame'):
    """
                                            Train DKF
    """
    assert not dkf.params['validate_only'],'cannot learn in validate only mode'
    assert len(dataset.shape)==3,'Expecting 3D tensor for data'
    assert dataset.shape[2]==dkf.params['dim_observations'],'Dim observations not valid'
    N = dataset.shape[0]
    idxlist   = range(N)
    batchlist = np.split(idxlist, range(batch_size,N,batch_size))

    bound_train_list,bound_valid_list,bound_tsbn_list,nll_valid_list = [],[],[],[]
    p_norm, g_norm, opt_norm = None, None, None

    #Lists used to track quantities for synthetic experiments
    mu_list_train, cov_list_train, mu_list_valid, cov_list_valid = [],[],[],[]

    #Start of training loop
    for epoch in range(epoch_start, epoch_end):
        #Shuffle
        if shuffle:
            np.random.shuffle(idxlist)
            batchlist = np.split(idxlist, range(batch_size,N,batch_size))
        #Always shuffle order the batches are presented in
        np.random.shuffle(batchlist)

        start_time = time.time()
        bound = 0
        for bnum, batch_idx in enumerate(batchlist):
            batch_idx  = batchlist[bnum]
            X       = dataset[batch_idx,:,:].astype(config.floatX)
            M       = mask[batch_idx,:].astype(config.floatX)
            U       = None

            #Tack on 0's if the matrix size (optimization->theano doesnt have to redefine matrices)
            if X.shape[0]<batch_size:
                Nremaining = int(batch_size-X.shape[0])
                X   = np.concatenate([X,np.zeros((Nremaining,X.shape[1],
                                                  X.shape[2]))],axis=0).astype(config.floatX)
                M   = np.concatenate([M,np.zeros((Nremaining,X.shape[1]))],axis=0).astype(config.floatX)

            #Reduce the dimensionality of the tensors based on the maximum size of the mask
            maxT    = int(np.max(M.sum(1)))
            X       = X[:,:maxT,:]
            M       = M[:,:maxT]


            eps     = np.random.randn(X.shape[0],maxT,dkf.params['dim_stochastic']).astype(config.floatX)
            batch_bound, p_norm, g_norm, opt_norm, negCLL, KL, anneal = dkf.train_debug(X=X, M=M, eps=eps)

            #Number of frames
            M_sum = M.sum()
            #Correction for replicating batch
            if replicate_K is not None:
                batch_bound, negCLL, KL = batch_bound/replicate_K, negCLL/replicate_K, KL/replicate_K, 
                M_sum   = M_sum/replicate_K
            #Update bound
            bound  += batch_bound
            ### Display ###
            if bnum%10==0:
                if normalization=='frame':
                    bval = batch_bound/float(M_sum)
                elif normalization=='sequence':
                    bval = batch_bound/float(X.shape[0])
                else:
                    assert False,'Invalid normalization'
                dkf._p(('Bnum: %d, Batch Bound: %.4f, |w|: %.4f, |dw|: %.4f, |w_opt|: %.4f')%(bnum,bval,p_norm, g_norm, opt_norm)) 
                dkf._p(('-veCLL:%.4f, KL:%.4f, anneal:%.4f')%(negCLL, KL, anneal))
        if normalization=='frame':
            bound /= float(mask.sum())
        elif normalization=='sequence':
            bound /= float(N)
        else:
            assert False,'Invalid normalization'
        bound_train_list.append((epoch,bound))
        end_time   = time.time()
        dkf._p(('(Ep %d) Bound: %.4f [Took %.4f seconds] ')%(epoch, bound, end_time-start_time))
        
        #Save at intermediate stages
        if savefreq is not None and epoch%savefreq==0:
            assert savefile is not None, 'expecting savefile'
            dkf._p(('Saving at epoch %d'%epoch))
            dkf._saveModel(fname = savefile+'-EP'+str(epoch))

            intermediate = {}
            if dataset_eval is not None and mask_eval is not None:
                tmpMap = {}
                bound_valid_list.append(
                    (epoch, 
                     DKF_evaluate.evaluateBound(dkf, dataset_eval, mask_eval, batch_size=batch_size, 
                                              additional = tmpMap, normalization=normalization)))
                bound_tsbn_list.append((epoch, tmpMap['tsbn_bound']))
                nll_valid_list.append(
                    DKF_evaluate.impSamplingNLL(dkf, dataset_eval, mask_eval, batch_size,
                                                                  normalization=normalization))
            intermediate['valid_bound'] = np.array(bound_valid_list)
            intermediate['train_bound'] = np.array(bound_train_list)
            intermediate['tsbn_bound']  = np.array(bound_tsbn_list)
            intermediate['valid_nll']  = np.array(nll_valid_list)
            if 'synthetic' in dkf.params['dataset']:
                mu_train, cov_train, mu_valid, cov_valid = _syntheticProc(dkf, dataset, dataset_eval)
                mu_list_train.append(mu_train)
                cov_list_train.append(cov_train)
                mu_list_valid.append(mu_valid)
                cov_list_valid.append(cov_valid)
                intermediate['mu_posterior_train']  = np.concatenate(mu_list_train, axis=2)
                intermediate['cov_posterior_train'] = np.concatenate(cov_list_train, axis=2)
                intermediate['mu_posterior_valid']  = np.concatenate(mu_list_valid, axis=2)
                intermediate['cov_posterior_valid'] = np.concatenate(cov_list_valid, axis=2)
            saveHDF5(savefile+'-EP'+str(epoch)+'-stats.h5', intermediate)
    #Final information to be collected
    retMap = {}
    retMap['train_bound']   = np.array(bound_train_list)
    retMap['valid_bound']   = np.array(bound_valid_list)
    retMap['tsbn_bound']   = np.array(bound_tsbn_list)
    retMap['valid_nll']  = np.array(nll_valid_list)
    if 'synthetic' in dkf.params['dataset']:
        retMap['mu_posterior_train']  = np.concatenate(mu_list_train, axis=2)
        retMap['cov_posterior_train'] = np.concatenate(cov_list_train, axis=2)
        retMap['mu_posterior_valid']  = np.concatenate(mu_list_valid, axis=2)
        retMap['cov_posterior_valid'] = np.concatenate(cov_list_valid, axis=2)
    return retMap

def _syntheticProc(dkf, dataset, dataset_eval):
    """
        Collect statistics on the synthetic dataset
    """
    allmus, alllogcov = [], []
    for s in range(100):
        _,mus, logcov = DKF_evaluate.infer(dkf,dataset)
        allmus.append(np.copy(mus))
        alllogcov.append(np.copy(logcov))
    allmus_v, alllogcov_v = [], []
    for s in range(100):
        _,mus, logcov = DKF_evaluate.infer(dkf,np.copy(dataset_eval))
        allmus_v.append(np.copy(mus))
        alllogcov_v.append(np.copy(logcov))

    mu_train = np.concatenate(allmus,axis=2).mean(2,keepdims=True)
    cov_train= np.exp(np.concatenate(alllogcov,axis=2)).mean(2,keepdims=True)
    mu_valid = np.concatenate(allmus_v,axis=2).mean(2,keepdims=True)
    cov_valid= np.exp(np.concatenate(alllogcov_v,axis=2)).mean(2,keepdims=True)
    return mu_train, cov_train, mu_valid, cov_valid
