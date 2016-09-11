"""
Functions for learning with a DKF object
"""
import evaluate as DKF_evaluate
import numpy as np
from utils.misc import saveHDF5
import time
from theano import config

def learn(dkf, dataset, indicators, actions, mask, epoch_start=0, epoch_end=1000, 
          batch_size=200, shuffle=True,
          savefreq=None, savefile = None, 
          dataset_eval = None, indicators_eval = None, actions_eval = None, mask_eval = None, 
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
    #Set data
    dkf.resetDataset(dataset, indicators, actions, mask)
    for epoch in range(epoch_start, epoch_end):
        #Shuffle
        if shuffle:
            np.random.shuffle(idxlist)
            batchlist = np.split(idxlist, range(batch_size,N,batch_size))
        np.random.shuffle(batchlist)
        start_time = time.time()
        bound = 0
        for bnum, batch_idx in enumerate(batchlist):
            batch_idx  = batchlist[bnum]
            batch_bound, p_norm, g_norm, opt_norm, negCLL, KL, anneal = dkf.train_debug(idx=batch_idx)
            M_sum = mask[batch_idx].sum()
            if replicate_K is not None:
                batch_bound, negCLL, KL = batch_bound/replicate_K, negCLL/replicate_K, KL/replicate_K, 
                M_sum   = M_sum/replicate_K
            bound  += batch_bound
            if bnum%10==0:
                bval = batch_bound/float(M_sum)
                dkf._p(('Bnum: %d, Batch Bound: %.4f, |w|: %.4f, |dw|: %.4f, |w_opt|: %.4f')%(bnum,bval,p_norm, g_norm, opt_norm)) 
                dkf._p(('-veCLL:%.4f, KL:%.4f, anneal:%.4f')%(negCLL, KL, anneal))
        bound /= float(mask.sum())
        bound_train_list.append((epoch,bound))
        end_time   = time.time()
        dkf._p(('(Ep %d) Bound: %.4f [Took %.4f seconds] ')%(epoch, bound, end_time-start_time))
        if savefreq is not None and epoch%savefreq==0:
            assert savefile is not None, 'expecting savefile'
            dkf._p(('Saving at epoch %d'%epoch))
            dkf._saveModel(fname = savefile+'-EP'+str(epoch))
            intermediate = {}
            if dataset_eval is not None and mask_eval is not None:
                tmpMap = {}
                bound_valid_list.append(
                    (epoch, 
                     DKF_evaluate.evaluateBound(dkf, dataset_eval, indicators_eval, actions_eval, mask_eval, batch_size=batch_size, 
                                              additional = tmpMap, normalization=normalization)))
                bound_tsbn_list.append((epoch, tmpMap['tsbn_bound']))
            intermediate['valid_bound'] = np.array(bound_valid_list)
            intermediate['train_bound'] = np.array(bound_train_list)
            intermediate['tsbn_bound']  = np.array(bound_tsbn_list)
            intermediate['samples']      = DKF_evaluate.sample(N=100,T=18)
            saveHDF5(savefile+'-EP'+str(epoch)+'-stats.h5', intermediate)
            dkf.resetDataset(dataset, indicators, actions, mask)
    #Final information to be collected
    retMap = {}
    retMap['train_bound']  = np.array(bound_train_list)
    retMap['valid_bound']  = np.array(bound_valid_list)
    retMap['tsbn_bound']   = np.array(bound_tsbn_list)
    retMap['samples']      = DKF_evaluate.sample(N=100,T=18)
    return retMap
