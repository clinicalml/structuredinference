
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
    return dkf.posterior_inference(dataset)

def evaluateBound(dkf, dataset, indicators, actions, mask, batch_size,S=2, normalization = 'frame', additional={}):
    """ Evaluate ELBO """
    bound = 0
    start_time = time.time()
    N = dataset.shape[0]
    tsbn_bound  = 0
    dkf.resetDataset(dataset,indicators, actions, mask)
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
    bound /= float(N)
    end_time   = time.time()
    dkf._p(('(Evaluate) Validation Bound: %.4f [Took %.4f seconds], TSBN Bound: %.4f')%(bound,end_time-start_time,tsbn_bound))
    additional['tsbn_bound'] = tsbn_bound
    return bound

def modelCfac(dkf, actions):
    N = actions.shape[0]
    z_init = np.random.randn(N, 1, dkf.params['dim_stochastic'])
    x_drug, z_drug     = sample(dkf, actions, z =  z_init)
    x_nodrug, z_nodrug = sample(dkf, np.zeros_like(actions), z = z_init) 
    modelCfac          = {}
    modelCfac['x_drug']= x_drug
    modelCfac['a_drug']= actions
    modelCfac['z_drug']= z_drug
    modelCfac['x_nodrug']= x_nodrug
    modelCfac['a_nodrug']= np.zeros_like(actions)
    modelCfac['z_nodrug']= z_nodrug
    return modelCfac

def getFirstIdx(act_data, drugIdx):
    act = act_data.squeeze()[:,drugIdx]
    assert act.ndim==1,'Expecting 1 d'
    idxlist = np.where(act==1.)[0]
    if len(idxlist)==0 or idxlist[0]<2 or idxlist[0]>len(act)-2:
        return -1
    else:
        return idxlist[0]

def dataCfac(dkf, dataset, actions, action_dict): 
    drugName = 'METFORMIN'
    drugIdx = [idx for idx,k in enumerate(action_dict) if drugName in k]
    if len(drugIdx)>1 or len(drugIdx)==0 :
        assert False,'Ambiguity!'
    else:
        drugIdx = drugIdx[0]
    dataCfac= {}
    for idx in range(dataset.shape[0]):
        pat_data = dataset[[idx],:,:] #1x18x48
        act_data = actions[[idx],:,:] #1x18x8
        #Conditions 
        #   Has a prescription of metformin + has some data (2 months) to do inference 
        st_idx = getFirstIdx(act_data, drugIdx) 
        if st_idx<0:
            continue
        #Do inference w/ the patient data up to st_idx
        _,mu,_        = dkf.posterior_inference(pat_data[:,:st_idx,:])
        remaining_act = act_data[:,st_idx:,:]
        x_drug, z_drug    = sample(dkf, remaining_act, z = np.copy(mu[:,[-1],:]))
        x_nodrug, z_nodrug= sample(dkf, np.zeros_like(remaining_act), z = np.copy(mu[:,[-1],:]))
        dataCfac[idx] = {}
        dataCfac[idx]['z_nodrug'] = z_nodrug
        dataCfac[idx]['x_nodrug'] = x_nodrug
        dataCfac[idx]['a_nodrug'] = np.zeros_like(remaining_act)
        dataCfac[idx]['z_drug']   = z_drug
        dataCfac[idx]['x_drug']   = x_drug
        dataCfac[idx]['a_drug']   = remaining_act
    print 'Processed: ',len(dataCfac),' out of ',dataset.shape[0],' patients for data based cfac inference'
    return dataCfac

def sample(dkf, actions, z = None):
    #Sample (fake patients) from the distribution
    #Since we don't have a model for actions, use the emperical distribution over actions, for each one
    #sample more than one patient to check the ability of the model to generalize. Use all 1s as indicators 
    N              = actions.shape[0]
    if z is None:
        z          = np.random.randn(actions.shape[0],1,dkf.params['dim_stochastic'])
    else:
        assert z.shape[0]==actions.shape[0], 'Check if we atleast have as many z as actions'
    all_z          = [np.copy(z)]
    for t in range(actions.shape[1]-1):
        #Use the transition means during sampling -Could vary this
        z,_        = dkf.transition_fxn(z= z, actions = actions[:,[t],:]) 
        all_z.append(np.copy(z))
    all_z    = np.concatenate(all_z,axis=1)
    x        = dkf.emission_fxn(all_z)
    return x, all_z

#How to store counterfactual results?
#Use pickle -> probably the best since it will be a bunch of hashmaps    
