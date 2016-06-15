
def infer(self, dataset):
    """
                            Posterior Inference using recognition network
    """
    assert len(dataset.shape)==3,'Expecting 3D tensor for data' 
    assert dataset.shape[2]==self.params['dim_observations'],'Data dim. not matching'
    eps  = np.random.randn(dataset.shape[0], dataset.shape[1], self.params['dim_stochastic']).astype(config.floatX)
    return self.posterior_inference(X=dataset.astype(config.floatX), eps=eps)

def evaluateBound(self, dataset, mask, batch_size, actions=None, S=2, normalization = 'frame', additional={}):
    """
                                    Evaluate bound on dataset
    """
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
        eps     = np.random.randn(X.shape[0],maxT,self.params['dim_stochastic']).astype(config.floatX)
        if actions is not None:
            U   = actions[st_idx:end_idx,:maxT,:].astype(config.floatX)
        
        maxS = S
        bound_sum, tsbn_bound_sum = 0, 0
        for s in range(S):
            if s%500==0:
                self._p('Done '+str(s))
            eps     = np.random.randn(X.shape[0],maxT,self.params['dim_stochastic']).astype(config.floatX)
            if actions is not None:
                batch_vec= self.evaluate(X=X, M=M, eps=eps, U=U)
            else:
                batch_vec= self.evaluate(X=X, M=M, eps=eps)
            if np.any(np.isnan(batch_vec)) or np.any(np.isinf(batch_vec)):
                self._p('NaN detected during evaluation. Ignoring this sample')
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
    self._p(('(Evaluate) Validation Bound: %.4f [Took %.4f seconds], TSBN Bound: %.4f')%(bound,end_time-start_time,tsbn_bound))
    additional['tsbn_bound'] = tsbn_bound
    return bound
