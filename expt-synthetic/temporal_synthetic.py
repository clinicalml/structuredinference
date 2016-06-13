from __future__ import division
import six.moves.cPickle as pickle
from collections import OrderedDict
import numpy as np
import sys, time, os, gzip, theano,math
from theano import config
theano.config.compute_test_value = 'warn'
from theano.printing import pydotprint
import theano.tensor as T
from utils.misc import saveHDF5
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from utils.optimizer import adam,rmsprop
from __init__ import BaseModel
from datasets.synthp import params_synthetic
from datasets.synthpTheano import updateParamsSynthetic


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
#"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
    
"""
                                                DEEP KALMAN FILTER
                                                
                                                Use for synthetic data
"""
class DKF(BaseModel, object):
    def __init__(self, params, paramFile=None, reloadFile=None):
        self.scan_updates = []
        super(DKF,self).__init__(params, paramFile=paramFile, reloadFile=reloadFile)
        if 'synthetic' in self.params['dataset'] and not hasattr(self, 'params_synthetic'):
            assert False, 'Expecting to have params_synthetic as an attribute in DKF class'
        assert self.params['nonlinearity']!='maxout','Maxout nonlinearity not supported'

    #"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""#
    def _fakeData(self):
        """
        Fake data for tag testing
        """
        T = 3
        N = 2
        mask = np.random.random((N,T)).astype(config.floatX)
        small = mask<0.5
        large = mask>=0.5
        mask[small] = 0.
        mask[large]= 1.
        eps  = np.random.randn(N,T,self.params['dim_stochastic']).astype(config.floatX)
        U = None
        if self.params['dim_actions']>0:
            U= np.random.random((N,T,self.params['dim_actions'])).astype(config.floatX)
        
        X    = np.random.randn(N,T,self.params['dim_observations']).astype(config.floatX)
        return X ,mask, eps, U
    
    #"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""#
    def _createParams(self):
        """
                                    Create parameters necessary for the model
        """
        npWeights = OrderedDict()
        self._createInferenceParams(npWeights)
        self._createGenerativeParams(npWeights)
        return npWeights
    
    def _createGenerativeParams(self, npWeights):
        """
                                    Create weights/params for generative model
        """
        assert 'synthetic' in self.params['dataset'],'Expecting synthetic data'
        if 'synthetic' in self.params['dataset']:
            updateParamsSynthetic(params_synthetic)
            self.params_synthetic = params_synthetic
            return

    def _createInferenceParams(self, npWeights):
        """
                                     Create weights/params for inference network
        """
        #Initial embedding for the inputs
        DIM_INPUT  = self.params['dim_observations']
        RNN_SIZE   = self.params['rnn_size']
        DIM_HIDDEN = RNN_SIZE
        DIM_STOC   = self.params['dim_stochastic']
        
        #Embed the Input -> RNN_SIZE
        dim_input, dim_output= DIM_INPUT, RNN_SIZE
        npWeights['q_W_input_0'] = self._getWeight((dim_input, dim_output))
        npWeights['q_b_input_0'] = self._getWeight((dim_output,))
        
        #Setup weights for LSTM
        self._createLSTMWeights(npWeights)
        
        #Embedding before MF/ST
        if self.params['inference_model']=='mean_field':
            for l in range(self.params['q_mlp_layers']):
                dim_input, dim_output = DIM_HIDDEN, DIM_HIDDEN
                if l==0:
                    dim_input = RNN_SIZE
                npWeights['q_W_mf_'+str(l)] = self._getWeight((dim_input, dim_output))
                npWeights['q_b_mf_'+str(l)] = self._getWeight((dim_output,))
            assert self.params['lr_combine']=='sum','Simple combination using the sum'      
        #### Parameters for ST-L, ST-R and ST-LR ####
        elif self.params['inference_model']=='structured':
            for l in range(self.params['q_mlp_layers']):
                dim_input, dim_output = DIM_HIDDEN, DIM_HIDDEN
                if l==0:
                    dim_input = RNN_SIZE
                if l==0:
                    npWeights['q_W_st_'+str(l)] = self._getWeight((dim_input+self.params['dim_stochastic'],
                                                                   dim_output))
                else:
                    npWeights['q_W_st_'+str(l)] = self._getWeight((dim_input, dim_output))
                npWeights['q_b_st_'+str(l)] = self._getWeight((dim_output,))
                
                npWeights['q_W_st_comb_'+str(l)]= self._getWeight((dim_input,dim_output))
                npWeights['q_b_st_comb_'+str(l)]= self._getWeight((dim_output,))
        else:
            assert False,'Invalid var model'
        DIM_HIDDEN = self.params['rnn_size']
        #For mean/cov
        npWeights['q_W_mu']       = self._getWeight((DIM_HIDDEN, self.params['dim_stochastic']))
        npWeights['q_b_mu']       = self._getWeight((self.params['dim_stochastic'],))
        npWeights['q_W_logsigsq'] = self._getWeight((DIM_HIDDEN, self.params['dim_stochastic']))
        npWeights['q_b_logsigsq'] = self._getWeight((self.params['dim_stochastic'],))
    
    def _createLSTMWeights(self, npWeights):
        #LSTM L/LR/R w/ orthogonal weight initialization
        suffices_to_build = []
        if self.params['var_model']=='lstmlr' or self.params['var_model']=='lstm':
            suffices_to_build.append('l')
        if self.params['var_model']=='lstmlr' or self.params['var_model']=='lstmr':
            suffices_to_build.append('r')
        RNN_SIZE = self.params['rnn_size']
        for suffix in suffices_to_build:
            for l in range(self.params['rnn_layers']):
                npWeights['W_lstm_'+suffix+'_'+str(l)] = self._getWeight((RNN_SIZE,RNN_SIZE*4))
                npWeights['b_lstm_'+suffix+'_'+str(l)] = self._getWeight((RNN_SIZE*4,), scheme='lstm')
                npWeights['U_lstm_'+suffix+'_'+str(l)] = self._getWeight((RNN_SIZE,RNN_SIZE*4),scheme='lstm')
        
    #"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""#
    def _getEmissionFxn(self, z):
        """
        Apply emission function to zs
        Input:  z [bs x T x dim]
        Output: (params, ) or (mu, logsigmasq) of size [bs x T x dim]
        """
        if 'synthetic' in self.params['dataset']:
            self._p('Using emission function for '+self.params['dataset'])
            mu = self.params_synthetic[self.params['dataset']]['obs_fxn'](z)
            logcov = T.log(T.ones_like(mu)*self.params_synthetic[self.params['dataset']]['obs_cov'])
            logcov.name = 'EmissionLogCov'
            return [mu,logcov]
    
    def _getTransitionFxn(self, z, u=None, fixedLogCov = None):
        """
        Apply transition function to zs
        Input:  z [bs x T x dim], u<if actions present in model> [bs x T x dim]
        Output: mu, logsigmasq of size [bs x T x dim]
        """
        if 'synthetic' in self.params['dataset']:
            self._p('Using transition function for '+self.params['dataset'])
            mu = self.params_synthetic[self.params['dataset']]['trans_fxn'](z)
            logcov = T.log(T.ones_like(mu)*self.params_synthetic[self.params['dataset']]['trans_cov'])
            logcov.name = 'TransitionLogCov'
            return mu,logcov
    
    #"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""#    
    def _buildLSTM(self, X, embedding, dropout_prob = 0.):
        """
        Take the embedding of bs x T x dim and return T x bs x dim that is the result of the scan operation
        for L/LR/R
        Input: embedding [bs x T x dim]
        Output:hidden_state [T x bs x dim]
        """
        start_time = time.time()
        self._p('In <_buildLSTM>')
        suffix = ''
        if self.params['var_model']=='lstmr':
            suffix='r'
            return self._LSTMlayer(embedding, suffix, dropout_prob)
        elif self.params['var_model']=='lstm':
            suffix='l'
            return self._LSTMlayer(embedding, suffix, dropout_prob)
        elif self.params['var_model']=='lstmlr':
            suffix='l'
            l2r   = self._LSTMlayer(embedding, suffix, dropout_prob)
            suffix='r'
            r2l   = self._LSTMlayer(embedding, suffix, dropout_prob)
            #Tried adding a linear here since l2r, r2l are coming from dropout, didn't help
            if self.params['lr_combine']=='sum':
                comb  = 0.5*(l2r+r2l)
            else:
                assert False,'Invalid value for lr_combine'
            return comb 
        else:
            assert False,'Invalid variational model'
        self._p(('Done <_buildLSTM> [Took %.4f]')%(time.time()-start_time))
        
    
    def _inferenceLayer(self, hidden_state, eps):
        """
        Take input of T x bs x dim and return z, mu, logsigsq each of size (bs x T x dim) 
        Input: hidden_state [T x bs x dim], eps [bs x T x dim]
        Output: z [bs x T x dim], mu [bs x T x dim], logsigsq [bs x T x dim]
        """
        if self.params['inference_model']=='structured':
            assert self.params['q_mlp_layers']==1,'Will not work for other sized layers'
        
        def _combinerFunction(h_t, eps_t, z_prev, 
                              q_W_st_0, q_b_st_0, #q_W_st_1,     q_b_st_1,
                              q_W_mu,   q_b_mu,   q_W_logsigsq, q_b_logsigsq):
            if self.params['dim_stochastic']==1:
                cat        = T.concatenate([h_t,z_prev[:,:-1]],axis=1)
            else:
                cat        = T.concatenate([h_t,z_prev],axis=1)
                
            hid2       = self._LinearNL(q_W_st_0,q_b_st_0,cat) 
            #hid2       = self._LinearNL(q_W_st_1,q_b_st_1,hid1) 
            mu         = T.dot(hid2,q_W_mu)+q_b_mu
            logsigmasq = T.dot(hid2,q_W_logsigsq)+q_b_logsigsq
            if self.params['cov_explicit']:
                cov    = T.nnet.softplus(logsigmasq)
                z          = theano.gradient.grad_clip(mu + T.sqrt(cov)*eps_t, -10,10)
                if self.params['dim_stochastic']==1:
                    z = T.concatenate([z,T.alloc(np.asarray(0.,dtype=config.floatX), z.shape[0], 1)],axis=1)
                    z.name = 'z_next'
                return z, mu, cov
            else:
                #Add a grad clip here so that gradients with respect to input here are bounded
                z          = theano.gradient.grad_clip(mu + T.exp(0.5*logsigmasq)*eps_t, -10,10)
                if self.params['dim_stochastic']==1:
                    z = T.concatenate([z,T.alloc(np.asarray(0.,dtype=config.floatX), z.shape[0], 1)],axis=1)
                    z.name = 'z_next'
                return z, mu, logsigmasq
        if self.params['inference_model']=='mean_field':
            if type(hidden_state) is list:
                #If we're using separate paths for l2r and r2l
                assert self.params['var_model']=='lstmlr' and self.params['lr_combine']=='split','expecting sp. l->r and r->l'
                l2r = hidden_state[0].swapaxes(0,1)
                r2l = hidden_state[1].swapaxes(0,1)
                hidl2r = l2r
                for l in range(self.params['q_mlp_layers']):
                    hidl2r = self._LinearNL(self.tWeights['q_W_mf_'+str(l)],self.tWeights['q_b_mf_'+str(l)],hidl2r)
                mu_1     = T.dot(hidl2r,self.tWeights['q_W_mu'])+self.tWeights['q_b_mu']
                logcov_1 = T.dot(hidl2r,self.tWeights['q_W_logsigsq'])+self.tWeights['q_b_logsigsq']
                hidr2l = r2l
                for l in range(self.params['q_mlp_layers']):
                    hidr2l = self._LinearNL(self.tWeights['q_W_mf_r_'+str(l)],self.tWeights['q_b_mf_r_'+str(l)],hidr2l)
                mu_2     = T.dot(hidr2l,self.tWeights['q_W_mu_r'])+self.tWeights['q_b_mu_r']
                logcov_2 = T.dot(hidr2l,self.tWeights['q_W_logsigsq_r'])+self.tWeights['q_b_logsigsq_r']
                
                if self.params['cov_explicit']:
                    cov_1    = T.nnet.softplus(logcov_1)
                    cov_2    = T.nnet.softplus(logcov_2)
                    if self.params['lr_join']=='pog':
                        #Product of Gaussians - Unnormalized, this should be OK for the posterior approx.
                        #since we only use the posterior means and variances for the KL divergence
                        mu     = (mu_1*cov_2+mu_2*cov_1)/(cov_1+cov_2)
                        cov    = (cov_1*cov_2)/(cov_1+cov_2)
                    elif self.params['lr_join']=='dvae':
                        #w/out softplus combine mu and logcov, Weirdly reformulated version of POG
                        mu     = ((mu_1/cov_1)+(mu_2/cov_2))/(1./cov_1 + 1./cov_2)
                        cov    = 1./(1./cov_1 + 1./cov_2)
                    elif self.params['lr_join']=='convolve':
                        mu     = mu_1+mu_2
                        cov    = cov_1+cov_2
                    else:
                        assert False,'Invalid. specify lr_join'
                    z = mu + T.sqrt(cov)*eps
                    return z,mu,cov
                else:
                    if self.params['lr_join']=='pog':
                        #Product of Gaussians - unnormalized, this should be OK for the posterior approx.
                        #since we only use the posterior means and variances for the KL divergence
                        K      = T.exp(logcov_1)+T.exp(logcov_2)
                        mu     = (mu_1*T.exp(logcov_2)+mu_2*T.exp(logcov_1))/K
                        logcov = (logcov_1+logcov_2)-T.log(K)
                    elif self.params['lr_join']=='dvae':
                        #w/out softplus combine mu and logcov
                        K      = T.exp(-logcov_1)+T.exp(-logcov_2)
                        mu     = (mu_1*T.exp(-logcov_1)+mu_2*T.exp(-logcov_2))/K
                        logcov = -T.log(K)
                    elif self.params['lr_join']=='convolve':
                        mu     = mu_1+mu_2
                        K      = T.exp(logcov_1)+T.exp(logcov_2)
                        logcov = T.log(K)
                    else:
                        assert False,'Invalid. specify lr_join'
                    z = mu + T.exp(0.5*logcov)*eps
                    return z,mu,logcov
            
            hid     = hidden_state.swapaxes(0,1)
            for l in range(self.params['q_mlp_layers']):
                hid = self._LinearNL(self.tWeights['q_W_mf_'+str(l)],self.tWeights['q_b_mf_'+str(l)],hid)
            mu        = T.dot(hid,self.tWeights['q_W_mu'])+self.tWeights['q_b_mu']
            logcov    = T.dot(hid,self.tWeights['q_W_logsigsq'])+ self.tWeights['q_b_logsigsq']
            
            if self.params['cov_explicit']:
                cov       = T.nnet.softplus(logcov)
                z         = mu + T.sqrt(cov)*eps
                return z, mu, cov 
            else:
                z         = mu + T.exp(0.5*logcov)*eps
                return z, mu, logcov
            
        elif self.params['inference_model']=='structured':
            n_samples= eps.shape[0]
            nsteps   = eps.shape[1]
            eps_swap = eps.swapaxes(0,1)
            toadd = 0
            if self.params['dim_stochastic']==1:
                toadd = 1
                
            for l in range(self.params['q_mlp_layers']):
                hidden_state = self._LinearNL(self.tWeights['q_W_st_comb_'+str(l)], 
                                              self.tWeights['q_b_st_comb_'+str(l)],hidden_state)
            rval, _ = theano.scan(_combinerFunction, 
                                    sequences=[hidden_state, eps_swap],
                                    outputs_info=[
                                    T.alloc(np.asarray(0.,dtype=config.floatX), 
                                            n_samples, self.params['dim_stochastic']+toadd),
                                    None,None],
                                    non_sequences=[self.tWeights[k] for k in 
                                                   ['q_W_st_0', 'q_b_st_0']]+ 
                                                  [self.tWeights[k] for k in 
                                                   ['q_W_mu','q_b_mu','q_W_logsigsq','q_b_logsigsq']],
                                    name='combinerFxn', 
                                    n_steps=nsteps,
                                    go_backwards=False)
            z, mu, logcov_or_cov = rval[0].swapaxes(0,1), rval[1].swapaxes(0,1), rval[2].swapaxes(0,1)
            if self.params['dim_stochastic']==1:
                z = z[:,:,:-1]
            return z, mu, logcov_or_cov
        else:
            assert False,'Invalid inference model' 

    def _qEmbeddingLayer(self, X):
        """
        Take input X and pass it through embedding function in q to reduce dimensionality
        """
        return self._LinearNL(self.tWeights['q_W_input_0'],self.tWeights['q_b_input_0'], X)
    
    #"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""# 
    def _inferenceAndReconstruction(self, X, eps, U=None, dropout_prob = 0.):
        """
        Returns z_q, mu_q and logcov_q
        """
        self._p('Building with dropout:'+str(dropout_prob))
        embedding         = self._qEmbeddingLayer(X)
        hidden_state      = self._buildLSTM(X, embedding, dropout_prob)
        z_q,mu_q,cov_logcov_q = self._inferenceLayer(hidden_state, eps)
        z_q.name          = 'z_q'
        
        observation_params = self._getEmissionFxn(z_q)
        mu_trans,logcov_trans = self._getTransitionFxn(z_q, u=U)
        prior_mu_val, prior_logcov_val = 0.,0.
        if 'synthetic' in self.params['dataset']:
            prior_mu_val     = float(params_synthetic[self.params['dataset']]['init_mu'])
            prior_logcov_val = np.log(float(params_synthetic[self.params['dataset']]['init_cov']))
            self._p(('Setting initial mean/logcov to (%.4f, %.4f)')%(prior_mu_val, prior_logcov_val))
        mu_prior     = T.concatenate([T.alloc(np.asarray(0.+prior_mu_val).astype(config.floatX),
                                              X.shape[0],1,self.params['dim_stochastic']),
                       mu_trans[:,:-1,:]],axis=1)
        logcov_prior = T.concatenate([T.alloc(np.asarray(0.+prior_logcov_val).astype(config.floatX),
                                              X.shape[0],1,self.params['dim_stochastic']),
                       logcov_trans[:,:-1,:]],axis=1)
        return observation_params, z_q, mu_q, cov_logcov_q, mu_prior, logcov_prior, mu_trans, logcov_trans
    
    def _getTemporalKL(self, mu_q, cov_logcov_q, mu_prior, cov_logcov_prior, M, batchVector = False):
        """
        TemporalKL divergence KL (q||p)
        
        KL(q_t||p_t) = 0.5*(log|sigmasq_p| -log|sigmasq_q|  -D + Tr(sigmasq_p^-1 sigmasq_q) 
                        + (mu_p-mu_q)^T sigmasq_p^-1 (mu_p-mu_q))
        M is a mask of size bs x T that should be applied once the KL divergence for each point
        across time has been estimated
        """
        if self.params['cov_explicit']:
            assert np.all(cov_logcov_q.tag.test_value>0.),'should be positive'
            assert np.all(cov_logcov_prior.tag.test_value>0.),'should be positive'
            diff_mu = mu_prior-mu_q
            #ASK URI ABOUT A NICER DECOMPOSITION
            KL_t    = T.log(cov_logcov_prior)-T.log(cov_logcov_q) - 1. + cov_logcov_q/cov_logcov_prior + diff_mu**2/cov_logcov_prior
            KLvec   = (0.5*KL_t.sum(2)*M).sum(1,keepdims=True)
        else:
            diff_mu = mu_prior-mu_q
            KL_t    = cov_logcov_prior - cov_logcov_q - 1. + T.exp(cov_logcov_q-cov_logcov_prior) \
                    + diff_mu*T.exp(-1*cov_logcov_prior)*diff_mu
            KLvec   = (0.5*KL_t.sum(2)*M).sum(1,keepdims=True)
        if batchVector:
            return KLvec
        else:
            return KLvec.sum()
    
    def _getNegCLL(self, obs_params, X, M, batchVector = False):
        """
        Estimate the negative conditional log likelihood of x|z under the generative model
        M: mask of size bs x T
        X: target of size bs x T x dim
        """
        #If its a NADE, then we would have computed the mean probs
        if self.params['emission_type']=='nade':
            #we would have gotten the hidden state and not the mean parameters. compute here
            assert False,'Not implemented'
            
        if self.params['data_type']=='binary':
            mean_p = obs_params[0]
            negCLL = (T.nnet.binary_crossentropy(mean_p,X).sum(2)*M).sum(1,keepdims=True)
        elif self.params['data_type']=='real':
            mu_p   = obs_params[0]
            logcov_p=obs_params[1]
            negCLL_t  = 0.5 * np.log(2 * np.pi) + 0.5*logcov_p + 0.5 * ((X - mu_p) / T.exp(0.5*logcov_p))**2
            negCLL    = (negCLL_t.sum(2)*M).sum(1,keepdims=True)
        elif self.params['data_type']=='handwriting':
            #In this case, dim_observations is set to be the number of parameters required
            mu_x, mu_y, sig_x, sig_y, mix_probs, corr, indicator_p = obs_params
            negCLL_m  = self._NLL_2MDN(X, mu_x, mu_y, sig_x, sig_y, mix_probs, corr, indicator_p)
            negCLL    = (negCLL_m*M).sum(1,keepdims=True)
        else:
            assert False,'Invalid data type: '+self.params['data_type']
        if batchVector:
            return negCLL
        else:
            return negCLL.sum()
        
    
    def _buildModel(self):
        """
        High level function to build and setup theano functions
        """
        X      = T.tensor3('X',   dtype=config.floatX)
        eps    = T.tensor3('eps', dtype=config.floatX)
        M      = T.matrix('M', dtype=config.floatX)
        U      = None
        if self.params['dim_actions']>0:
            U  = T.tensor3('U',   dtype=config.floatX)
        X.tag.test_value, M.tag.test_value, eps.tag.test_value, u_tag   = self._fakeData()
        if U is not None:
            U.tag.test_value = u_tag
        
        #Learning Rates and annealing objective function
        #Add them to npWeights/tWeights to be tracked [do not have a prefix _W or _b so wont be diff.]
        self._addWeights('lr', np.asarray(self.params['lr'],dtype=config.floatX),borrow=False)
        self._addWeights('anneal', np.asarray(0.01,dtype=config.floatX),borrow=False)
        self._addWeights('update_ctr', np.asarray(1.,dtype=config.floatX),borrow=False)
        
        lr             = self.tWeights['lr']
        anneal         = self.tWeights['anneal']
        iteration_t    = self.tWeights['update_ctr'] 
        
        lr_update = [(lr,lr)]#T.switch(lr/1.0001<1e-5,lr,lr/1.0001))]                          
        anneal_div     = 1000.
        if 'anneal_rate' in self.params:
            self._p('Anneal = 1 in '+str(self.params['anneal_rate'])+' param. updates')
            anneal_div = self.params['anneal_rate']
        if 'synthetic' in self.params['dataset']:
            anneal_div = 100.
        anneal_update  = [(iteration_t, iteration_t+1),
                          (anneal,T.switch(0.01+iteration_t/anneal_div>1,1,0.01+iteration_t/anneal_div))]
        
        if not self.params['validate_only']:
            ############# Setup training functions ###########
            obs_params, z_q, mu_q, cov_logcov_q, mu_prior, cov_logcov_prior, _, _ = self._inferenceAndReconstruction( 
                                                              X, eps, U= U,
                                                              dropout_prob = self.params['rnn_dropout'])
            negCLL = self._getNegCLL(obs_params, X, M)
            TemporalKL = self._getTemporalKL(mu_q, cov_logcov_q, mu_prior, cov_logcov_prior, M)
            train_cost = negCLL+anneal*TemporalKL

            #Get updates from optimizer
            model_params         = self._getModelParams()
            optimizer_up, norm_list  = self._setupOptimizer(train_cost, model_params,lr = lr, 
                                                            reg_type =self.params['reg_type'], 
                                                            reg_spec =self.params['reg_spec'], 
                                                            reg_value= self.params['reg_value'],
                                                           grad_norm = 1.,
                                                           divide_grad = M.sum())
            #Add annealing updates
            optimizer_up +=anneal_update
            self.decay_lr   = theano.function([],lr.sum(),name = 'Update LR',updates=lr_update)

            ############# Setup train & evaluate functions ###########
            fxn_inputs = [X, M, eps]
            if U is not None:
                fxn_inputs.append(U)
            self.train_debug         = theano.function(fxn_inputs,[train_cost,norm_list[0],norm_list[1],
                                                                        norm_list[2],negCLL, TemporalKL, anneal.sum()], 
                                                           updates = optimizer_up, name='Train (with Debug)')
        
        eval_obs_params, eval_z_q, eval_mu_q, eval_cov_logcov_q, eval_mu_prior, eval_cov_logcov_prior, \
        eval_mu_trans, eval_cov_logcov_trans = self._inferenceAndReconstruction(
                                                          X, eps, U= U,
                                                          dropout_prob = 0.)
        eval_z_q.name = 'eval_z_q'
        
        eval_CNLLvec=self._getNegCLL(eval_obs_params, X, M, batchVector = True)
        eval_KLvec  = self._getTemporalKL(eval_mu_q, eval_cov_logcov_q,eval_mu_prior, 
                                          eval_cov_logcov_prior, M, batchVector = True)
        eval_cost   = eval_CNLLvec + eval_KLvec
        
        #From here on, convert to the log covariance since we only use it for evaluation
        if self.params['cov_explicit']:
            assert np.all(eval_cov_logcov_q.tag.test_value>0.),'should be positive'
            assert np.all(eval_cov_logcov_prior.tag.test_value>0.),'should be positive'
            assert np.all(eval_cov_logcov_trans.tag.test_value>0.),'should be positive'
            #convert to log
            eval_logcov_q     = T.log(eval_cov_logcov_q)
            eval_logcov_prior = T.log(eval_cov_logcov_prior)
            eval_logcov_trans = T.log(eval_cov_logcov_trans)
        else:
            eval_logcov_q     =eval_cov_logcov_q
            eval_logcov_prior = eval_cov_logcov_prior
            eval_logcov_trans = eval_cov_logcov_trans
            eval_logcov_q     = eval_logcov_q
        
        ll_prior     = self._llGaussian(eval_z_q, eval_mu_prior, eval_logcov_prior).sum(2)*M
        ll_posterior = self._llGaussian(eval_z_q, eval_mu_q, eval_logcov_q).sum(2)*M
        ll_estimate  = -1*eval_CNLLvec+ll_prior.sum(1,keepdims=True)-ll_posterior.sum(1,keepdims=True)
        
        eval_inputs = [eval_z_q]
        if U is not None:
            eval_inputs.append(U)
        
        self.likelihood          = theano.function(fxn_inputs, ll_estimate, name = 'Importance Sampling based likelihood')
        self.evaluate            = theano.function(fxn_inputs, eval_cost, name = 'Evaluate Bound')
        self.transition_fxn      = theano.function(eval_inputs,[eval_mu_trans, eval_logcov_trans],
                                                       name='Transition Function')
        self.emission_fxn = theano.function([eval_z_q], eval_obs_params, name='Emission Function')
        self.posterior_inference = theano.function([X, eps], [eval_z_q, eval_mu_q, eval_logcov_q],name='Posterior Inference') 
        
        #Add in trigger for warm start
        #DO NOT CALL THIS FUNCTION MULTIPLE TIMES IT WILL REPEATEDLY SET parameters
        if self.params['var_model']=='lstmlr' and self.params['lr_combine']=='warm_start':
            updates = []
            updates.append((self.tWeights['warmstartLR'],0.5))
            diff    = 0
            for k in self.npWeights:
                if 'lstm_r' in k:
                    self._p('Replacing (%s) with (%s)'%(k,k.replace('_r_','_l_')))
                    updates.append((self.tWeights[k],self.tWeights[k.replace('_r_','_l_')]))
                    diff += T.sum((self.tWeights[k]-self.tWeights[k.replace('_r_','_l_')])**2)
            self.warm_start_lr   = theano.function([],self.tWeights['warmstartLR'].sum(),updates = updates)
            self.probe           = theano.function([],[T.sqrt(diff),self.tWeights['warmstartLR'].sum()])
    #"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""# 
    def sampleGaussian(self,mu,logsigmasq):
        return mu + np.random.randn(*mu.shape)*np.exp(0.5*logsigmasq)
    """
                                      Sample from Generative Model
    """
    def sample(self, nsamples=100, T=10, U= None):
        if self.params['dim_actions']>0:
            assert U is not None,'Specify U for sampling model conditioned on actions'
        assert T>1, 'Sample atleast 2 timesteps'
        #Initial sample
        z      = np.random.randn(nsamples,1,self.params['dim_stochastic']).astype(config.floatX)
        all_zs = [np.copy(z)]
        for t in range(T-1):
            mu,logcov = self.transition_fxn(z)
            z           = self.sampleGaussian(mu,logcov).astype(config.floatX)
            all_zs.append(np.copy(z))
        zvec = np.concatenate(all_zs,axis=1)
        return self.emission_fxn(zvec), zvec
    
    def infer(self, dataset):
        """
                                Posterior Inference using recognition network
        """
        assert len(dataset.shape)==3,'Expecting 3D tensor for data' 
        assert dataset.shape[2]==self.params['dim_observations'],'Data dim. not matching'
        eps  = np.random.randn(dataset.shape[0], dataset.shape[1], self.params['dim_stochastic']).astype(config.floatX)
        return self.posterior_inference(X=dataset.astype(config.floatX), eps=eps)
    
    
    def evaluateBound(self, dataset, mask, batch_size, actions=None, S=5, normalization = 'frame', additional={}):
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
    
    def learn(self, dataset, mask, actions = None, epoch_start=0, epoch_end=1000, 
              batch_size=200, shuffle=False,
              savefreq=None, savefile = None, 
              dataset_eval = None, mask_eval = None, actions_eval = None,
              replicate_K = None,
              normalization = 'frame'):
        """
                                                Train DKF
        """
        assert not self.params['validate_only'],'cannot learn in validate only mode'
        assert len(dataset.shape)==3,'Expecting 3D tensor for data'
        assert dataset.shape[2]==self.params['dim_observations'],'Dim observations not valid'
        N = dataset.shape[0]
        idxlist = range(N)
        bound_train_list,bound_valid_list,bound_tsbn_list = [],[],[]
        p_norm, g_norm, opt_norm = None, None, None

        #Lists used to track quantities
        mu_list_train, cov_list_train, mu_list_valid, cov_list_valid, nll_valid_list = [],[],[],[],[]
        current_lr = self.params['lr']
        
        #Start of training loop
        for epoch in range(epoch_start, epoch_end):
            if shuffle:
                np.random.shuffle(idxlist)
            start_time = time.time()
            bound = 0
            current_lr = self.decay_lr()
            
            if self.params['var_model']=='lstmlr' and self.params['lr_combine']=='warm_start' and epoch==1000:
                self._p('<Starting update of LSTM-LR>')
                self.warm_start_lr()
            
            #Going through the dataset in batches
            for bnum,st_idx in enumerate(range(0,N,batch_size)):
                end_idx = min(st_idx+batch_size, N)
                X       = dataset[idxlist[st_idx:end_idx],:,:].astype(config.floatX)
                M       = mask[idxlist[st_idx:end_idx],:].astype(config.floatX)
                U       = None
                
                #Reduce the dimensionality of the tensors based on the maximum size of the mask
                maxT    = int(np.max(M.sum(1)))
                X       = X[:,:maxT,:]
                M       = M[:,:maxT]
                
                if replicate_K is not None:
                    X   = np.tile(X,(replicate_K, 1, 1))
                    M   = np.tile(M,(replicate_K, 1))
                    if U is not None:
                        U = np.tile(U,(replicate_K, 1, 1))
                
                eps     = np.random.randn(X.shape[0],maxT,self.params['dim_stochastic']).astype(config.floatX)
                
                if actions is not None:
                    U   = actions[idxlist[st_idx:end_idx],:maxT,:].astype(config.floatX)
                    batch_bound = self.train(X=X, M=M, eps=eps, U=U)
                else:
                    batch_bound, p_norm, g_norm, opt_norm, negCLL, KL, anneal = self.train_debug(X=X, M=M, eps=eps)
                
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
                    if p_norm is not None:
                        self._p(('Bnum:%d, Batch Bound: %.4f, |w|: %.4f, |dw|: %.4f,|w_opt|:%.4f, -veCLL:%.4f, KL:%.4f: , anneal:%.4f')% (bnum,bval,p_norm, g_norm, opt_norm, negCLL, KL, anneal)) 
                    else:
                        self._p(('Bnum:%d, Batch Bound: %.4f')%(bnum,bval)) 
            
            if normalization=='frame':
                bound /= float(mask.sum())
            elif normalization=='sequence':
                bound /= float(N)
            else:
                assert False,'Invalid normalization'
            bound_train_list.append((epoch,bound))
            end_time   = time.time()
            self._p(('(Ep %d) Bound: %.4f , LR: %.4e[Took %.4f seconds] ')%(epoch, bound,current_lr, end_time-start_time))
            
            #Save at intermediate stages
            if savefreq is not None and epoch%savefreq==0:
                assert savefile is not None, 'expecting savefile'
                self._p(('Saving at epoch %d'%epoch))
                self._saveModel(fname = savefile+'-EP'+str(epoch))
                
                intermediate = {}
                if dataset_eval is not None and mask_eval is not None:
                    tmpMap = {}
                    #Track the validation bound and the TSBN bound
                    bound_valid_list.append(
                        (epoch,self.evaluateBound(dataset_eval, mask_eval, 
                                                  actions=actions_eval, batch_size=batch_size, 
                                                  additional = tmpMap, normalization=normalization)))
                    bound_tsbn_list.append((epoch, tmpMap['tsbn_bound']))
                    nll_valid_list.append(self.impSamplingNLL(dataset_eval, mask_eval, batch_size, 
                                                              actions=actions_eval, normalization=normalization))
                    
                    if 'synthetic' in self.params['dataset']:
                        mu_train, cov_train, mu_valid, cov_valid = self._syntheticProc(dataset, dataset_eval)
                        mu_list_train.append(mu_train)
                        cov_list_train.append(cov_train)
                        mu_list_valid.append(mu_valid)
                        cov_list_valid.append(cov_valid)
                        intermediate['mu_posterior_train']  = np.concatenate(mu_list_train, axis=2)
                        intermediate['cov_posterior_train'] = np.concatenate(cov_list_train, axis=2)
                        intermediate['mu_posterior_valid']  = np.concatenate(mu_list_valid, axis=2)
                        intermediate['cov_posterior_valid'] = np.concatenate(cov_list_valid, axis=2)
                intermediate['valid_bound'] = np.array(bound_valid_list)
                intermediate['train_bound'] = np.array(bound_train_list)
                intermediate['tsbn_bound']  = np.array(bound_tsbn_list)
                intermediate['valid_nll']  = np.array(nll_valid_list)
                saveHDF5(savefile+'-EP'+str(epoch)+'-stats.h5', intermediate)
        
        #Final information to be collected
        retMap = {}
        retMap['train_bound']   = np.array(bound_train_list)
        retMap['valid_bound']   = np.array(bound_valid_list)
        retMap['tsbn_bound']   = np.array(bound_tsbn_list)
        retMap['valid_nll']  = np.array(nll_valid_list)
        if 'synthetic' in self.params['dataset']:
            retMap['mu_posterior_train']  = np.concatenate(mu_list_train, axis=2)
            retMap['cov_posterior_train'] = np.concatenate(cov_list_train, axis=2)
            retMap['mu_posterior_valid']  = np.concatenate(mu_list_valid, axis=2)
            retMap['cov_posterior_valid'] = np.concatenate(cov_list_valid, axis=2)
        return retMap
    
    def impSamplingNLL(self, dataset, mask, batch_size, actions=None, S = 5, normalization = 'frame'):
        """
                                    Importance sampling based log likelihood
        """
        ll = 0
        start_time = time.time()
        N = dataset.shape[0]
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
            lllist = []
            for s in range(S):
                eps     = np.random.randn(X.shape[0],maxT,self.params['dim_stochastic']).astype(config.floatX)
                if U is not None:
                    batch_vec = self.likelihood(X=X, M=M, eps=eps, U=U)
                else:
                    batch_vec = self.likelihood(X=X, M=M, eps=eps)
                if np.any(np.isnan(batch_vec)) or np.any(np.isinf(batch_vec)):
                    self._p('NaN detected during evaluation. Ignoring this sample')
                    maxS -=1
                    continue
                else:
                    lllist.append(batch_vec)
            ll  += self.meanSumExp(np.concatenate(lllist,axis=1), axis=1).sum()
        if normalization=='frame':
            ll /= float(mask.sum())
        elif normalization=='sequence':
            ll /= float(N)
        else:
            assert False,'Invalid normalization specified'
            
        end_time   = time.time()
        self._p(('(Evaluate w/ Imp. Sampling) Validation LL: %.4f [Took %.4f seconds]')%(ll,end_time-start_time))
        return ll
    
    def _syntheticProc(self, dataset, dataset_eval):
        #Estimate reconstruction on the training set
        allmus, alllogcov = [], []
        for s in range(100):
            _,mus, logcov = self.infer(dataset)
            allmus.append(np.copy(mus))
            alllogcov.append(np.copy(logcov))
        #Processing for synthetic data
        allmus_v, alllogcov_v = [], []
        for s in range(100):
            _,mus, logcov = self.infer(np.copy(dataset_eval))
            allmus_v.append(np.copy(mus))
            alllogcov_v.append(np.copy(logcov))
        
        mu_train = np.concatenate(allmus,axis=2).mean(2,keepdims=True)
        cov_train= np.exp(np.concatenate(alllogcov,axis=2)).mean(2,keepdims=True)
        mu_valid = np.concatenate(allmus_v,axis=2).mean(2,keepdims=True)
        cov_valid= np.exp(np.concatenate(alllogcov_v,axis=2)).mean(2,keepdims=True)
        return mu_train, cov_train, mu_valid, cov_valid

