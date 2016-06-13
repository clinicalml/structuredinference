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
from models.__init__ import BaseModel
from datasets.synthp import params_synthetic
from datasets.synthpTheano import updateParamsSynthetic


    
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
#"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""#
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#
    
"""
                                                DEEP KALMAN FILTER
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
        #Transition Emission
        DIM_HIDDEN = self.params['dim_hidden']
        DIM_STOCHASTIC = self.params['dim_stochastic']
        
        if self.params['transition_type']=='mlp':
            DIM_HIDDEN_TRANS = DIM_HIDDEN*2
            for l in range(self.params['transition_layers']):
                dim_input,dim_output = DIM_HIDDEN_TRANS, DIM_HIDDEN_TRANS
                if l==0:
                    dim_input = self.params['dim_stochastic']
                npWeights['p_trans_W_'+str(l)] = self._getWeight((dim_input, dim_output))
                npWeights['p_trans_b_'+str(l)] = self._getWeight((dim_output,))
            if self.params['use_prev_input']:
                npWeights['p_trans_W_0'] = self._getWeight((DIM_STOCHASTIC+self.params['dim_observations'],
                                                                DIM_HIDDEN_TRANS))
                npWeights['p_trans_b_0'] = self._getWeight((DIM_HIDDEN_TRANS,))
            MU_COV_INP = DIM_HIDDEN_TRANS
        elif self.params['transition_type']=='simple_gated':
            DIM_HIDDEN_TRANS = DIM_HIDDEN*2
            npWeights['p_gate_embed_W_0'] = self._getWeight((DIM_STOCHASTIC, DIM_HIDDEN_TRANS))
            npWeights['p_gate_embed_b_0'] = self._getWeight((DIM_HIDDEN_TRANS,))
            npWeights['p_gate_embed_W_1'] = self._getWeight((DIM_HIDDEN_TRANS, DIM_STOCHASTIC))
            npWeights['p_gate_embed_b_1'] = self._getWeight((DIM_STOCHASTIC,))
            npWeights['p_z_W_0'] = self._getWeight((DIM_STOCHASTIC, DIM_HIDDEN_TRANS))
            npWeights['p_z_b_0'] = self._getWeight((DIM_HIDDEN_TRANS,))
            npWeights['p_z_W_1'] = self._getWeight((DIM_HIDDEN_TRANS, DIM_STOCHASTIC))
            npWeights['p_z_b_1'] = self._getWeight((DIM_STOCHASTIC,))
            if self.params['use_prev_input']:
                npWeights['p_z_W_0'] = self._getWeight((DIM_STOCHASTIC+self.params['dim_observations'], DIM_HIDDEN_TRANS))
                npWeights['p_z_b_0'] = self._getWeight((DIM_HIDDEN_TRANS,))
                npWeights['p_gate_embed_W_0'] = self._getWeight((DIM_STOCHASTIC+self.params['dim_observations'],
                                                                DIM_HIDDEN_TRANS))
                npWeights['p_gate_embed_b_0'] = self._getWeight((DIM_HIDDEN_TRANS,))
            MU_COV_INP = DIM_STOCHASTIC
        else:
            assert False,'Invalid transition type: '+self.params['transition_type']
        
        if self.params['transition_type']=='simple_gated':
            weight= np.eye(self.params['dim_stochastic']).astype(config.floatX)
            bias  = np.zeros((self.params['dim_stochastic'],)).astype(config.floatX)
            npWeights['p_trans_W_mu'] = weight
            npWeights['p_trans_b_mu'] = bias
        else:
            npWeights['p_trans_W_mu']       = self._getWeight((MU_COV_INP, self.params['dim_stochastic']))
            npWeights['p_trans_b_mu']       = self._getWeight((self.params['dim_stochastic'],))
        npWeights['p_trans_W_cov'] = self._getWeight((MU_COV_INP, self.params['dim_stochastic']))
        npWeights['p_trans_b_cov'] = self._getWeight((self.params['dim_stochastic'],))
        
        
        #Emission Function [MLP]
        if self.params['emission_type'] == 'mlp':
            for l in range(self.params['emission_layers']):
                dim_input,dim_output = DIM_HIDDEN, DIM_HIDDEN
                if l==0:
                    dim_input = self.params['dim_stochastic']
                npWeights['p_emis_W_'+str(l)] = self._getWeight((dim_input, dim_output))
                npWeights['p_emis_b_'+str(l)] = self._getWeight((dim_output,))
        elif self.params['emission_type'] =='conditional':
            for l in range(self.params['emission_layers']):
                dim_input,dim_output = DIM_HIDDEN, DIM_HIDDEN
                if l==0:
                    dim_input = self.params['dim_stochastic']+self.params['dim_observations']
                npWeights['p_emis_W_'+str(l)] = self._getWeight((dim_input, dim_output))
                npWeights['p_emis_b_'+str(l)] = self._getWeight((dim_output,))
        elif self.params['emission_type'] =='conditional_linear':
            dim_input,dim_output = self.params['dim_stochastic'], self.params['dim_observations']
            npWeights['p_emis_W_0'] = self._getWeight((dim_input, dim_output))
            npWeights['p_emis_b_0'] = self._getWeight((dim_output,))
            dim_input = self.params['dim_observations']
            npWeights['p_inp_W'] = self._getWeight((dim_input, dim_input))
            npWeights['p_inp_b'] = self._getWeight((dim_input,))
        else:
            assert False, 'Invalid emission type: '+str(self.params['emission_type'])
        
        if self.params['data_type']=='binary':
            npWeights['p_emis_W_ber'] = self._getWeight((self.params['dim_hidden'], self.params['dim_observations']))
            npWeights['p_emis_b_ber'] = self._getWeight((self.params['dim_observations'],))
        elif self.params['data_type']=='binary_nade':
            n_visible, n_hidden   = self.params['dim_observations'], self.params['dim_hidden']
            npWeights['p_nade_W'] = self._getWeight((n_visible, n_hidden))
            npWeights['p_nade_U'] = self._getWeight((n_visible,n_hidden))
            npWeights['p_nade_b'] = self._getWeight((n_visible,))
        else:
            assert False,'Invalid datatype: '+params['data_type']

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
            pass       
        elif self.params['inference_model']=='structured':
            npWeights['q_W_st_0'] = self._getWeight((self.params['dim_stochastic'], self.params['rnn_size']))
            npWeights['q_b_st_0'] = self._getWeight((self.params['rnn_size'],))
        else:
            assert False,'Invalid inference model: '+self.params['inference_model']
        RNN_SIZE = self.params['rnn_size']
        npWeights['q_W_mu']       = self._getWeight((RNN_SIZE, self.params['dim_stochastic']))
        npWeights['q_b_mu']       = self._getWeight((self.params['dim_stochastic'],))
        npWeights['q_W_cov'] = self._getWeight((RNN_SIZE, self.params['dim_stochastic']))
        npWeights['q_b_cov'] = self._getWeight((self.params['dim_stochastic'],))
        if self.params['var_model']=='lstmlr' and self.params['inference_model']=='mean_field':
            npWeights['q_W_mu_r']       = self._getWeight((RNN_SIZE, self.params['dim_stochastic']))
            npWeights['q_b_mu_r']       = self._getWeight((self.params['dim_stochastic'],))
            npWeights['q_W_cov_r'] = self._getWeight((RNN_SIZE, self.params['dim_stochastic']))
            npWeights['q_b_cov_r'] = self._getWeight((self.params['dim_stochastic'],))

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
    def _getEmissionFxn(self, z, X=None):
        """
        Apply emission function to zs
        Input:  z [bs x T x dim]
        Output: (params, ) or (mu, cov) of size [bs x T x dim]
        """
        
        if self.params['emission_type']=='mlp':
            self._p('EMISSION TYPE: MLP')
            hid = z
            for l in range(self.params['emission_layers']):
                hid = self._LinearNL(self.tWeights['p_emis_W_'+str(l)], self.tWeights['p_emis_b_'+str(l)], hid)
        elif self.params['emission_type']=='conditional':
            self._p('EMISSION TYPE: conditional')
            X_prev  = T.concatenate([T.zeros_like(X[:,[0],:]),X[:,:-1,:]],axis=1)
            hid= T.concatenate([z,X_prev],axis=2)
            for l in range(self.params['emission_layers']):
                hid = self._LinearNL(self.tWeights['p_emis_W_'+str(l)], self.tWeights['p_emis_b_'+str(l)], hid)
        elif self.params['emission_type']=='conditional_linear':
            self._p('EMISSION TYPE: Linear')
            X_prev  = T.concatenate([T.zeros_like(X[:,[0],:]),X[:,:-1,:]],axis=1)
            hid_z   = T.dot(z, self.tWeights['p_emis_W_0'])+self.tWeights['p_emis_b_0']
            hid_x   = T.dot(X_prev, self.tWeights['p_inp_W'])+ self.tWeights['p_inp_b']
            return [T.nnet.sigmoid(hid_z+hid_x)]
        else:
            assert False,'Invalid emission type'
    
        if self.params['data_type']=='binary':
            mean_params=T.nnet.sigmoid(self._LinearNL(self.tWeights['p_emis_W_ber'],
                                      self.tWeights['p_emis_b_ber'],hid,onlyLinear=True))
            return [mean_params]
        elif self.params['data_type']=='binary_nade':
            self._p('NADE observations')
            assert X is not None,'Need observations for NADE'
            #Shuffle the dimensions of the predicted matrix
            x_reshaped   = X.dimshuffle(2,0,1)
            x0 = T.ones_like(x_reshaped[0]) # bs x T
            a0 = hid #bs x T x nstoc (condition on stoc. states)
            nl= T.nnet.relu
            W = self.tWeights['p_nade_W']
            V = self.tWeights['p_nade_U']
            b = self.tWeights['p_nade_b']
            #Use a NADE at the output
            def NADEDensity(x, w, v, b,a_prev, x_prev):#, #Estimating likelihood
                a   = a_prev + T.dot(T.shape_padright(x_prev, 1), T.shape_padleft(w, 1))
                h   = T.nnet.sigmoid(a) #bs x T x nhid
                p_xi_is_one = T.nnet.sigmoid(T.dot(h, v) + b)
                return (a, x, p_xi_is_one)
            ([_, _, mean_params], _) = theano.scan(NADEDensity,
                                                   sequences=[x_reshaped, W, V, b],
                                                   outputs_info=[a0, x0, None])
            mean_params = mean_params.dimshuffle(1,2,0)
            return [mean_params]
        else:
            assert False,'Invalid type of data'
    
    def _getTransitionFxn(self, z, u=None, X=None, fixedLogCov = None):
        """
        Apply transition function to zs
        Input:  z [bs x T x dim], u<if actions present in model> [bs x T x dim]
        Output: mu, cov of size [bs x T x dim]
        """
        
        
        if self.params['transition_type']=='simple_gated':
            def mlp(inp, W1,b1,W2,b2,X_prev=None,W_prev=None,b_prev=None):
                if X_prev is not None:
                    h1 = self._LinearNL(W1,b1, T.concatenate([inp,X_prev],axis=2))
                else:
                    h1 = self._LinearNL(W1,b1, inp)
                h2 = T.dot(h1,W2)+b2
                return h2
            gateInp= z
            X_prev,W_prev,b_prev = None,None,None
            if self.params['use_prev_input']:
                #Concatenate Inputs: [-1] [0] [1] [2] ... [T-1]
                X_prev = T.concatenate([T.zeros_like(X[:,[0],:]),X[:,:-1,:]],axis=1)
            gate   = T.nnet.sigmoid(mlp(gateInp, self.tWeights['p_gate_embed_W_0'], self.tWeights['p_gate_embed_b_0'], 
                                        self.tWeights['p_gate_embed_W_1'],self.tWeights['p_gate_embed_b_1'],
                                        X_prev = X_prev, W_prev = W_prev, b_prev = b_prev))
            
            z_prop = mlp(z,self.tWeights['p_z_W_0'] ,self.tWeights['p_z_b_0'],
                         self.tWeights['p_z_W_1'] , self.tWeights['p_z_b_1'],
                        X_prev = X_prev, W_prev = W_prev, b_prev = b_prev)
            mu     = gate*z_prop + (1.-gate)*(T.dot(z, self.tWeights['p_trans_W_mu'])+self.tWeights['p_trans_b_mu'])
            cov    = T.nnet.softplus(T.dot(self._applyNL(z_prop), self.tWeights['p_trans_W_cov'])+
                                     self.tWeights['p_trans_b_cov'])
            return mu,cov
        elif self.params['transition_type']=='mlp':
            hid = z
            for l in range(self.params['transition_layers']):
                if l==0 and self.params['use_prev_input']:
                    X_prev = T.concatenate([T.zeros_like(X[:,[0],:]),X[:,:-1,:]],axis=1)
                    hid    = T.concatenate([hid,X_prev],axis=2)
                hid = self._LinearNL(self.tWeights['p_trans_W_'+str(l)],
                                     self.tWeights['p_trans_b_'+str(l)],
                                     hid)
            mu     = self._LinearNL(self.tWeights['p_trans_W_mu'],
                                      self.tWeights['p_trans_b_mu'],hid,onlyLinear=True)
            cov    = T.nnet.softplus(T.dot(hid, self.tWeights['p_trans_W_cov'])+
                                     self.tWeights['p_trans_b_cov'])
            return mu,cov
        else:
            assert False,'Invalid Transition type: '+str(self.params['transition_type'])

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
            return [l2r,r2l]
        else:
            assert False,'Invalid variational model'
        self._p(('Done <_buildLSTM> [Took %.4f]')%(time.time()-start_time))
        
    
    def _inferenceLayer(self, hidden_state, eps):
        """
        Take input of T x bs x dim and return z, mu, 
        sq each of size (bs x T x dim) 
        Input: hidden_state [T x bs x dim], eps [bs x T x dim]
        Output: z [bs x T x dim], mu [bs x T x dim], cov [bs x T x dim]
        """
        
        def structuredApproximation(h_t, eps_t, z_prev, 
                                    q_W_st_0, q_b_st_0,
                                    q_W_mu, q_b_mu,
                                    q_W_cov,q_b_cov):
            h_next     = T.tanh(T.dot(z_prev,q_W_st_0)+q_b_st_0)
            if self.params['var_model']=='lstmlr':
                h_next = (1./3.)*(h_t+h_next)
            else:
                h_next = (1./2.)*(h_t+h_next)
            mu_t         = T.dot(h_next,q_W_mu)+q_b_mu
            cov_t        = T.nnet.softplus(T.dot(h_next,q_W_cov)+q_b_cov)
            z_t          = mu_t+T.sqrt(cov_t)*eps_t
            return z_t, mu_t, cov_t
        
        if self.params['inference_model']=='structured':
            #Structured recognition networks
            if self.params['var_model']=='LR':
                state   = hidden_state[0]+hidden_state[1]
            else:
                state   = hidden_state
            eps_swap= eps.swapaxes(0,1)
            #Create empty extra dimension
            rval, _ = theano.scan(structuredApproximation, 
                                    sequences=[state, eps_swap],
                                    outputs_info=[
                                    T.alloc(np.asarray(0.,dtype=config.floatX), 
                                            eps_swap.shape[1], self.params['dim_stochastic']),
                                            None,None],
                                    non_sequences=[self.tWeights[k] for k in 
                                                   ['q_W_st_0', 'q_b_st_0']]+
                                                  [self.tWeights[k] for k in 
                                                   ['q_W_mu','q_b_mu','q_W_cov','q_b_cov']],
                                    name='structuredApproximation')
            z, mu, cov = rval[0].swapaxes(0,1), rval[1].swapaxes(0,1), rval[2].swapaxes(0,1)
            return z, mu, cov
        elif self.params['inference_model']=='mean_field':
            if self.params['var_model']=='LR':
                l2r = hidden_state[0].swapaxes(0,1)
                r2l = hidden_state[1].swapaxes(0,1)
                hidl2r = l2r
                mu_1     = T.dot(hidl2r,self.tWeights['q_W_mu'])+self.tWeights['q_b_mu']
                cov_1    = T.nnet.softplus(T.dot(hidl2r,self.tWeights['q_W_cov'])+self.tWeights['q_b_cov'])
                hidr2l = r2l
                mu_2     = T.dot(hidr2l,self.tWeights['q_W_mu_r'])+self.tWeights['q_b_mu_r']
                cov_2    = T.nnet.softplus(T.dot(hidr2l,self.tWeights['q_W_cov_r'])+self.tWeights['q_b_cov_r'])
                mu = (mu_1*cov_2+mu_2*cov_1)/(cov_1+cov_2)
                cov= (cov_1*cov_2)/(cov_1+cov_2)
                z = mu + T.sqrt(cov)*eps
            else:
                hid       = hidden_state.swapaxes(0,1)
                mu        = T.dot(hid,self.tWeights['q_W_mu'])+self.tWeights['q_b_mu']
                cov       = T.nnet.softplus(T.dot(hid,self.tWeights['q_W_cov'])+ self.tWeights['q_b_cov'])
                z         = mu + T.sqrt(cov)*eps
            return z,mu,cov
        else:
            assert False,'Invalid recognition model'
        
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
        z_q,mu_q,cov_q    = self._inferenceLayer(hidden_state, eps)
        
        #Regularize z_q (for train) - Added back in May 16
        if dropout_prob>0.:
            z_q  = z_q + self.srng.normal(z_q.shape, 0.,0.0025,dtype=config.floatX)
        z_q.name          = 'z_q'
        
        observation_params = self._getEmissionFxn(z_q,X=X)
        mu_trans,cov_trans = self._getTransitionFxn(z_q, u=U,X=X)
        mu_prior     = T.concatenate([T.alloc(np.asarray(0.).astype(config.floatX),
                                              X.shape[0],1,self.params['dim_stochastic']),
                       mu_trans[:,:-1,:]],axis=1)
        cov_prior = T.concatenate([T.alloc(np.asarray(1.).astype(config.floatX),
                                              X.shape[0],1,self.params['dim_stochastic']),
                       cov_trans[:,:-1,:]],axis=1)
        return observation_params, z_q, mu_q, cov_q, mu_prior, cov_prior, mu_trans, cov_trans
    
    
    def _getTemporalKL(self, mu_q, cov_q, mu_prior, cov_prior, M, batchVector = False):
        """
        TemporalKL divergence KL (q||p)
        
        KL(q_t||p_t) = 0.5*(log|sigmasq_p| -log|sigmasq_q|  -D + Tr(sigmasq_p^-1 sigmasq_q) 
                        + (mu_p-mu_q)^T sigmasq_p^-1 (mu_p-mu_q))
        M is a mask of size bs x T that should be applied once the KL divergence for each point
        across time has been estimated
        """
        assert np.all(cov_q.tag.test_value>0.),'should be positive'
        assert np.all(cov_prior.tag.test_value>0.),'should be positive'
        diff_mu = mu_prior-mu_q
        KL_t    = T.log(cov_prior)-T.log(cov_q) - 1. + cov_q/cov_prior + diff_mu**2/cov_prior
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
        
        assert self.params['data_type']=='binary' or self.params['data_type']=='binary_nade','binary only'
        mean_p = obs_params[0]
        negCLL = (T.nnet.binary_crossentropy(mean_p,X).sum(2)*M).sum(1,keepdims=True)
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
        U = None
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
        #Halve the learning rate
        #lr_update = [(lr,T.switch(lr*0.90<1e-4,lr,lr*0.90))]                          
        anneal_div     = 1000.
        if 'anneal_rate' in self.params:
            self._p('Anneal = 1 in '+str(self.params['anneal_rate'])+' param. updates')
            anneal_div = self.params['anneal_rate']
        if 'synthetic' in self.params['dataset']:
            anneal_div = 100.
        anneal_update  = [(iteration_t, iteration_t+1),
                          (anneal,T.switch(0.01+iteration_t/anneal_div>1,1,0.01+iteration_t/anneal_div))]
        
        fxn_inputs = [X, M, eps]
        if U is not None:
            fxn_inputs.append(U)
        assert U is None,'Will not work with U'
        if not self.params['validate_only']:
            print '****** CREATING TRAINING FUNCTION*****'
            ############# Setup training functions ###########
            obs_params, z_q, mu_q, cov_q, mu_prior, cov_prior, _, _ = self._inferenceAndReconstruction( 
                                                              X, eps, U= U,
                                                              dropout_prob = self.params['rnn_dropout'])
            negCLL = self._getNegCLL(obs_params, X, M)
            TemporalKL = self._getTemporalKL(mu_q, cov_q, mu_prior, cov_prior, M)
            train_cost = negCLL+anneal*TemporalKL

            #Get updates from optimizer
            model_params         = self._getModelParams()
            optimizer_up, norm_list  = self._setupOptimizer(train_cost, model_params,lr = lr, 
                                                            reg_type =self.params['reg_type'], 
                                                            reg_spec =self.params['reg_spec'], 
                                                            reg_value= self.params['reg_value'],
                                                            divide_grad = T.cast(X.shape[0],dtype=config.floatX),   
                                                           grad_norm = 1.)
                                                           
            #Add annealing updates
            optimizer_up +=anneal_update
            #self.decay_lr   = theano.function([],lr.sum(),name = 'Update LR',updates=lr_update)

            ############# Setup train & evaluate functions ###########
            self.train_debug         = theano.function(fxn_inputs,[train_cost,norm_list[0],norm_list[1],
                                                                        norm_list[2],negCLL, TemporalKL, anneal.sum()], 
                                                           updates = optimizer_up, name='Train (with Debug)')
        
        eval_obs_params, eval_z_q, eval_mu_q, eval_cov_q, eval_mu_prior, eval_cov_prior, \
        eval_mu_trans, eval_cov_trans = self._inferenceAndReconstruction(
                                                          X, eps, U= U,
                                                          dropout_prob = 0.)
        eval_z_q.name = 'eval_z_q'
        
        eval_CNLLvec=self._getNegCLL(eval_obs_params, X, M, batchVector = True)
        eval_KLvec  = self._getTemporalKL(eval_mu_q, eval_cov_q,eval_mu_prior, 
                                          eval_cov_prior, M, batchVector = True)
        eval_cost   = eval_CNLLvec + eval_KLvec
        
        #From here on, convert to the log covariance since we only use it for evaluation
        assert np.all(eval_cov_q.tag.test_value>0.),'should be positive'
        assert np.all(eval_cov_prior.tag.test_value>0.),'should be positive'
        assert np.all(eval_cov_trans.tag.test_value>0.),'should be positive'
        
        #convert to log domain - easier to work with
        eval_logcov_q     = T.log(eval_cov_q)
        eval_logcov_prior = T.log(eval_cov_prior)
        eval_logcov_trans = T.log(eval_cov_trans)
        
        ll_prior     = self._llGaussian(eval_z_q, eval_mu_prior, eval_logcov_prior).sum(2)*M
        ll_posterior = self._llGaussian(eval_z_q, eval_mu_q, eval_logcov_q).sum(2)*M
        ll_estimate  = -1*eval_CNLLvec+ll_prior.sum(1,keepdims=True)-ll_posterior.sum(1,keepdims=True)
        
        eval_inputs = [eval_z_q]
        if U is not None:
            eval_inputs.append(U)
        
        self.likelihood          = theano.function(fxn_inputs, ll_estimate, name = 'Importance Sampling based likelihood')
        self.evaluate            = theano.function(fxn_inputs, eval_cost, name = 'Evaluate Bound')
        if self.params['use_prev_input']:
            eval_inputs.append(X)
        self.transition_fxn      = theano.function(eval_inputs,[eval_mu_trans, eval_logcov_trans],
                                                       name='Transition Function')
        emission_inputs = [eval_z_q]
        if self.params['emission_type']=='conditional' or self.params['emission_type']=='conditional_linear':
            emission_inputs.append(X)
        if self.params['data_type']!='binary_nade': #Don't write the emission function for NADE
            self.emission_fxn = theano.function(emission_inputs, eval_obs_params, name='Emission Function')
        self.posterior_inference = theano.function([X, eps], [eval_z_q, eval_mu_q, eval_logcov_q],name='Posterior Inference') 

    #"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""# 
    def sampleGaussian(self,mu,logsigmasq):
        return mu + np.random.randn(*mu.shape)*np.exp(0.5*logsigmasq)
    """
                                      Sample from Generative Model
    """
    def sample(self, nsamples=100, T=10, U= None, additional = {}):
        if self.params['dim_actions']>0:
            assert U is not None,'Specify U for sampling model conditioned on actions'
        assert T>1, 'Sample atleast 2 timesteps'
        #Initial sample
        z      = np.random.randn(nsamples,1,self.params['dim_stochastic']).astype(config.floatX)
        all_zs = [np.copy(z)]
        additional['mu']     = []
        additional['logcov'] = []
        for t in range(T-1):
            mu,logcov = self.transition_fxn(z)
            z           = self.sampleGaussian(mu,logcov).astype(config.floatX)
            all_zs.append(np.copy(z))
            additional['mu'].append(np.copy(mu))
            additional['logcov'].append(np.copy(logcov))
        zvec = np.concatenate(all_zs,axis=1)
        additional['mu']     = np.concatenate(additional['mu'], axis=1)
        additional['logcov'] = np.concatenate(additional['logcov'], axis=1)
        return self.emission_fxn(zvec), zvec
    
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
    
    def annealLearningRate(self, valid_err):
        """
        Calls the function to anneal learning rate (if cost larger than mean of last 3)
        """
        assert valid_err.ndim==1,'Expecting vector of costs not matrix'
        if valid_err.shape[0]>3:
            last_val     = valid_err[-1]
            running_mean = np.mean(valid_err[-3:])
            if last_val>running_mean:
                return True
        return False
    
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
        idxlist   = range(N)
        batchlist = np.split(idxlist, range(batch_size,N,batch_size))
                             
        bound_train_list,bound_valid_list,bound_tsbn_list = [],[],[]
        p_norm, g_norm, opt_norm = None, None, None

        #Lists used to track quantities
        mu_list_train, cov_list_train, mu_list_valid, cov_list_valid, nll_valid_list = [],[],[],[],[]
        current_lr = self.params['lr']
        
        assert actions is None,'No actions supported'
        
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
            
            #if epoch>100 and epoch%500==0:
            #    self._p('ANNEALING LEARNING RATE NOW')
            #    current_lr = self.decay_lr()
                
            for bnum, batch_idx in enumerate(batchlist):
                batch_idx  = batchlist[bnum]
                X       = dataset[batch_idx,:,:].astype(config.floatX)
                M       = mask[batch_idx,:].astype(config.floatX)
                U       = None
                
                #Tack on 0's if the matrix size (optimization->theano doesnt have to redefine matrices)
                if X.shape[0]<batch_size:
                    Nremaining = int(batch_size-X.shape[0])
                    X   = np.concatenate([X,np.zeros((Nremaining,X.shape[1],X.shape[2]))],axis=0).astype(config.floatX)
                    M   = np.concatenate([M,np.zeros((Nremaining,X.shape[1]))],axis=0).astype(config.floatX)
                
                #Reduce the dimensionality of the tensors based on the maximum size of the mask
                maxT    = int(np.max(M.sum(1)))
                X       = X[:,:maxT,:]
                M       = M[:,:maxT]
                
                
                eps     = np.random.randn(X.shape[0],maxT,self.params['dim_stochastic']).astype(config.floatX)
                
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
    
    def impSamplingNLL(self, dataset, mask, batch_size, actions=None, S = 2, normalization = 'frame'):
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
                if s%500==0:
                    self._p('Done '+str(s))
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
