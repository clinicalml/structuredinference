from __future__ import division
import six.moves.cPickle as pickle
from collections import OrderedDict
import numpy as np
import sys, time, os, gzip, theano,math
sys.path.append('../')
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
                                                DEEP MARKOV MODEL [DEEP KALMAN FILTER]
"""
class DKF(BaseModel, object):
    def __init__(self, params, paramFile=None, reloadFile=None):
        self.scan_updates = []
        super(DKF,self).__init__(params, paramFile=paramFile, reloadFile=reloadFile)
        if 'synthetic' in self.params['dataset'] and not hasattr(self, 'params_synthetic'):
            assert False, 'Expecting to have params_synthetic as an attribute in DKF class'
        assert self.params['nonlinearity']!='maxout','Maxout nonlinearity not supported'

    #"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""#
    def _createParams(self):
        """ Model parameters """
        npWeights = OrderedDict()
        self._createInferenceParams(npWeights)
        self._createGenerativeParams(npWeights)
        return npWeights
    
    def _createGenerativeParams(self, npWeights):
        """ Create weights/params for generative model """
        if 'synthetic' in self.params['dataset']:
            updateParamsSynthetic(params_synthetic)
            self.params_synthetic = params_synthetic
            for k in self.params_synthetic[self.params['dataset']]['params']:
                npWeights[k+'_W'] = np.array(np.random.uniform(-0.2,0.2),dtype=config.floatX) 
            return
        DIM_HIDDEN     = self.params['dim_hidden']
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
            #Initialize the weights to be identity
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
        elif self.params['emission_type'] == 'res':
            for l in range(self.params['emission_layers']):
                dim_input,dim_output = DIM_HIDDEN, DIM_HIDDEN
                if l==0:
                    dim_input = self.params['dim_stochastic']
                npWeights['p_emis_W_'+str(l)] = self._getWeight((dim_input, dim_output))
                npWeights['p_emis_b_'+str(l)] = self._getWeight((dim_output,))
            dim_res_out = self.params['dim_observations']
            if self.params['data_type']=='binary_nade':
                dim_res_out = DIM_HIDDEN
            npWeights['p_res_W'] = self._getWeight((self.params['dim_stochastic'], dim_res_out))
        elif self.params['emission_type'] =='conditional':
            for l in range(self.params['emission_layers']):
                dim_input,dim_output = DIM_HIDDEN, DIM_HIDDEN
                if l==0:
                    dim_input = self.params['dim_stochastic']+self.params['dim_observations']
                npWeights['p_emis_W_'+str(l)] = self._getWeight((dim_input, dim_output))
                npWeights['p_emis_b_'+str(l)] = self._getWeight((dim_output,))
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
        """  Create weights/params for inference network """
        
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
        
        #Embedding before MF/ST inference model
        if self.params['inference_model']=='mean_field':
            pass 
        elif self.params['inference_model']=='structured':
            DIM_INPUT = self.params['dim_stochastic']
            if self.params['use_generative_prior']:
                DIM_INPUT = self.params['rnn_size']
            npWeights['q_W_st_0'] = self._getWeight((DIM_INPUT, self.params['rnn_size']))
            npWeights['q_b_st_0'] = self._getWeight((self.params['rnn_size'],))
        else:
            assert False,'Invalid inference model: '+self.params['inference_model']
        RNN_SIZE = self.params['rnn_size']
        npWeights['q_W_mu']       = self._getWeight((RNN_SIZE, self.params['dim_stochastic']))
        npWeights['q_b_mu']       = self._getWeight((self.params['dim_stochastic'],))
        npWeights['q_W_cov'] = self._getWeight((RNN_SIZE, self.params['dim_stochastic']))
        npWeights['q_b_cov'] = self._getWeight((self.params['dim_stochastic'],))
        if self.params['var_model']=='LR' and self.params['inference_model']=='mean_field':
            npWeights['q_W_mu_r']       = self._getWeight((RNN_SIZE, self.params['dim_stochastic']))
            npWeights['q_b_mu_r']       = self._getWeight((self.params['dim_stochastic'],))
            npWeights['q_W_cov_r'] = self._getWeight((RNN_SIZE, self.params['dim_stochastic']))
            npWeights['q_b_cov_r'] = self._getWeight((self.params['dim_stochastic'],))

    def _createLSTMWeights(self, npWeights):
        #LSTM L/LR/R w/ orthogonal weight initialization
        suffices_to_build = []
        if self.params['var_model']=='LR' or self.params['var_model']=='L':
            suffices_to_build.append('l')
        if self.params['var_model']=='LR' or self.params['var_model']=='R':
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
        if 'synthetic' in self.params['dataset']:
            self._p('Using emission function for '+self.params['dataset'])
            tParams = {}
            for k in self.params_synthetic[self.params['dataset']]['params']:
                tParams[k] = self.tWeights[k+'_W']
            mu       = self.params_synthetic[self.params['dataset']]['obs_fxn'](z, fxn_params = tParams)
            cov      = T.ones_like(mu)*self.params_synthetic[self.params['dataset']]['obs_cov']
            cov.name = 'EmissionCov'
            return [mu,cov]
        
        if self.params['emission_type'] in ['mlp','res']:
            self._p('EMISSION TYPE: MLP or RES')
            hid = z
        elif self.params['emission_type']=='conditional':
            self._p('EMISSION TYPE: conditional')
            X_prev  = T.concatenate([T.zeros_like(X[:,[0],:]),X[:,:-1,:]],axis=1)
            hid     = T.concatenate([z,X_prev],axis=2)
        else:
            assert False,'Invalid emission type'
        for l in range(self.params['emission_layers']):
            if self.params['data_type']=='binary_nade' and l==self.params['emission_layers']-1:
                hid = T.dot(hid, self.tWeights['p_emis_W_'+str(l)]) + self.tWeights['p_emis_b_'+str(l)]
            else:
                hid = self._LinearNL(self.tWeights['p_emis_W_'+str(l)],  self.tWeights['p_emis_b_'+str(l)], hid)
            
        if self.params['data_type']=='binary':
            if self.params['emission_type']=='res':
                hid = T.dot(z,self.tWeights['p_res_W'])+T.dot(hid,self.tWeights['p_emis_W_ber'])+self.tWeights['p_emis_b_ber']
                mean_params=T.nnet.sigmoid(hid)
            else:
                mean_params=T.nnet.sigmoid(T.dot(hid,self.tWeights['p_emis_W_ber'])+self.tWeights['p_emis_b_ber'])
            return [mean_params]
        elif self.params['data_type']=='binary_nade':
            self._p('NADE observations')
            assert X is not None,'Need observations for NADE'
            if self.params['emission_type']=='res':
                hid += T.dot(z,self.tWeights['p_res_W'])
            x_reshaped   = X.dimshuffle(2,0,1)
            x0 = T.ones((hid.shape[0],hid.shape[1]))#x_reshaped[0]) # bs x T
            a0 = hid #bs x T x nhid
            W = self.tWeights['p_nade_W']
            V = self.tWeights['p_nade_U']
            b = self.tWeights['p_nade_b']
            #Use a NADE at the output
            def NADEDensity(x, w, v, b, a_prev, x_prev):#Estimating likelihood
                a   = a_prev + T.dot(T.shape_padright(x_prev, 1), T.shape_padleft(w, 1)) 
                h   = T.nnet.sigmoid(a) #Original - bs x T x nhid
                p_xi_is_one = T.nnet.sigmoid(T.dot(h, v) + b)
                return (a, x, p_xi_is_one)
            ([_, _, mean_params], _) = theano.scan(NADEDensity,
                                                   sequences=[x_reshaped, W, V, b],
                                                   outputs_info=[a0, x0,None])
            #theano function to sample from NADE
            def NADESample(w, v, b, a_prev_s, x_prev_s):
                a_s   = a_prev_s + T.dot(T.shape_padright(x_prev_s, 1), T.shape_padleft(w, 1))
                h_s   = T.nnet.sigmoid(a_s) #Original - bs x T x nhid
                p_xi_is_one_s = T.nnet.sigmoid(T.dot(h_s, v) + b)
                x_s   = T.switch(p_xi_is_one_s>0.5,1.,0.)
                return (a_s, x_s, p_xi_is_one_s)
            ([_, _, sampled_params], _) = theano.scan(NADESample,
                                                   sequences=[W, V, b],
                                                   outputs_info=[a0, x0,None])
            """
            def NADEDensityAndSample(x, w, v, b, 
                                     a_prev,   x_prev, 
                                     a_prev_s, x_prev_s ):
                a     = a_prev + T.dot(T.shape_padright(x_prev, 1), T.shape_padleft(w, 1))
                h     = T.nnet.sigmoid(a) #bs x T x nhid
                p_xi_is_one = T.nnet.sigmoid(T.dot(h, v) + b)
                
                a_s   = a_prev_s + T.dot(T.shape_padright(x_prev_s, 1), T.shape_padleft(w, 1))
                h_s   = T.nnet.sigmoid(a_s) #bs x T x nhid
                p_xi_is_one_s = T.nnet.sigmoid(T.dot(h_s, v) + b)
                x_s   = T.switch(p_xi_is_one_s>0.5,1.,0.)
                return (a, x, a_s, x_s, p_xi_is_one, p_xi_is_one_s)
            
            ([_, _, _, _, mean_params,sampled_params], _) = theano.scan(NADEDensityAndSample,
                                                   sequences=[x_reshaped, W, V, b],
                                                   outputs_info=[a0, x0, a0, x0, None, None])
            """
            sampled_params = sampled_params.dimshuffle(1,2,0)
            mean_params    = mean_params.dimshuffle(1,2,0)
            return [mean_params,sampled_params]
        else:
            assert False,'Invalid type of data'
    
    def _getTransitionFxn(self, z, X=None):
        """
        Apply transition function to zs
        Input:  z [bs x T x dim], u<if actions present in model> [bs x T x dim]
        Output: mu, cov of size [bs x T x dim]
        """
        if 'synthetic' in self.params['dataset']:
            self._p('Using transition function for '+self.params['dataset'])
            tParams = {}
            for k in self.params_synthetic[self.params['dataset']]['params']:
                tParams[k] = self.tWeights[k+'_W']
            mu  = self.params_synthetic[self.params['dataset']]['trans_fxn'](z, fxn_params = tParams)
            cov = T.ones_like(mu)*self.params_synthetic[self.params['dataset']]['trans_cov']
            cov.name = 'TransitionCov'
            return mu,cov
        
        if self.params['transition_type']=='simple_gated':
            def mlp(inp, W1,b1,W2,b2, X_prev=None):
                if X_prev is not None:
                    h1 = self._LinearNL(W1,b1, T.concatenate([inp,X_prev],axis=2))
                else:
                    h1 = self._LinearNL(W1,b1, inp)
                h2 = T.dot(h1,W2)+b2
                return h2
            
            gateInp= z
            X_prev = None
            if self.params['use_prev_input']:
                X_prev = T.concatenate([T.zeros_like(X[:,[0],:]),X[:,:-1,:]],axis=1)
            gate   = T.nnet.sigmoid(mlp(gateInp, self.tWeights['p_gate_embed_W_0'], self.tWeights['p_gate_embed_b_0'], 
                                        self.tWeights['p_gate_embed_W_1'],self.tWeights['p_gate_embed_b_1'],
                                        X_prev = X_prev))
            z_prop = mlp(z,self.tWeights['p_z_W_0'] ,self.tWeights['p_z_b_0'], 
                         self.tWeights['p_z_W_1'] , self.tWeights['p_z_b_1'], X_prev = X_prev)
            mu     = gate*z_prop + (1.-gate)*(T.dot(z, self.tWeights['p_trans_W_mu'])+self.tWeights['p_trans_b_mu'])
            cov    = T.nnet.softplus(T.dot(self._applyNL(z_prop), self.tWeights['p_trans_W_cov'])+
                                     self.tWeights['p_trans_b_cov'])
            return mu,cov
        elif self.params['transition_type']=='mlp':
            hid = z
            if self.params['use_prev_input']:
                X_prev = T.concatenate([T.zeros_like(X[:,[0],:]),X[:,:-1,:]],axis=1)
                hid    = T.concatenate([hid,X_prev],axis=2)
            for l in range(self.params['transition_layers']):
                hid = self._LinearNL(self.tWeights['p_trans_W_'+str(l)],self.tWeights['p_trans_b_'+str(l)],hid)
            mu     = T.dot(hid, self.tWeights['p_trans_W_mu']) + self.tWeights['p_trans_b_mu']
            cov    = T.nnet.softplus(T.dot(hid, self.tWeights['p_trans_W_cov'])+self.tWeights['p_trans_b_cov'])
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
        if self.params['var_model']=='R':
            suffix='r'
            return self._LSTMlayer(embedding, suffix, dropout_prob)
        elif self.params['var_model']=='L':
            suffix='l'
            return self._LSTMlayer(embedding, suffix, dropout_prob)
        elif self.params['var_model']=='LR':
            suffix='l'
            l2r   = self._LSTMlayer(embedding, suffix, dropout_prob)
            suffix='r'
            r2l   = self._LSTMlayer(embedding, suffix, dropout_prob)
            return [l2r,r2l]
        else:
            assert False,'Invalid variational model'
        self._p(('Done <_buildLSTM> [Took %.4f]')%(time.time()-start_time))
        
    
    def _inferenceLayer(self, hidden_state):
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
            #Using the prior distribution directly
            if self.params['use_generative_prior']:
                assert not self.params['use_prev_input'],'No support for using previous input'
                #Get mu/cov from z_prev through prior distribution
                mu_1,cov_1 = self._getTransitionFxn(z_prev) 
                #Combine with estimate of mu/cov from data
                h_data     = T.tanh(T.dot(h_t,q_W_st_0)+q_b_st_0)
                mu_2       = T.dot(h_data,q_W_mu)+q_b_mu
                cov_2      = T.nnet.softplus(T.dot(h_data,q_W_cov)+q_b_cov)
                mu         = (mu_1*cov_2+mu_2*cov_1)/(cov_1+cov_2)
                cov        = (cov_1*cov_2)/(cov_1+cov_2)
                z          = mu + T.sqrt(cov)*eps_t
                return z, mu, cov
            else:
                h_next     = T.tanh(T.dot(z_prev,q_W_st_0)+q_b_st_0)
                if self.params['var_model']=='LR':
                    h_next = (1./3.)*(h_t+h_next)
                else:
                    h_next = (1./2.)*(h_t+h_next)
                mu_t       = T.dot(h_next,q_W_mu)+q_b_mu
                cov_t      = T.nnet.softplus(T.dot(h_next,q_W_cov)+q_b_cov)
                z_t        = mu_t+T.sqrt(cov_t)*eps_t
                return z_t, mu_t, cov_t
        if type(hidden_state) is list:
            eps         = self.srng.normal(size=(hidden_state[0].shape[1],hidden_state[0].shape[0],self.params['dim_stochastic'])) 
        else:
            eps         = self.srng.normal(size=(hidden_state.shape[1],hidden_state.shape[0],self.params['dim_stochastic'])) 
        if self.params['inference_model']=='structured':
            #Structured recognition networks
            if self.params['var_model']=='LR':
                state   = hidden_state[0]+hidden_state[1]
            else:
                state   = hidden_state
            eps_swap    = eps.swapaxes(0,1)
            if self.params['dim_stochastic']==1:
                """
                TODO: Write to theano authors regarding this issue.
                Workaround for theano issue: The result of a matrix multiply is a "matrix"
                even if one of the dimensions is 1. However defining a tensor with one dimension one
                means theano regards the resulting tensor as a matrix and consequently in the
                scan as a column. This results in a mismatch in tensor type in input (column)
                and output (matrix) and throws an error. This is a workaround that preserves
                type while not affecting dimensions
                """
                z0 = T.zeros((eps_swap.shape[1], self.params['rnn_size']))
                z0 = T.dot(z0,T.zeros_like(self.tWeights['q_W_mu']))
            else:
                z0 = T.zeros((eps_swap.shape[1], self.params['dim_stochastic']))
            rval, _     = theano.scan(structuredApproximation, 
                                    sequences=[state, eps_swap],
                                    outputs_info=[z0, None,None],
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
                hidl2r   = l2r
                mu_1     = T.dot(hidl2r,self.tWeights['q_W_mu'])+self.tWeights['q_b_mu']
                cov_1    = T.nnet.softplus(T.dot(hidl2r, self.tWeights['q_W_cov'])+self.tWeights['q_b_cov'])
                hidr2l   = r2l
                mu_2     = T.dot(hidr2l,self.tWeights['q_W_mu_r'])+self.tWeights['q_b_mu_r']
                cov_2    = T.nnet.softplus(T.dot(hidr2l, self.tWeights['q_W_cov_r'])+self.tWeights['q_b_cov_r'])
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
        """ Embed for q """
        return self._LinearNL(self.tWeights['q_W_input_0'],self.tWeights['q_b_input_0'], X)
    #"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""# 
    
    def _inferenceAndReconstruction(self, X, dropout_prob = 0.):
        """
        Returns z_q, mu_q and cov_q
        """
        self._p('Building with dropout:'+str(dropout_prob))
        embedding         = self._qEmbeddingLayer(X)
        hidden_state      = self._buildLSTM(X, embedding, dropout_prob)
        z_q,mu_q,cov_q    = self._inferenceLayer(hidden_state)
        
        #Regularize z_q (for train) 
        #if dropout_prob>0.:
        #    z_q  = z_q + self.srng.normal(z_q.shape, 0.,0.0025,dtype=config.floatX)
        #z_q.name          = 'z_q'
        
        observation_params = self._getEmissionFxn(z_q,X=X)
        mu_trans, cov_trans= self._getTransitionFxn(z_q, X=X)
        mu_prior     = T.concatenate([T.alloc(np.asarray(0.).astype(config.floatX),
                                              X.shape[0],1,self.params['dim_stochastic']), mu_trans[:,:-1,:]],axis=1)
        cov_prior = T.concatenate([T.alloc(np.asarray(1.).astype(config.floatX),
                                              X.shape[0],1,self.params['dim_stochastic']), cov_trans[:,:-1,:]],axis=1)
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
        if self.params['data_type']=='real':
            mu_p      = obs_params[0]
            cov_p     = obs_params[1]
            std_p     = T.sqrt(cov_p)
            negCLL_t  = 0.5 * np.log(2 * np.pi) + 0.5*T.log(cov_p) + 0.5 * ((X - mu_p) / std_p)**2
            negCLL    = (negCLL_t.sum(2)*M).sum(1,keepdims=True)
        else:
            mean_p = obs_params[0]
            negCLL = (T.nnet.binary_crossentropy(mean_p,X).sum(2)*M).sum(1,keepdims=True)
        if batchVector:
            return negCLL
        else:
            return negCLL.sum()
    
    def resetDataset(self, newX,newM,quiet=False):
        if not quiet:
            ddim,mdim = self.dimData()
            self._p('Original dim:'+str(ddim)+', '+str(mdim))
        self.setData(newX=newX.astype(config.floatX),newMask=newM.astype(config.floatX))
        if not quiet:
            ddim,mdim = self.dimData()
            self._p('New dim:'+str(ddim)+', '+str(mdim))
    def _buildModel(self):
        if 'synthetic' in self.params['dataset']:
            self.params_synthetic = params_synthetic
        """ High level function to build and setup theano functions """
        #X      = T.tensor3('X',   dtype=config.floatX)
        #eps    = T.tensor3('eps', dtype=config.floatX)
        #M      = T.matrix('M', dtype=config.floatX)
        idx                = T.vector('idx',dtype='int64')
        idx.tag.test_value = np.array([0,1]).astype('int64')
        self.dataset       = theano.shared(np.random.uniform(0,1,size=(3,5,self.params['dim_observations'])).astype(config.floatX))
        self.mask          = theano.shared(np.array(([[1,1,1,1,0],[1,1,0,0,0],[1,1,1,0,0]])).astype(config.floatX))
        X_o                = self.dataset[idx]
        M_o                = self.mask[idx]
        maxidx             = T.cast(M_o.sum(1).max(),'int64')
        X                  = X_o[:,:maxidx,:]
        M                  = M_o[:,:maxidx]
        newX,newMask       = T.tensor3('newX',dtype=config.floatX),T.matrix('newMask',dtype=config.floatX)
        self.setData       = theano.function([newX,newMask],None,updates=[(self.dataset,newX),(self.mask,newMask)])
        self.dimData       = theano.function([],[self.dataset.shape,self.mask.shape])
        
        #Learning Rates and annealing objective function
        #Add them to npWeights/tWeights to be tracked [do not have a prefix _W or _b so wont be diff.]
        self._addWeights('lr', np.asarray(self.params['lr'],dtype=config.floatX),borrow=False)
        self._addWeights('anneal', np.asarray(0.01,dtype=config.floatX),borrow=False)
        self._addWeights('update_ctr', np.asarray(1.,dtype=config.floatX),borrow=False)
        lr             = self.tWeights['lr']
        anneal         = self.tWeights['anneal']
        iteration_t    = self.tWeights['update_ctr']
        
        anneal_div     = 1000.
        if 'anneal_rate' in self.params:
            self._p('Anneal = 1 in '+str(self.params['anneal_rate'])+' param. updates')
            anneal_div = self.params['anneal_rate']
        if 'synthetic' in self.params['dataset']:
            anneal_div = 100.
        anneal_update  = [(iteration_t, iteration_t+1),
                          (anneal,T.switch(0.01+iteration_t/anneal_div>1,1,0.01+iteration_t/anneal_div))]
        fxn_inputs = [idx]
        if not self.params['validate_only']:
            print '****** CREATING TRAINING FUNCTION*****'
            ############# Setup training functions ###########
            obs_params, z_q, mu_q, cov_q, mu_prior, cov_prior, _, _ = self._inferenceAndReconstruction( 
                                                              X, dropout_prob = self.params['rnn_dropout'])
            negCLL = self._getNegCLL(obs_params, X, M)
            TemporalKL = self._getTemporalKL(mu_q, cov_q, mu_prior, cov_prior, M)
            train_cost = negCLL+anneal*TemporalKL

            #Get updates from optimizer
            model_params         = self._getModelParams()
            optimizer_up, norm_list  = self._setupOptimizer(train_cost, model_params,lr = lr, 
                                                            #Turning off for synthetic
                                                            #reg_type =self.params['reg_type'], 
                                                            #reg_spec =self.params['reg_spec'], 
                                                            #reg_value= self.params['reg_value'],
                                                            divide_grad = T.cast(X.shape[0],dtype=config.floatX),   
                                                           grad_norm = 1.)
                                                           
            #Add annealing updates
            optimizer_up +=anneal_update+self.updates
            self._p(str(len(self.updates))+' other updates')
            ############# Setup train & evaluate functions ###########
            self.train_debug         = theano.function(fxn_inputs,[train_cost,norm_list[0],norm_list[1],
                                                                        norm_list[2],negCLL, TemporalKL, anneal.sum()], 
                                                           updates = optimizer_up, name='Train (with Debug)')
        #Updates ack
        self.updates_ack = True
        eval_obs_params, eval_z_q, eval_mu_q, eval_cov_q, eval_mu_prior, eval_cov_prior, \
        eval_mu_trans, eval_cov_trans = self._inferenceAndReconstruction(X,dropout_prob = 0.)
        eval_z_q.name = 'eval_z_q'
        eval_CNLLvec=self._getNegCLL(eval_obs_params, X, M, batchVector = True)
        eval_KLvec  = self._getTemporalKL(eval_mu_q, eval_cov_q,eval_mu_prior, eval_cov_prior, M, batchVector = True)
        eval_cost   = eval_CNLLvec + eval_KLvec
        
        #From here on, convert to the log covariance since we only use it for evaluation
        assert np.all(eval_cov_q.tag.test_value>0.),'should be positive'
        assert np.all(eval_cov_prior.tag.test_value>0.),'should be positive'
        assert np.all(eval_cov_trans.tag.test_value>0.),'should be positive'
        eval_logcov_q     = T.log(eval_cov_q)
        eval_logcov_prior = T.log(eval_cov_prior)
        eval_logcov_trans = T.log(eval_cov_trans)
        
        ll_prior     = self._llGaussian(eval_z_q, eval_mu_prior, eval_logcov_prior).sum(2)*M
        ll_posterior = self._llGaussian(eval_z_q, eval_mu_q, eval_logcov_q).sum(2)*M
        ll_estimate  = -1*eval_CNLLvec+ll_prior.sum(1,keepdims=True)-ll_posterior.sum(1,keepdims=True)
        
        eval_inputs = [eval_z_q]
        self.likelihood          = theano.function(fxn_inputs, ll_estimate, name = 'Importance Sampling based likelihood')
        self.evaluate            = theano.function(fxn_inputs, eval_cost, name = 'Evaluate Bound')
        if self.params['use_prev_input']:
            eval_inputs.append(X)
        self.transition_fxn      = theano.function(eval_inputs,[eval_mu_trans, eval_logcov_trans],
                                                       name='Transition Function')
        emission_inputs = [eval_z_q]
        if self.params['emission_type']=='conditional':
            emission_inputs.append(X)
        if self.params['data_type']=='binary_nade':
            self.emission_fxn = theano.function(emission_inputs, 
                                                eval_obs_params[1], name='Emission Function')
        else:
            self.emission_fxn = theano.function(emission_inputs, 
                                                eval_obs_params[0], name='Emission Function')
        self.posterior_inference = theano.function(fxn_inputs, 
                                                   [eval_z_q, eval_mu_q, eval_logcov_q],
                                                   name='Posterior Inference') 
    #"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""# 
if __name__=='__main__':
    """ use this to check compilation for various options"""
    from parse_args_dkf import params
    if params['use_nade']:
        params['data_type'] = 'binary_nade'
    else:
        params['data_type'] = 'binary'
    params['dim_observations']  = 10
    dkf = DKF(params, paramFile = 'tmp')
    os.unlink('tmp')
    import ipdb;ipdb.set_trace()
