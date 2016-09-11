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
    def _createParams(self):
        """ Model parameters """
        npWeights = OrderedDict()
        self._createInferenceParams(npWeights)
        self._createGenerativeParams(npWeights)
        return npWeights
    def _createGenerativeParams(self, npWeights):
        """ Create weights/params for generative model """
        DIM_HIDDEN     = self.params['dim_hidden']
        DIM_STOCHASTIC = self.params['dim_stochastic']
        DIM_HIDDEN_TRANS = DIM_HIDDEN*2
        for l in range(self.params['transition_layers']):
            dim_input,dim_output = DIM_HIDDEN_TRANS, DIM_HIDDEN_TRANS
            if l==0:
                dim_input = self.params['dim_stochastic']
                if self.params['dim_actions']>0.:
                    dim_input+=self.params['dim_actions']
            npWeights['p_trans_W_'+str(l)] = self._getWeight((dim_input, dim_output))
            npWeights['p_trans_b_'+str(l)] = self._getWeight((dim_output,))
        MU_COV_INP = DIM_HIDDEN_TRANS
        if self.params['dim_actions']>0:
            npWeights['p_action_W'] = self._getWeight((self.params['dim_actions'],self.params['dim_actions']))
        npWeights['p_trans_W_mu']       = self._getWeight((MU_COV_INP, self.params['dim_stochastic']))
        npWeights['p_trans_b_mu']       = self._getWeight((self.params['dim_stochastic'],))
        npWeights['p_trans_W_cov']      = self._getWeight((MU_COV_INP, self.params['dim_stochastic']))
        npWeights['p_trans_b_cov']      = self._getWeight((self.params['dim_stochastic'],))
        
        for l in range(self.params['emission_layers']):
            dim_input,dim_output = DIM_HIDDEN, DIM_HIDDEN
            if l==0:
                dim_input = self.params['dim_stochastic']
            npWeights['p_emis_W_'+str(l)] = self._getWeight((dim_input, dim_output))
            npWeights['p_emis_b_'+str(l)] = self._getWeight((dim_output,))
            #0 out the relevant parts based on the indicator functions, multiply by 1 elsewhere
        if self.params['data_type']=='binary':
            npWeights['p_emis_W_ber'] = self._getWeight((self.params['dim_hidden'], self.params['dim_observations']))
            npWeights['p_emis_b_ber'] = self._getWeight((self.params['dim_observations'],))
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
    def _getEmissionFxn(self, z, I = None):
        """
        Apply emission function to zs
        Input:  z [bs x T x dim]
        Output: (params, ) or (mu, cov) of size [bs x T x dim]
        """
        hid = z
        for l in range(self.params['emission_layers']):
                hid = self._LinearNL(self.tWeights['p_emis_W_'+str(l)],  self.tWeights['p_emis_b_'+str(l)], hid)
        if self.params['data_type']=='binary':
            mean_params     = T.nnet.sigmoid(T.dot(hid,self.tWeights['p_emis_W_ber'])+self.tWeights['p_emis_b_ber'])
            #if self.params['dim_indicators']>0:
            #    assert I is not None,'Requires I'
            #    mean_params = mean_params*I
            return [mean_params]
        else:
            assert False,'Invalid type of data'
    
    def _getTransitionFxn(self, z, A=None):
        """
        Apply transition function to zs
        Input:  z [bs x T x dim], u<if actions present in model> [bs x T x dim]
        Output: mu, cov of size [bs x T x dim]
        """
        hid = z
        if self.params['dim_actions']>0:
            embed = T.dot(A,self.tWeights['p_action_W'])
            hid = T.concatenate([embed, hid],axis=2)
        for l in range(self.params['transition_layers']):
            hid = self._LinearNL(self.tWeights['p_trans_W_'+str(l)],self.tWeights['p_trans_b_'+str(l)],hid)
        mu     = T.dot(hid, self.tWeights['p_trans_W_mu']) + self.tWeights['p_trans_b_mu']
        cov    = T.nnet.softplus(T.dot(hid, self.tWeights['p_trans_W_cov'])+self.tWeights['p_trans_b_cov'])
        return mu,cov

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
        else:
            assert False,'Invalid recognition model'
        
    def _qEmbeddingLayer(self, X):
        """ Embed for q """
        return self._LinearNL(self.tWeights['q_W_input_0'],self.tWeights['q_b_input_0'], X)
    #"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""# 
    
    def _inferenceAndReconstruction(self, X, I, A, dropout_prob = 0.):
        """ Returns z_q, mu_q and cov_q """
        self._p('Building with dropout:'+str(dropout_prob))
        embedding          = self._qEmbeddingLayer(X)
        hidden_state       = self._buildLSTM(X, embedding, dropout_prob)
        z_q,mu_q,cov_q     = self._inferenceLayer(hidden_state)
        
        observation_params = self._getEmissionFxn(z_q, I=I)
        mu_trans, cov_trans= self._getTransitionFxn(z_q, A=A)
        mu_prior           = T.concatenate([T.alloc(np.asarray(0.).astype(config.floatX),
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
    
    def _getNegCLL(self, obs_params, X, M, I = None, batchVector = False):
        """
        Estimate the negative conditional log likelihood of x|z under the generative model
        M: mask of size bs x T
        X: target of size bs x T x dim
        """
        mean_p = obs_params[0]
        bce    = T.nnet.binary_crossentropy(mean_p,X)
        if self.params['dim_indicators']>0:
            assert I is not None,'Expect I'
            bce= bce*I
        negCLL = (bce.sum(2)*M).sum(1,keepdims=True)
        if batchVector:
            return negCLL
        else:
            return negCLL.sum()
    
    def resetDataset(self, newX, newI, newA, newM, quiet=False):
        if not quiet:
            ddim,idim,adim,mdim = self.dimData()
            self._p('Original dim:'+str(ddim)+', '+str(idim)+', '+str(adim)+', '+str(mdim))
        self.setData(newX=newX.astype(config.floatX),
                newIndicators=newI.astype(config.floatX),
                newActions=newA.astype(config.floatX),
                newMask=newM.astype(config.floatX))
        if not quiet:
            ddim,idim,adim,mdim = self.dimData()
            self._p('New dim:'+str(ddim)+', '+str(idim)+', '+str(adim)+', '+str(mdim))
    def _buildModel(self):
        if 'synthetic' in self.params['dataset']:
            self.params_synthetic = params_synthetic
        """ High level function to build and setup theano functions """
        idx                = T.vector('idx',dtype='int64')
        idx.tag.test_value = np.array([0,1]).astype('int64')
        self.dataset       = theano.shared(np.random.uniform(0,1,size=(3,5,self.params['dim_observations'])).astype(config.floatX))
        self.indicators    = theano.shared(np.random.uniform(0,1,size=(3,5,self.params['dim_indicators'])).astype(config.floatX))
        self.actions       = theano.shared(np.random.uniform(0,1,size=(3,5,self.params['dim_actions'])).astype(config.floatX))
        self.mask          = theano.shared(np.array(([[1,1,1,1,0],[1,1,0,0,0],[1,1,1,0,0]])).astype(config.floatX))
        X_o                = self.dataset[idx]
        I_o                = self.indicators[idx]
        A_o                = self.actions[idx]
        M_o                = self.mask[idx]
        maxidx             = T.cast(M_o.sum(1).max(),'int64')
        X                  = X_o[:,:maxidx,:]
        I                  = I_o[:,:maxidx,:]
        A                  = A_o[:,:maxidx,:]
        M                  = M_o[:,:maxidx]
        newX,newMask       = T.tensor3('newX',dtype=config.floatX),T.matrix('newMask',dtype=config.floatX)
        newIndicators      = T.tensor3('newIndicators',dtype=config.floatX)
        newActions         = T.tensor3('newActions',dtype=config.floatX)
        self.setData       = theano.function([newX,newIndicators,newActions,newMask],None,
                updates=[(self.dataset,newX),(self.mask,newMask),(self.indicators,newIndicators),(self.actions,newActions)])
        self.dimData       = theano.function([],[self.dataset.shape,self.indicators.shape, self.actions.shape, self.mask.shape])
        
        #Learning Rates and annealing objective function
        #Add them to npWeights/tWeights to be tracked [do not have a prefix _W or _b so wont be differentiated]
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
                                                              X, I, A, dropout_prob = self.params['rnn_dropout'])
            negCLL     = self._getNegCLL(obs_params, X, M, I=I)
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
            optimizer_up +=anneal_update+self.updates
            self._p(str(len(self.updates))+' other updates')
            ############# Setup train & evaluate functions ###########
            self.train_debug         = theano.function(fxn_inputs,[train_cost,norm_list[0],norm_list[1],
                                                                        norm_list[2],negCLL, TemporalKL, anneal.sum()], 
                                                           updates = optimizer_up, name='Train (with Debug)')
        #Updates ack
        self.updates_ack = True
        eval_obs_params, eval_z_q, eval_mu_q, eval_cov_q, eval_mu_prior, eval_cov_prior, \
        eval_mu_trans, eval_cov_trans = self._inferenceAndReconstruction(X, I, A, dropout_prob = 0.)
        eval_z_q.name = 'eval_z_q'
        eval_CNLLvec=self._getNegCLL(eval_obs_params, X, M, I=I, batchVector = True)
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
        
        self.likelihood          = theano.function(fxn_inputs, ll_estimate, name = 'Importance Sampling based likelihood')
        self.evaluate            = theano.function(fxn_inputs, eval_cost, name = 'Evaluate Bound')
        eval_z_q.name = 'z'
        A.name        = 'actions'
        eval_inputs = [eval_z_q]
        if self.params['dim_actions']>0:
            eval_inputs.append(A)
        self.transition_fxn      = theano.function(eval_inputs,[eval_mu_trans, eval_logcov_trans],
                                                       name='Transition Function', allow_input_downcast= True)
        emission_inputs = [eval_z_q]
        #if self.params['dim_indicators']>0:
        #    emission_inputs.append(I)
        self.emission_fxn = theano.function(emission_inputs, eval_obs_params[0], name='Emission Function', allow_input_downcast=True)
        self.posterior_inference = theano.function([X], 
                                                   [eval_z_q, eval_mu_q, eval_logcov_q],
                                                   name='Posterior Inference', allow_input_downcast=True) 
    #"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""# 
if __name__=='__main__':
    """ use this to check compilation for various options"""
    from parse_args_dkf_medical import params
    params['data_type']         = 'binary'
    params['dim_observations']  = 10
    dkf = DKF(params, paramFile = 'tmp')
    os.unlink('tmp')
    import ipdb;ipdb.set_trace()
