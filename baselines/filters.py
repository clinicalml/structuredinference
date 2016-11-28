from pykalman import KalmanFilter
from pykalman import UnscentedKalmanFilter
from pykalman import AdditiveUnscentedKalmanFilter
import numpy as np

#Running UKF/KF
def runFilter(observations, params, dname, filterType):
    """
    Run Kalman Filter (UKF/KF) and return smoothed means/covariances
    observations : nsample x T
    params       : {'dname'... contains all the necessary parameters for KF/UKF}
    filterType   : 'KF' or 'UKF'
    """
    s1 = set(params[dname].keys())
    s2 = set(['trans_fxn','obs_fxn','trans_cov','obs_cov','init_mu','init_cov',
              'trans_mult','obs_mult','trans_drift','obs_drift','baseline'])
    for k in s2:
        assert k in s1,k+' not found in params'
    #assert s1.issubset(s2) and s1.issuperset(s2),'Missing in params: '+', '.join(list(s2.difference(s1)))
    assert filterType=='KF' or filterType=='UKF','Expecting KF/UKF'
    model,mean,var = None,None,None
    X = observations.squeeze()
    #assert len(X.shape)==2,'observations must be nsamples x T'
    if filterType=='KF':
        def setupArr(arr):
            if type(arr) is np.ndarray:
                return arr
            else:
                return np.array([arr])
        model=KalmanFilter(
                    transition_matrices   = setupArr(params[dname]['trans_mult']),  #multiplier for z_t-1
                    observation_matrices  = setupArr(params[dname]['obs_mult']).T, #multiplier for z_t
                    transition_covariance = setupArr(params[dname]['trans_cov_full']),  #transition cov
                    observation_covariance= setupArr(params[dname]['obs_cov_full']),  #obs cov
                    transition_offsets    = setupArr(params[dname]['trans_drift']),#additive const. in trans
                    observation_offsets   = setupArr(params[dname]['obs_drift']),   #additive const. in obs
                    initial_state_mean    = setupArr(params[dname]['init_mu']),
                    initial_state_covariance = setupArr(params[dname]['init_cov_full']))
    else:
        #In this case, the transition and emission function may have other parameters
        #Create wrapper functions that are instantiated w/ the true parameters
        #and pass them to the UKF
        def trans_fxn(z):
            return params[dname]['trans_fxn'](z, fxn_params = params[dname]['params'])
        def obs_fxn(z):
            return params[dname]['obs_fxn'](z, fxn_params = params[dname]['params'])

        model=AdditiveUnscentedKalmanFilter(
                    transition_functions  = trans_fxn, #params[dname]['trans_fxn'],
                    observation_functions = obs_fxn,  #params[dname]['obs_fxn'],
                    transition_covariance = np.array([params[dname]['trans_cov']]),  #transition cov
                    observation_covariance= np.array([params[dname]['obs_cov']]),  #obs cov
                    initial_state_mean    = np.array([params[dname]['init_mu']]),
                    initial_state_covariance = np.array(params[dname]['init_cov']))
    #Run smoothing algorithm with model
    dim_stoc = params[dname]['dim_stoc']
    if dim_stoc>1:
        mus     = np.zeros((X.shape[0],X.shape[1],dim_stoc))
        cov     = np.zeros((X.shape[0],X.shape[1],dim_stoc))
    else:
        mus     = np.zeros(X.shape)
        cov     = np.zeros(X.shape)
    ll      = 0
    for n in range(X.shape[0]):
        (smoothed_state_means, smoothed_state_covariances) = model.smooth(X[n,:])
        if dim_stoc>1:
            mus[n,:] = smoothed_state_means
            cov[n,:] = np.concatenate([np.diag(k)[None,:] for k in smoothed_state_covariances],axis=0)
        else:
            mus[n,:] = smoothed_state_means.ravel()
            cov[n,:] = smoothed_state_covariances.ravel()
        if filterType=='KF':
            ll      += model.loglikelihood(X[n,:])
    return mus,cov,ll

#Generating Data
def sampleGaussian(mu,cov):
    """
    Sample from gaussian with mu/cov
    
    returns: random sample from N(mu,cov) of shape mu
    mu: must be numpy array
    cov: can be scalar or same shape as mu
    """
    return np.multiply(np.random.randn(*mu.shape),np.sqrt(cov))+mu

def generateData(N,T,params, dname):
    """
    Generate sequential dataset
    returns: N x T matrix of observations, latents
    N      : #samples
    T      : time steps
    params : {'dname'...contains necessary functions}
    dname  : dataset name
    """
    np.random.seed(1)
    assert dname in params,dname+' not found in params'
    Z = np.zeros((N,T))
    X = np.zeros((N,T))
    Z[:,0] = sampleGaussian(params[dname]['init_mu']*np.ones((N,)),
                params[dname]['init_cov'])
    for t in range(1,T):
        Z[:,t] = sampleGaussian(params[dname]['trans_fxn'](Z[:,t-1]),params[dname]['trans_cov'])
    for t in range(T):
        X[:,t] = sampleGaussian(params[dname]['obs_fxn'](Z[:,t]),params[dname]['obs_cov'])
    return X,Z

#Reconstruction
def reconsMus(mus_posterior, params, dname):
    """
    Estimate the observation means using posterior means
    mus_posterior : N x T matrix of posterior means
    params : {'dname'...contains necessary functions}
    """
    mu_rec   = np.zeros(mus_posterior.shape)
    for t in range(mus_posterior.shape[1]):
        mu_rec[:,t] = params[dname]['obs_fxn'](mus_posterior[:,t])
    return mu_rec
