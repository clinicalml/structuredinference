from utils.misc import getPYDIR,loadHDF5
import numpy as np
import os

def loadMedicalData(setting = 'A'):
    """
    Need a good way to consider different configurations (with and without actions)
    Settings:
    A: X (observations) I (indicators) A (actions)      #
    B: X (observations|indicators) I (None) A (actions) #Actions on entire observations
    C: X (observations|indicators) I (None) A (None)    #Only for density estimation
    """
    assert os.path.exists('/m/users/rahul'),'Not on the medical machines. Aborting.'
    suffix = 'filtered3months_split'
    DIR = os.path.dirname(os.path.realpath(__file__)).split('medical_data')[0]+'medical_data/'
    dataset = loadHDF5(DIR+'data_'+suffix+'.h5') 
    dataset['act_dict'] = [t.strip() for t in open(DIR+'action_dictionary_'+suffix+'.txt').readlines()]
    dataset['ind_dict'] = [t.strip() for t in open(DIR+'indicator_dictionary_'+suffix+'.txt').readlines()]
    dataset['obs_dict'] = [t.strip() for t in open(DIR+'observation_dictionary_'+suffix+'.txt').readlines()]
    dataset['mids']     = np.loadtxt(DIR+'patient_list_'+suffix+'.txt',dtype=str)
    if setting == 'A':
        dataset['dim_observations'] = dataset['train_obs'].shape[2]
        dataset['dim_indicators']   = dataset['train_ind'].shape[2]
        dataset['dim_actions']      = dataset['train_act'].shape[2]
    elif setting=='B':
        dataset['train_obs']        = np.concatenate([dataset['train_obs'],dataset['train_ind']],axis=2)
        dataset['valid_obs']        = np.concatenate([dataset['valid_obs'],dataset['valid_ind']],axis=2)
        dataset['test_obs']         = np.concatenate([dataset['test_obs'],dataset['test_ind']],axis=2)
        dataset['dim_observations'] = dataset['train_obs'].shape[2]
        dataset['train_ind']        *= 0.
        dataset['valid_ind']        *= 0. 
        dataset['test_ind']         *= 0.
        dataset['dim_indicators']   = 0 
        dataset['obs_dict']         = dataset['obs_dict']+dataset['ind_dict']
        dataset['ind_dict']         = []
        dataset['dim_actions']      = dataset['train_act'].shape[2]
    elif setting=='C':
        dataset['train_obs']        = np.concatenate([dataset['train_obs'],dataset['train_ind']],axis=2)
        dataset['valid_obs']        = np.concatenate([dataset['valid_obs'],dataset['valid_ind']],axis=2)
        dataset['test_obs']         = np.concatenate([dataset['test_obs'],dataset['test_ind']],axis=2)
        dataset['dim_observations'] = dataset['train_obs'].shape[2]
        dataset['train_ind']        *= 0.
        dataset['valid_ind']        *= 0. 
        dataset['test_ind']         *= 0.
        dataset['train_act']        *= 0.
        dataset['valid_act']        *= 0. 
        dataset['test_act']         *= 0.
        dataset['dim_indicators']   = 0 
        dataset['dim_actions']      = 0 
        dataset['obs_dict']         = dataset['obs_dict']+dataset['ind_dict']
        dataset['ind_dict']         = []
        dataset['act_dict']         = []
    else:
        assert False,'Invalid setting :'+str(setting)
    #All 1's mask
    dataset['train_mask'] = np.ones_like(dataset['train_obs'][:,:,0]) 
    dataset['valid_mask'] = np.ones_like(dataset['valid_obs'][:,:,0]) 
    dataset['test_mask']  = np.ones_like(dataset['test_obs'][:,:,0]) 
    dataset['data_type']  = 'binary'
    return dataset

if __name__=='__main__':
    dset = loadMedicalData() 
    import ipdb;ipdb.set_trace()
