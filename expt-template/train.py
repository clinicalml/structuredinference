import os,time,sys

""" Add the higher level directory to PYTHONPATH to be able to access the models """
sys.path.append('../')

""" Change this to modify the loadDataset function """
from load import loadDataset

""" 
This will contain a hashmap where the 
parameters correspond to the default ones modified
by any command line options given to this script
"""
from parse_args_dkf import params 

""" Some utility functions from theanomodels """
from utils.misc import removeIfExists,createIfAbsent,mapPrint,saveHDF5,displayTime


""" Load the dataset into a hashmap. See load.py for details  """
dataset = loadDataset()
params['savedir']+='-template'
createIfAbsent(params['savedir'])

""" Add dataset and NADE parameters to "params"
    which will become part of the model
"""
for k in ['dim_observations','data_type']:
    params[k] = dataset[k]
mapPrint('Options: ',params)
if params['use_nade']:
    params['data_type']='binary_nade'

"""
import DKF + learn/evaluate functions
"""
start_time = time.time()
from stinfmodel.dkf import DKF 
import stinfmodel.learning as DKF_learn
import stinfmodel.evaluate as DKF_evaluate
displayTime('import DKF',start_time, time.time())
dkf    = None
 
#Remove from params
start_time = time.time()
removeIfExists('./NOSUCHFILE')
reloadFile = params.pop('reloadFile')
""" Reload parameters if reloadFile exists otherwise setup model from scratch
and initialize parameters randomly.
"""
if os.path.exists(reloadFile):
    pfile=params.pop('paramFile')
    """ paramFile is set inside the BaseClass in theanomodels 
    to point to the pickle file containing params"""
    assert os.path.exists(pfile),pfile+' not found. Need paramfile'
    print 'Reloading trained model from : ',reloadFile
    print 'Assuming ',pfile,' corresponds to model'
    dkf  = DKF(params, paramFile = pfile, reloadFile = reloadFile) 
else:
    pfile= params['savedir']+'/'+params['unique_id']+'-config.pkl'
    print 'Training model from scratch. Parameters in: ',pfile
    dkf  = DKF(params, paramFile = pfile)
displayTime('Building dkf',start_time, time.time())

"""Set save prefix"""
savef     = os.path.join(params['savedir'],params['unique_id']) 
print 'Savefile: ',savef
start_time= time.time()

"""Learn the model (see stinfmodel/learning.py)"""
savedata = DKF_learn.learn(dkf, dataset['train'], dataset['mask_train'], 
                                epoch_start =0 , 
                                epoch_end = params['epochs'], 
                                batch_size = params['batch_size'],
                                savefreq   = params['savefreq'],
                                savefile   = savef,
                                dataset_eval=dataset['valid'],
                                mask_eval  = dataset['mask_valid'],
                                replicate_K= params['replicate_K'],
                                shuffle    = False
                                )
displayTime('Running DKF',start_time, time.time())

""" Evaluate bound on test set (see stinfmodel/evaluate.py)"""
savedata['bound_test'] = DKF_evaluate.evaluateBound(dkf, dataset['test'], dataset['mask_test'], 
                                           batch_size = params['batch_size'])
saveHDF5(savef+'-final.h5',savedata)
print 'Test Bound: ',savedata['bound_test']
import ipdb;ipdb.set_trace()
