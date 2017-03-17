import os,time,sys
sys.path.append('../')
from datasets.load import loadDataset
from parse_args_dkf import params 
from utils.misc import removeIfExists,createIfAbsent,mapPrint,saveHDF5,displayTime,getLowestError

if params['dataset']=='':
    params['dataset']='jsb'
dataset = loadDataset(params['dataset'])
params['savedir']+='-'+params['dataset']
createIfAbsent(params['savedir'])

#Saving/loading
for k in ['dim_observations','dim_actions','data_type']:
    params[k] = dataset[k]
mapPrint('Options: ',params)

if params['use_nade']:
    params['data_type']='binary_nade'
#Setup VAE Model (or reload from existing savefile)
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
if os.path.exists(reloadFile):
    pfile=params.pop('paramFile')
    assert os.path.exists(pfile),pfile+' not found. Need paramfile'
    print 'Reloading trained model from : ',reloadFile
    print 'Assuming ',pfile,' corresponds to model'
    dkf  = DKF(params, paramFile = pfile, reloadFile = reloadFile) 
else:
    pfile= params['savedir']+'/'+params['unique_id']+'-config.pkl'
    print 'Training model from scratch. Parameters in: ',pfile
    dkf  = DKF(params, paramFile = pfile)
displayTime('Building dkf',start_time, time.time())

savef     = os.path.join(params['savedir'],params['unique_id']) 
print 'Savefile: ',savef
start_time= time.time()
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
displayTime('Running DKF',start_time, time.time()         )
"""
Load the best DKF based on the validation error
"""
epochMin, valMin, idxMin = getLowestError(savedata['valid_bound'])
reloadFile= pfile.replace('-config.pkl','')+'-EP'+str(int(epochMin))+'-params.npz'
print 'Loading from : ',reloadFile
params['validate_only'] = True
dkf_best  = DKF(params, paramFile = pfile, reloadFile = reloadFile)
additional = {}
savedata['bound_test_best'] = DKF_evaluate.evaluateBound(dkf_best, dataset['test'], dataset['mask_test'], S = 2, batch_size = params['batch_size'], additional =additional) 
savedata['bound_tsbn_test_best'] = additional['tsbn_bound']
savedata['ll_test_best']    = DKF_evaluate.impSamplingNLL(dkf_best, dataset['test'], dataset['mask_test'], S = 2000, batch_size = params['batch_size'])
saveHDF5(savef+'-final.h5',savedata)
print 'Test Bound: ',savedata['bound_test_best'],savedata['bound_tsbn_test_best'],savedata['ll_test_best']
import ipdb;ipdb.set_trace()
