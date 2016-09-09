import os,time,sys
import fcntl,errno
import socket
sys.path.append('../')
from datasets.load import loadDataset
from parse_args_dkf_medical import params 
from utils.misc import removeIfExists,createIfAbsent,mapPrint,saveHDF5,displayTime,getLowestError
from medical_data.load import loadMedicalData

dataset = loadMedicalData(setting=params['dataset'])
params['savedir']+='-'+params['dataset']
createIfAbsent(params['savedir'])

#Saving/loading
for k in ['dim_observations','dim_indicators','dim_actions','data_type']:
    params[k] = dataset[k]
mapPrint('Options: ',params)

start_time = time.time()
from stinfmodel_medical.dkf import DKF
import stinfmodel_medical.learning as DKF_learn
import stinfmodel_medical.evaluate as DKF_evaluate
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
savedata = DKF_learn.learn(dkf, dataset['train_obs'], dataset['train_ind'], dataset['train_act'], dataset['train_mask'], 
                                epoch_start =0 , 
                                epoch_end = params['epochs'], 
                                batch_size = params['batch_size'],
                                savefreq   = params['savefreq'],
                                savefile   = savef,
                                dataset_eval=dataset['valid_obs'],
                                indicators_eval=dataset['valid_ind'],
                                actions_eval=dataset['valid_act'],
                                mask_eval  = dataset['valid_mask'],
                                replicate_K= params['replicate_K'],
                                shuffle    = False
                                )
displayTime('Running DKF',start_time, time.time()         )
import ipdb;ipdb.set_trace()
