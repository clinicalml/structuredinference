import os,time,sys
sys.path.append('../')
import numpy as np
from datasets.load import loadDataset
from parse_args_dkf import params 
from utils.misc import removeIfExists,createIfAbsent,mapPrint,saveHDF5,displayTime


if params['dataset']=='':
    params['dataset']='synthetic9'
dataset = loadDataset(params['dataset'])

dataset['train']  = dataset['train'][:params['ntrain']]
params['savedir']+='-'+params['dataset']
createIfAbsent(params['savedir'])

#Saving/loading
for k in ['dim_observations','dim_actions','data_type', 'dim_stochastic']:
    params[k] = dataset[k]
mapPrint('Options: ',params)

#Setup VAE Model (or reload from existing savefile)
start_time = time.time()
from stinfmodel_fast.dkf import DKF
import stinfmodel_fast.evaluate as DKF_evaluate
import stinfmodel_fast.learning as DKF_learn
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
                                replicate_K = 5
                                )
displayTime('Running DKF',start_time, time.time())
#Save file log file
saveHDF5(savef+'-final.h5',savedata)
#import ipdb;ipdb.set_trace()
