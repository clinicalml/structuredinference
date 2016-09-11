import os,time,sys
import numpy as np
import fcntl,errno
import socket
sys.path.append('../')
from datasets.load import loadDataset
from utils.misc import removeIfExists,createIfAbsent,mapPrint,saveHDF5,displayTime,getLowestError
from utils.misc import readPickle, loadHDF5, getConfigFile, savePickle
from medical_data.load import loadMedicalData

dataset    = loadMedicalData(setting='A')
reloadFile = 'chkpt-A/DKF_lr-8_0000e-04-vm-R-inf-structured-dh-200-ds-100-nl-relu-bs-256-ep-2000-rs-400-ar-1_0000e+01-rv-5_0000e-02-uid-EP1400-params.npz' 
pfile      = getConfigFile(reloadFile)
params     = readPickle(pfile)[0]

from stinfmodel_medical.dkf import DKF
import stinfmodel_medical.learning as DKF_learn
import stinfmodel_medical.evaluate as DKF_evaluate
dkf  = DKF(params, paramFile = pfile, reloadFile = reloadFile) 

#Visualize in ipynb - display samples, display cfac on test, display 
fname= 'check_evaluation.pkl'
x_sampled, z_sampled  = DKF_evaluate.sample(dkf, dataset['test_act'])

tosave = {}
tosave['x_s'] = x_sampled
tosave['a_s'] = dataset['test_act']

dataCfac     = DKF_evaluate.dataCfac(dkf, dataset['test_obs'], dataset['test_act'], dataset['act_dict'])
modelCfac = DKF_evaluate.modelCfac(dkf, dataset['test_act'])
savePickle([tosave, dataCfac, modelCfac], fname)
print 'Done evaluation'
