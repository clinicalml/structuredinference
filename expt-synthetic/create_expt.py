import os
import numpy as np
import sys
import itertools
import argparse
parser = argparse.ArgumentParser(description='Create experiments for synthetic data')
parser.add_argument("-s",'--onScreen', type=bool, default=False,help ='create command to run on screen')
args = parser.parse_args()
np.random.seed(1)
## Standard checks
var_model       = ['LR','L','R']
inference_model = ['mean_field','structured']
optimization    = ['adam']
lr              = ['0.0008']
nonlinearity    = ['relu']
datasets        = ['synthetic9','synthetic10']
dim_hidden      = [40]
rnn_layers      = [1]
rnn_size        = [40]
batch_size      = [250]
rnn_dropout     = [0.00001]
#Add the list and it2s name here 
takecrossprod = [inference_model,datasets,optimization, lr, var_model, nonlinearity, dim_hidden, rnn_layers, rnn_size, batch_size, rnn_dropout]
names         = ['infm','dset','opt','lr','vm','nl','dh','rl','rs','bs','rd']

def buildConfig(element,paramnames):
    config = {}
    for idx,paramvalue in enumerate(element): 
        config[paramnames[idx]] = paramvalue 
    return config
def buildParamString(config):
    paramstr = ''
    for paramname in config:
        paramstr += '-'+paramname+' '+str(config[paramname])+' '
    return paramstr

tovis = {} 
for idx, element in enumerate(itertools.product(*takecrossprod)):
    config  = buildConfig(element,names) 
    #Fixed parameters
    config['ep'] = '1500'
    config['sfreq'] = '100'
    gpun = 1
    if idx%2==0:
        gpun = 2
    if 'lr' not in config:
        config['lr']='0.001'
    if 'opt' not in config:
        config['opt']='adam'

    paramstr= buildParamString(config)
    assert 'dset' in config and 'vm' in config and 'infm' in config, 'Expecting dataset, var_model and inference_model to be in config'
    #Savefile to look for in visualize
    savestr = 'dkf_'+config['dset']+'_ep'+config['ep']+'_rs20dh20ds1vm'+config['vm']+'lr'+config['lr']
    savestr+= 'ep'+config['ep']+'opt'+config['opt']
    savestr +='infm'+config['infm']+'synthetic'
    tovis[paramstr.replace(' ','')] = savestr

    cmd = "export CUDA_VISIBLE_DEVICES="+str(gpun-1)+";"
    if args.onScreen:
        randStr = 'THEANO_FLAGS="lib.cnmem=0.4,compiledir_format=compiledir_%(platform)s-%(processor)s-%(python_version)s-%(python_bitwidth)s-'+str(np.random.randint(100))+'"'
        cmd += randStr+ " python2.7 train.py "+paramstr
        name = paramstr.replace(' ','').strip().replace('inference_model','').replace('var_model','').replace('optimization','')
        name = name.replace('-','')
        name = name.replace('synthetic','S').replace('nonlinearity','NL').replace('dataset','')
        print "screen -S "+name+" -t "+name+" -d -m"
        print "screen -r "+name+" -p 0 -X stuff $\'"+cmd+"\\n\'"
        #+"| tee ./checkpointdir_"+config['dataset']+'/'+savestr.replace('t7','log')+"\\n\'"
    else:
        cmd += "python2.7 train.py "+paramstr
        print cmd
