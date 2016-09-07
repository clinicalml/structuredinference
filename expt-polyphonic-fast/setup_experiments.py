"""
Rahul G. Krishnan

Script to setup experiments either on HPC or individually 
"""
import numpy as np
from collections import OrderedDict
import argparse,os

parser = argparse.ArgumentParser(description='Setup Expts')
parser.add_argument('-hpc','--onHPC',action='store_true') 
parser.add_argument('-dset','--dataset', default='jsb',action='store')
parser.add_argument('-ngpu','--num_gpus', default=4,action='store',type=int)
args = parser.parse_args()

#MAIN FLAGS
onHPC        = args.onHPC
DATASET      = args.dataset 
THFLAGS      = 'THEANO_FLAGS="lib.cnmem=1.,scan.allow_gc=False,compiledir_format=gpu<rand_idx>" ' 

#Get dataset
dataset      = DATASET.split('-')[0]
all_datasets = ['jsb','piano','nottingham','musedata']
assert dataset in all_datasets,'Dset not found: '+dataset
all_expts    = OrderedDict()
for dset in all_datasets:
    all_expts[dset] = OrderedDict()

#Experiments to run for each dataset
all_expts['jsb']['ST-R'] = 'python2.7 train_dkf.py -vm R -infm structured -dset <dataset>'
all_expts['jsb']['MF-LR'] = 'python2.7 train_dkf.py -vm LR -infm mean_field -dset <dataset>'
all_expts['jsb']['ST-LR'] = 'python2.7 train_dkf.py -vm LR -infm structured -dset <dataset>'
all_expts['jsb']['ST-R-mlp'] = 'python2.7 train_dkf.py -vm R -infm structured -ttype mlp -dset <dataset>'
all_expts['jsb']['ST-L'] = 'python2.7 train_dkf.py -vm L -infm structured -dset <dataset>'
all_expts['jsb']['DKF-ar'] ='python2.7 train_dkf.py -vm R -infm structured -ar 5000 -dset <dataset>' 
all_expts['jsb']['DKF-aug'] ='python2.7 train_dkf.py -vm R -infm structured -ar 5000 -etype conditional -previnp -dset <dataset>'
all_expts['jsb']['DKF-aug-nade'] ='python2.7 train_dkf.py -vm R -infm structured -etype conditional -previnp -usenade -dset <dataset>'

all_expts['nottingham']['ST-R'] = 'python2.7 train_dkf.py -vm R -infm structured -dset <dataset>'
all_expts['nottingham']['MF-LR'] = 'python2.7 train_dkf.py -vm LR -infm mean_field -dset <dataset>'
all_expts['nottingham']['ST-LR'] = 'python2.7 train_dkf.py -vm LR -infm structured -dset <dataset>'
all_expts['nottingham']['ST-R-mlp'] = 'python2.7 train_dkf.py -vm R -infm structured -ttype mlp -dset <dataset>'
all_expts['nottingham']['ST-L'] = 'python2.7 train_dkf.py -vm L -infm structured -dset <dataset>'
all_expts['nottingham']['DKF-ar'] ='python2.7 train_dkf.py -vm R -infm structured -ar 5000 -dset <dataset>' 
all_expts['nottingham']['DKF-aug'] ='python2.7 train_dkf.py -vm R -infm structured -ar 5000 -etype conditional -previnp -dset <dataset>'
all_expts['nottingham']['DKF-aug-nade'] ='python2.7 train_dkf.py -vm R -infm structured -ar 1000 -etype conditional -previnp -usenade -dset <dataset>'

all_expts['musedata']['ST-R'] = 'python2.7 train_dkf.py -vm R -infm structured -dset <dataset>'
all_expts['musedata']['MF-LR'] = 'python2.7 train_dkf.py -vm LR -infm mean_field -dset <dataset>'
all_expts['musedata']['ST-LR'] = 'python2.7 train_dkf.py -vm LR -infm structured -dset <dataset>'
all_expts['musedata']['ST-R-mlp'] = 'python2.7 train_dkf.py -vm R -infm structured -ttype mlp -dset <dataset>'
all_expts['musedata']['ST-L'] = 'python2.7 train_dkf.py -vm L -infm structured -dset <dataset>'
all_expts['musedata']['DKF-ar'] ='python2.7 train_dkf.py -vm R -infm structured -ar 5000 -dset <dataset>' 
all_expts['musedata']['DKF-aug'] ='python2.7 train_dkf.py -vm R -infm structured -ar 5000 -etype conditional -previnp -dset <dataset>'
all_expts['musedata']['DKF-aug-nade'] ='python2.7 train_dkf.py -vm R -infm structured -ar 1000 -etype conditional -previnp -usenade -dset <dataset> -ds 50 -dh 100 -rs 400'

all_expts['piano']['ST-R'] = 'python2.7 train_dkf.py -vm R -infm structured -dset <dataset>'
all_expts['piano']['MF-LR'] = 'python2.7 train_dkf.py -vm LR -infm mean_field -dset <dataset>'
all_expts['piano']['ST-LR'] = 'python2.7 train_dkf.py -vm LR -infm structured -dset <dataset>'
all_expts['piano']['ST-R-mlp'] = 'python2.7 train_dkf.py -vm R -infm structured -ttype mlp -dset <dataset>'
all_expts['piano']['ST-L'] = 'python2.7 train_dkf.py -vm L -infm structured -dset <dataset>'
all_expts['piano']['DKF-ar'] ='python2.7 train_dkf.py -vm R -infm structured -ar 5000 -dset <dataset>' 
all_expts['piano']['DKF-aug'] ='python2.7 train_dkf.py -vm R -infm structured -ar 5000 -etype conditional -previnp -dset <dataset>'
all_expts['piano']['DKF-aug-nade'] ='python2.7 train_dkf.py -vm R -infm structured -ar 1000 -etype conditional -previnp -usenade -dset <dataset> -ds 50 -dh 100 -rs 400'

if onHPC:
    DIR = './hpc_'+dataset
    os.system('rm -rf '+DIR) 
    os.system('mkdir -p '+DIR)
    with open('template.q') as ff:
        template = ff.read()
    runallcmd    = ''
    for name in all_expts[dataset]:
        runcmd  = all_expts[dataset][name].replace('<dataset>',DATASET)+' -uid '+name
        command = THFLAGS.replace('<rand_idx>',str(np.random.randint(args.num_gpus)))+runcmd
        with open(DIR+'/'+name+'.q','w') as f:
            f.write(template.replace('<name>',name).replace('<command>',command))
        print 'Wrote to:',DIR+'/'+name+'.q'
        runallcmd+= 'qsub '+name+'.q\n'
    with open(DIR+'/runall.sh','w') as f:
        f.write(runallcmd)
else:
    for name in all_expts[dataset]:
        runcmd  = all_expts[dataset][name].replace('<dataset>',DATASET)+' -uid '+name
        command = THFLAGS.replace('<rand_idx>',str(np.random.randint(args.num_gpus)))+runcmd
        print command
