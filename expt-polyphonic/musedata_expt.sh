#MF-LR
THEANO_FLAGS="lib.cnmem=1.,scan.allow_gc=False,compiledir_format=gpu0" python2.7 train_dkf.py -vm LR -infm mean_field -dset musedata-sorted

#ST-R
THEANO_FLAGS="lib.cnmem=1.,scan.allow_gc=False,compiledir_format=gpu1" python2.7 train_dkf.py -vm R -infm structured -dset musedata-sorted

#ST-R -ttype mlp
THEANO_FLAGS="lib.cnmem=1.,scan.allow_gc=False,compiledir_format=gpu0" python2.7 train_dkf.py -vm R -infm structured -ttype mlp -dset musedata-sorted

#ST-LR
THEANO_FLAGS="lib.cnmem=1.,scan.allow_gc=False,compiledir_format=gpu1" python2.7 train_dkf.py -vm LR -infm structured -dset musedata-sorted

#DKF w/ AR
THEANO_FLAGS="lib.cnmem=1.,scan.allow_gc=False,compiledir_format=gpu0" python2.7 train_dkf.py -vm R -infm structured -ar 5000 -dset musedata-sorted 

#DKF Aug
THEANO_FLAGS="lib.cnmem=1.,scan.allow_gc=False,compiledir_format=gpu1" python2.7 train_dkf.py -vm R -infm structured -ar 5000 -etype conditional -previnp -dset musedata-sorted 

#DKF Aug w/ NADE
THEANO_FLAGS="lib.cnmem=1.,scan.allow_gc=False,compiledir_format=gpu0" python2.7 train_dkf.py -vm R -infm structured -ar 5000 -etype conditional -previnp -usenade -dset musedata-sorted
