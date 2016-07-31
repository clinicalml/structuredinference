#MF-LR
THEANO_FLAGS="lib.cnmem=1.,scan.allow_gc=False,compiledir_format=gpu0" python2.7 train_dkf.py -vm LR -infm mean_field -dset nottingham-sorted
#7.098 7.042 -6.6828
#Look into this -> makes sense? Rerunning on rose2[0]

#ST-R
THEANO_FLAGS="lib.cnmem=1.,scan.allow_gc=False,compiledir_format=gpu1" python2.7 train_dkf.py -vm R -infm structured -dset nottingham-sorted
#7.0145 6.9564 -6.5863

#ST-R -ttype mlp
THEANO_FLAGS="lib.cnmem=1.,scan.allow_gc=False,compiledir_format=gpu0" python2.7 train_dkf.py -vm R -infm structured -ttype mlp -dset nottingham-sorted
#7.0976 7.0421 -6.652

#ST-LR
THEANO_FLAGS="lib.cnmem=1.,scan.allow_gc=False,compiledir_format=gpu1" python2.7 train_dkf.py -vm LR -infm structured -dset nottingham-sorted
#

#DKF w/ AR
THEANO_FLAGS="lib.cnmem=1.,scan.allow_gc=False,compiledir_format=gpu0" python2.7 train_dkf.py -vm R -infm structured -ar 5000 -dset nottingham-sorted 
#6.904 6.854 -6.4320

#DKF Aug
THEANO_FLAGS="lib.cnmem=1.,scan.allow_gc=False,compiledir_format=gpu1" python2.7 train_dkf.py -vm R -infm structured -ar 5000 -etype conditional -previnp -dset nottingham-sorted 
#6.867 6.7935 -6.3199

#DKF Aug w/ NADE
THEANO_FLAGS="lib.cnmem=1.,scan.allow_gc=False,compiledir_format=gpu0" python2.7 train_dkf.py -vm R -infm structured -ar 5000 -etype conditional -previnp -usenade -dset nottingham-sorted
#5.398 5.325 -5.3809
