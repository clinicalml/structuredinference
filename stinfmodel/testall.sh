THEANO_FLAGS="lib.cnmem=1.,scan.allow_gc=False,compiledir_format=gpu0" python2.7 dkf.py -vm L -infm mean-field -dset jsb
THEANO_FLAGS="lib.cnmem=1.,scan.allow_gc=False,compiledir_format=gpu0" python2.7 dkf.py -vm LR -infm mean-field -dset jsb
THEANO_FLAGS="lib.cnmem=1.,scan.allow_gc=False,compiledir_format=gpu0" python2.7 dkf.py -vm R -infm structured -dset jsb
THEANO_FLAGS="lib.cnmem=1.,scan.allow_gc=False,compiledir_format=gpu0" python2.7 dkf.py -vm R -infm structured -ttype mlp -dset jsb
THEANO_FLAGS="lib.cnmem=1.,scan.allow_gc=False,compiledir_format=gpu0" python2.7 dkf.py -vm LR -infm structured -dset jsb
THEANO_FLAGS="lib.cnmem=1.,scan.allow_gc=False,compiledir_format=gpu0" python2.7 dkf.py -vm R -infm structured -ar 5000 -dset jsb
THEANO_FLAGS="lib.cnmem=1.,scan.allow_gc=False,compiledir_format=gpu0" python2.7 dkf.py -vm R -infm structured -ar 5000 -etype conditional -previnp -dset jsb 
THEANO_FLAGS="lib.cnmem=1.,scan.allow_gc=False,compiledir_format=gpu0" python2.7 dkf.py -vm R -infm structured -ar 5000 -etype conditional -previnp -usenade -dset jsb
