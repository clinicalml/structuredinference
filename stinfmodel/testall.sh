THEANO_FLAGS="lib.cnmem=1.,scan.allow_gc=False,compiledir_format=gpu0" python2.7 dkf.py -vm L -infm mean_field -dset jsb >> A.result
THEANO_FLAGS="lib.cnmem=1.,scan.allow_gc=False,compiledir_format=gpu0" python2.7 dkf.py -vm LR -infm mean_field -dset jsb >> A.result
THEANO_FLAGS="lib.cnmem=1.,scan.allow_gc=False,compiledir_format=gpu0" python2.7 dkf.py -vm R -infm structured -dset jsb >> A.result
THEANO_FLAGS="lib.cnmem=1.,scan.allow_gc=False,compiledir_format=gpu0" python2.7 dkf.py -vm R -infm structured -ttype mlp -dset jsb >>A.result
THEANO_FLAGS="lib.cnmem=1.,scan.allow_gc=False,compiledir_format=gpu0" python2.7 dkf.py -vm LR -infm structured -dset jsb >>A.result
THEANO_FLAGS="lib.cnmem=1.,scan.allow_gc=False,compiledir_format=gpu0" python2.7 dkf.py -vm R -infm structured -ar 5000 -dset jsb >>A.result
THEANO_FLAGS="lib.cnmem=1.,scan.allow_gc=False,compiledir_format=gpu0" python2.7 dkf.py -vm R -infm structured -ar 5000 -usenade -dset jsb >>A.result 
THEANO_FLAGS="lib.cnmem=1.,scan.allow_gc=False,compiledir_format=gpu0" python2.7 dkf.py -vm R -infm structured -ar 5000 -etype conditional -usenade -dset jsb >>A.result 
THEANO_FLAGS="lib.cnmem=1.,scan.allow_gc=False,compiledir_format=gpu0" python2.7 dkf.py -vm R -infm structured -ar 5000 -etype conditional -previnp -dset jsb >>A.result 
THEANO_FLAGS="lib.cnmem=1.,scan.allow_gc=False,compiledir_format=gpu0" python2.7 dkf.py -vm R -infm structured -ar 5000 -etype conditional -previnp -usenade -dset jsb >> A.result
grep -n "buildModel" A.result
rm -rf A.result
