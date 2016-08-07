#!/bin/bash
#PBS -l nodes=1:ppn=2:gpus=1:k80
#PBS -l walltime=24:00:00
#PBS -l mem=16GB
#PBS -N piano-MFLR
#PBS -M rahul@cs.nyu.edu
#PBS -j oe

module purge
module load node
module load cmake
module load python/intel/2.7.6
module load numpy/intel/1.9.2
module load hdf5/intel/1.8.12
module load cuda/7.5.18
module load cudnn/7.0

RUNDIR=$SCRATCH/structuredinference/expt-polyphonic
cd $RUNDIR
THEANO_FLAGS="lib.cnmem=0.9,scan.allow_gc=False,compiledir_format=compiledir_format=compiledir_%(platform)s-%(processor)s-%(python_version)s-%(python_bitwidth)s-1" python train_dkf.py -vm LR -infm mean_field -dset piano-sorted
