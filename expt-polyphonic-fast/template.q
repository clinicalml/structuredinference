#!/bin/bash
#PBS -l nodes=1:ppn=2:gpus=1:k80
#PBS -l walltime=30:00:00
#PBS -l mem=16GB
#PBS -N <name>
#PBS -M rahul@cs.nyu.edu
#PBS -j oe

module purge
module load node
module load cmake
module load python/intel/2.7.6
module load numpy/intel/1.9.2
module load hdf5/intel/1.8.12
module load cuda/7.5.18
module load cudnn/7.5v5.1

RUNDIR=$SCRATCH/structuredinference/expt-polyphonic-fast
cd $RUNDIR
<command>
