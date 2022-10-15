#!/bin/bash
#SBATCH -J cov_N1n4
#SBATCH -o svm_COVTYPE_N1n4.o%j
#SBATCH --ntasks-per-node=4 -N 1
#SBATCH -t 05:30:0
#SBATCH --gpus-per-node=4
#SBATCH --mem-per-cpu=32GB
#SBATCH --distribution=block:block

module load GCC intel
module load cmake
module load cudatoolkit/11.6


echo "*********************************"
echo "*********************************"
echo "COVTYPE n=580864 d=54 k=32 q=1"
time mpirun ./testsvm "/project/pwu/dataset/covtype" 580864 54 32 0.01 1 32 covtype.model