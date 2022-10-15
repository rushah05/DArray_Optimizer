#!/bin/bash
#SBATCH -J cov_N3n3
#SBATCH -o svm_COVTYPE_N3n3.o%j
#SBATCH --ntasks-per-node=3 -N 3
#SBATCH -t 05:00:0
#SBATCH --gpus-per-node=3
# SBATCH --mem-per-cpu=64GB
#SBATCH --distribution=block:block

module load GCC intel
module load cmake
module load cudatoolkit/11.6


echo "*********************************"
echo "*********************************"
echo "COVTYPE n=580863 d=54 k=32 q=1"
time mpirun ./testsvm "/project/pwu/dataset/covtype" 580863 54 32 0.01 1 32 covtype.model