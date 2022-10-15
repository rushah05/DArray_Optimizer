#!/bin/bash
#SBATCH -J susy_N4n4
#SBATCH -o svm_SUSY_N4n4_1m.o%j
#SBATCH --ntasks-per-node=4 -N 4
#SBATCH -t 07:30:0
#SBATCH --gpus-per-node=4
#SBATCH --mem-per-cpu=64GB
#SBATCH --distribution=block:block

module load GCC intel
module load cmake
module load cudatoolkit/11.6

echo "*********************************"
echo "*********************************"
echo "SUSY n=1000000 d=18 k=500 qq=1 (N 4 np 4)"
time mpirun ./testsvm "/project/pwu/dataset/SUSY" 1000000 18 500 0.05 1 32 covtype.model