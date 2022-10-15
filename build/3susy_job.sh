#!/bin/bash
#SBATCH -J susy_N3n3
#SBATCH -o svm_SUSY_N3n3.o%j
#SBATCH --ntasks-per-node=3 -N 3
#SBATCH -t 07:30:0
#SBATCH --gpus-per-node=3
#SBATCH --mem-per-cpu=64GB
#SBATCH --distribution=block:block

module load GCC intel
module load cmake
module load cudatoolkit/11.6

echo "*********************************"
echo "*********************************"
echo "SUSY n=5000000 d=18 k=32 qq=1 (N 3 np 3)"
time mpirun ./testsvm "/project/pwu/dataset/SUSY" 1000000 18 32 0.05 1 32 covtype.model