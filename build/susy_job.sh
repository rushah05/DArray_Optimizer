#!/bin/bash
#SBATCH -J susy_N4n4
#SBATCH -o svm_SUSY_N4n4.o%j
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
echo "SUSY n=5000000 d=18 k=32 qq=1 (N 1 np 4)"
time mpirun ./testsvm "/project/pwu/dataset/SUSY" 5000000 18 32 0.05 1 32