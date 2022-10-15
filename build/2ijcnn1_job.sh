#!/bin/bash
#SBATCH -J ijcnn1_N2n2
#SBATCH -o svm_IJCNN_N2n2.o%j
#SBATCH --ntasks-per-node=2 -N 2
#SBATCH -t 0:3:0
#SBATCH --gpus-per-node=2
#SBATCH --mem-per-cpu=32GB
#SBATCH --distribution=block:block

module load GCC intel
module load cmake
module load cudatoolkit/11.6


echo "*********************************"
echo "*********************************"
echo "ijcnn1 n=49990 d=22 k=32 qq=1 (N 1 np 1)"
time mpirun ./testsvm "/project/pwu/dataset/ijcnn1" 49990 22 32 0.05 1 100