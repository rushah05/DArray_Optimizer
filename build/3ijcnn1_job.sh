#!/bin/bash
#SBATCH -J ijcnn1_N3n3
#SBATCH -o svm_IJCNN_N3n3.o%j
#SBATCH --ntasks-per-node=3 -N 3
#SBATCH -t 0:3:0
#SBATCH --gpus-per-node=3
#SBATCH --mem-per-cpu=32GB
#SBATCH --distribution=block:block
#SBATCH -A nvtsekos

module load GCC intel
module load cmake
module load cudatoolkit/11.6


echo "*********************************"
echo "*********************************"
echo "ijcnn1 n=49990 d=22 k=32 qq=1 (N 1 np 1)"
time mpirun ./testsvm "/project/pwu/dataset/ijcnn1" 49990 22 32 0.045 1 100

