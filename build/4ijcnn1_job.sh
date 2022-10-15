#!/bin/bash
#SBATCH -J ijcnn1_N4n4
#SBATCH -o svm_IJCNN_N4n4.o%j
#SBATCH --ntasks-per-node=4 -N 4
#SBATCH -t 0:3:0
#SBATCH --gpus-per-node=4
#SBATCH --mem-per-cpu=32GB
#SBATCH --distribution=block:block

module load GCC intel
module load cmake
module load cudatoolkit/11.6


echo "*********************************"
echo "*********************************"
echo "ijcnn1 n=49990 d=22 k=256 qq=1 (N 1 np 1)"
time mpirun ./testsvm "/project/pwu/dataset/ijcnn1" 10000 22 256 0.05 1 100 ijcnn4.model