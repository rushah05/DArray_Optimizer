#!/bin/bash
#SBATCH -J ijcnn1_N1n4
#SBATCH -o svm_IJCNN_N1n4.o%j
#SBATCH --ntasks-per-node=4 -N 1
#SBATCH -t 01:10:0
#SBATCH --gpus-per-node=4
#SBATCH --mem-per-cpu=64GB
#SBATCH --distribution=block:block

module load GCC intel
module load cmake
module load cudatoolkit/11.6


echo "*********************************"
echo "*********************************"
echo "ijcnn1 n=49990 d=22 k=200 qq=1 (N 1 np 4)"
time mpirun ./testsvm "/project/pwu/dataset/ijcnn1" 49990 22 200 0.0455 1 100 ijcnn1.model

# echo "*********************************"
# echo "*********************************"
# echo "ijcnn1 n=49990 d=22 k=400 qq=1 (N 2 np 2)"
# time mpirun ./testsvm "/project/pwu/dataset/ijcnn1" 49990 22 400 0.0455 1 100

# echo "*********************************"
# echo "*********************************"
# echo "ijcnn1 n=49990 d=22 k=600 qq=1 (N 2 np 2)"
# time mpirun ./testsvm "/project/pwu/dataset/ijcnn1" 49990 22 600 0.0455 1 100


# echo "*********************************"
# echo "*********************************"
# echo "ijcnn1 n=49990 d=22 k=800 qq=1 (N 2 np 2)"
# time mpirun ./testsvm "/project/pwu/dataset/ijcnn1" 49990 22 800 0.0455 1 100


