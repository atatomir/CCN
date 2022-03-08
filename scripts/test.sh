#!/bin/sh

#SBATCH --nodes=1
#SBATCH --job-name=test_CCN
#SBATCH --mail-type=ALL
#SBATCH --mail-user=atatomir5@gmail.com
#SBATCH --partition=devel
#SBATCH --gres=gpu:4

source ~/.bashrc
source activate ccn

cd ..
make test
