#!/bin/bash

#SBATCH --job-name=getresizedchex
##SBATCH --nodes=1
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G  
##SBATCH --gres=gpu:1
##SBATCH --gres=gpu:v100:1
#SBATCH --time=05:00:00


#SBATCH --error=%x.err
#SBATCH --output=%x.out
#SBATCH --mail-type=all            # when something happens
#SBATCH --mail-user=paa9751@nyu.edu # send me mail


singularity exec --nv --overlay /scratch/paa9751/chexpert.sqf:ro --overlay /scratch/paa9751/pytorch-example/overlay-25GB-500K.ext3 --overlay /scratch/paa9751/mimic.sqf:ro /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash -c "source /ext3/env.sh; python3 /scratch/paa9751/mlhc-project/access-new-images.py"
