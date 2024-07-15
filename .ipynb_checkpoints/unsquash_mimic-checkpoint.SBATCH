#!/bin/bash

#SBATCH --job-name=unsquashfs-mimic
##SBATCH --nodes=1
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G  
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00

    
#SBATCH --error=%x.err
#SBATCH --output=%x.out
#SBATCH --mail-type=all            # when something happens
#SBATCH --mail-user=paa9751@nyu.edu # send me mail



singularity exec --nv --overlay /scratch/paa9751/pytorch-example/overlay-25GB-500K.ext3:ro /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash -c "source /ext3/env.sh; unsquashfs -d mimic-new /scratch/apm470/datasets_sqfs/mimic.sqf"

