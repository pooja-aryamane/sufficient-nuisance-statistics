#!/bin/bash

#SBATCH --job-name=truegroups
##SBATCH --nodes=1
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G  
#SBATCH --gres=gpu:1
##SBATCH --gres=gpu:v100:1
#SBATCH --time=02:00:00


#SBATCH --error=%x.err
#SBATCH --output=%x.out
#SBATCH --mail-type=all            # when something happens
#SBATCH --mail-user=paa9751@nyu.edu # send me mail



singularity exec --nv --overlay /scratch/paa9751/pytorch-example/overlay-25GB-500K.ext3:ro /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash -c "source /ext3/env.sh; python3 /scratch/paa9751/mlhc-project/train_gdro.py --LEARNING_RATE 1e-4 --BATCH_SIZE 256 --K 10000 --MODEL_TYPE 'resnet18' --WEIGHT_DECAY 1 --OUTDIR 'stage2-outdir-newgroups' --GROUP_CNAME 'true_group_idx' --n_groups 4 --PROJECT_NAME 'stage-2-new-sns' --MAX_EPOCHS 200 --TRANSFORM --PRETRAINED --STEP_SIZE 2 --RANDOMSEED 20240607"
