#!/bin/bash

#SBATCH --job-name=samplesplit1
##SBATCH --nodes=1
##SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G  
#SBATCH --gres=gpu:1
##SBATCH --gres=gpu:v100:1
#SBATCH --time=05:00:00


#SBATCH --error=%x.err
#SBATCH --output=%x.out
#SBATCH --mail-type=all            # when something happens
#SBATCH --mail-user=paa9751@nyu.edu # send me mail



singularity exec --nv --overlay /scratch/paa9751/pytorch-example/overlay-25GB-500K.ext3:ro /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash -c "source /ext3/env.sh; python3 /scratch/paa9751/mlhc-project/stage1-training.py --PROJECT_NAME 'stage-1-training-sample-split' --BORDER_SZ 25 --LEARNING_RATE 1e-3 --BATCH_SIZE 256 --WEIGHT_DECAY 1e-2 --MODEL_TYPE 'resnet18' --OUTDIR stage1-outdir --MAX_EPOCHS 100 --SPLIT_IDX 1 --TRANSFORM --PRETRAINED --BORDER_SZ 56"
