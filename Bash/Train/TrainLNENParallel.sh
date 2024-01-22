#!/bin/bash
#SBATCH --job-name=BarlowTwinsLNEN
#SBATCH --qos=qos_gpu-t4
#SBATCH --partition=gpu_p2
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=3
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:8
# nombre de taches MPI par noeud
#SBATCH --time=100:00:00   # temps d execution maximum demande (HH:MM:SS)
#SBATCH --output=BarlowTwins_LNEN_z128_harsh_dataaug.out          # nom du fichier de sortie
#SBATCH --error=BarlowTwins_LNEN_z128_harsh_dataaug.error     
#SBATCH --account uli@v100


module load pytorch-gpu/py3/1.9.0
mkdir $SCRATCH/Checkpoint_LNEN_z128
srun python ../../main.py --list-dir $SCRATCH/LNENFilesList/all_tiles_LNEN_inference.txt  --projector 1024-512-256-128 --batch-size 896 --checkpoint-dir $SCRATCH/Checkpoint_LNEN_z128 --parallel
