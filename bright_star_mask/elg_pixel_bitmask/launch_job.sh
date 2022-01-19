#!/bin/bash
#SBATCH --qos=regular
#SBATCH --nodes=4
#SBATCH --ntasks-per-node 1
#SBATCH --constraint=haswell
#SBATCH --time=4:00:00
#SBATCH --account=desi

srun --wait=0 payload.sh tasks.txt
