#!/bin/sh
#SBATCH -J Carlo-GNNs
#SBATCH -n 1
#SBATCH -p gpu_h100
#SBATCH -t 0-16:00:00
#SBATCH --cpus-per-task 8
#SBATCH --gpus 3
#SBATCH --mem-per-gpu=50G
#SBATCH --mail-type=END
#SBATCH --mail-user=c.saccardi@tudelft.nl
#SBATCH --output=job-50masking-%j.log   # Save stdout to job-<jobid>.log
#SBATCH --error=job-50masking-%j.err    # Save stderr to job-<jobid>.err

## Explanation about SLURM parameters
# -J job-name
# -t time D-HH:MM:SS
# -p partition
# -n number of nodes
# --ntasks-per-node=n
# --cpus-per-task=n
# --mem=memory
# --gpus number of gpus
# --mail-type=type
# --mail-user=user

# For more info about partitions and SLURM variables go to:
# https://servicedesk.surf.nl/wiki/display/WIKI/Snellius+partitions+and+accounting
# https://servicedesk.surf.nl/wiki/display/WIKI/Writing+a+job+script

# Load any required modules (if needed)
module load 2023
# Source the conda initialization script from the correct path.
source /sw/arch/RHEL8/EB_production/2023/software/Miniconda3/23.5.2-0/etc/profile.d/conda.sh
# Activate the "carlo" environment.
conda activate carlo

# Prepend the activated environment’s bin directory to PATH.
export PATH="$CONDA_PREFIX/bin:$PATH"

# Diagnostic prints to verify correct environment activation.
echo "Active environment: $CONDA_DEFAULT_ENV"
echo "Using python: $(which python)"
python --version

# Run your training script.
export WANDB_API_KEY=e0e7c4a64ff0915a3fe22c7e59a2b5cb82e00322
python train_mask.py --config="configs/highRes_only/MAE3.yaml"


