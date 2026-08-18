#!/bin/bash
# Explication : Spécifie que ce script doit être exécuté par le shell Bash.

# ---------------------------------------------------------------------
# DIRECTIVES SLURM (Paramétrage des ressources du cluster HPC)
# ---------------------------------------------------------------------

#SBATCH --job-name=bPC_train
# Explication : Donne un nom à votre job pour le repérer facilement dans la file d'attente (via la commande 'squeue').

#SBATCH --output=logs/%x_%j.out
# Explication : Fichier où sera écrite la sortie standard (vos 'print'). %x est remplacé par le nom du job, %j par son ID. Assurez-vous que le dossier 'logs' existe.

#SBATCH --error=logs/%x_%j.err
# Explication : Fichier où seront écrites les erreurs. Très utile pour débugger si le script plante.

#SBATCH -C a100
# Explication : Demande à utiliser la partition (file d'attente) dédiée aux GPUs. À adapter selon les noms configurés sur votre cluster (ex: 'gpu_p13', 'rtx3090', etc.).

#SBATCH --nodes=1
# Explication : Nombre de nœuds de calcul demandés. Pour un entraînement sur un seul GPU ou multi-GPU sur une même machine physique, on laisse 1.

#SBATCH --ntasks=1
# Explication : Nombre de tâches (processus). Pour du PyTorch standard sans distribution complexe (DDP multinœuds), on garde 1.

#SBATCH --cpus-per-task=8
# Explication : Nombre de cœurs CPU alloués. Important si vous augmentez le 'num_workers' de vos DataLoaders PyTorch pour charger les images plus vite.

##SBATCH --gres=gpu:1
# Explication : Demande l'allocation d'un GPU (Generic Resource). Si vous visez une carte spécifique, cela peut devenir '--gres=gpu:v100:1' ou '--gres=gpu:a100:1'.

#SBATCH --time=12:00:00
# Explication : Temps maximum alloué pour le calcul (Format HH:MM:SS). Si l'entraînement dépasse cette durée, il sera coupé automatiquement par le cluster.

#SBATCH --mem=32G
# Explication : Quantité de mémoire vive CPU (RAM) allouée au job. Attention, ce n'est pas la mémoire VRAM de la carte graphique.

# ---------------------------------------------------------------------
# PRÉPARATION DE L'ENVIRONNEMENT
# ---------------------------------------------------------------------

# Explication : Nettoie l'environnement pour s'assurer qu'aucun module indésirable n'est chargé par défaut.
#module purge

# Explication : Charge les modules nécessaires. Les versions varient selon les clusters. 
# Utilisez la commande 'module avail' sur le terminal de votre cluster pour trouver les noms exacts.
module load conda/25.9.1
module load python/3.12.12
module load cuda-toolkit/12.9.1

# Explication : Activation de votre environnement virtuel ou Conda. (Décommentez la ligne correspondant à votre installation).
# source /chemin/vers/votre/environnement/venv/bin/activate
# ou pour conda :
conda activate torch_env

# ---------------------------------------------------------------------
# EXÉCUTION DU SCRIPT PYTHON
# ---------------------------------------------------------------------

# Explication : Affiche quelques informations de diagnostic dans le fichier de log (.out).
echo "========================================="
echo "Début du job sur le noeud : $SLURM_NODELIST"
echo "Date de début : $(date)"
echo "========================================="
module purge

# 2. LA CORRECTION C++ : On force Linux à utiliser les librairies de ton environnement Conda
export LD_LIBRARY_PATH="/home/ppicard/.conda/envs/torch_env/lib:$LD_LIBRARY_PATH"
# ── LA CORRECTION VRAM : On interdit formellement la pré-allocation XLA/JAX ──
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
export TF_FORCE_GPU_ALLOW_GROWTH=true
# Authentification silencieuse pour WandB
# Explication : Exécution du script d'entraînement. 
# L'option '-u' (unbuffered) est cruciale sur HPC : elle permet d'écrire les 'print' instantanément dans le fichier log au lieu d'attendre la fin de l'exécution.
python -u src/pcn/train_bPC.py                   


# Explication : Trace de fin pour confirmer que le job ne s'est pas coupé en plein milieu.
echo "========================================="
echo "Fin du job : $(date)"
echo "========================================="
