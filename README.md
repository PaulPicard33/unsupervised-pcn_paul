# Run experiments on PLaFRIM

Prerequisite: create an account on wandb.ai and retrieve your API key (in User Settings)

## Steps 
* Create an account on PLaFRIM
* Connect to PLaFRIM using Git Bash
```
ssh username@plafrim
```
* Load conda
```
module load build/conda/4.10
```
* Create conda environment
```
conda create --name torch_env python=3.11
```
* Activate environment
```
conda activate torch_env
```
* Clone Github repo
```
module load tools/git/2.36.0
git clone https://github.com/fontaluc/unsupervised-pcn.git
```
* Install pytorch using the official Pytorch index and not using PyPI with requirements.txt because CUDA-specific versions of CUDA are not listed in PyPI
```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```
* Install project as package (in editable mode)
```
cd unsupervised-pcn
pip install -e .
```
If you do not intend to change the code from the package, you can remove "-e". 
* Set the WANDB_API_KEY environment variable to your API key in .bashrc and apply changes
```
export WANDB_API_KEY=<your_api_key>
```
The .bashrc file in your home directory should look like:
```
# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
        . /etc/bashrc
fi

# Uncomment the following line if you don't like systemctl's auto-paging feature:
# export SYSTEMD_PAGER=

# User specific aliases and functions
module load tools/git/2.36.0
export WANDB_API_KEY=<your_api_key>
```
We are loading git in the .bashrc file so that we don't need to load it every time we log on plafrim. 
* Apply changes
```
source ~/.bashrc
```
* Create dataset
```
python src/pcn/make_mnist.py
```
* Launch job on PLaFRIM
```
sbatch hpc/train_PC.sh
sbatch hpc/eval_PC.sh
```
* Exit PlaFRIM
```
exit
```

## Copy result files from PLaFRIM to local machine
```
eval $(ssh-agent)
ssh-add
rsync -avz --exclude='logs' plafrim:/beegfs/lfontain/unsupervised-pcn/outputs/ ~/.git/unsupervised-pcn/plafrim-outputs/

```
The first two lines are used to only have to enter the passphrase once to copy files from PLaFRIM, using a SSH-Agent. The last line synchronizes the outputs directory on PLaFRIM except the logs folder with the outputs directory of the local machine. In order to use rsync, install it using the installation script provided [here](https://scicomp.aalto.fi/scicomp/rsynconwindows/) (if you are on Windows). 

## Important check for the bPC files 
In all these files it is assumed that you have already connected your wandb account via the terminal in the distant node of Plafri. If you have not done so yet just add: wandb.login(YOUR_API_KEY)

## Useful tips for PLaFRIM
* Check available modules
```
module avail
```
* Check CUDA version
```
nvidia-smi
```
We use A100 nodes in PLaFRIM which have CUDA 12.3. NVIDIA drivers are backward compatible with older CUDA versions, so we can install Pytorch built for CUDA 11.8 (latest Pytorch version is not built for 12.3). 
* Load recurrent modules automatically after login, by adding in $HOME/.bashrc
```
module load modulename
```
where modulename is tools/git/2.36.0 for example. Apply changes with  
```
source ~/.bashrc
```
* Create python project in /beegfs/username/ in plafrim where there is more storage. 
* List all files including those starting with .
```
ls -a
```

## References
* Formation SSH: https://www.math.u-bordeaux.fr/imb/cellule/Formations-SSH
* Formation Moyens de calcul: https://www.math.u-bordeaux.fr/imb/cellule/Formations-Moyens-de-calcul