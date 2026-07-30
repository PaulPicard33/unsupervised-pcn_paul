import torch
import torch.optim as optim
import wandb
import os
import sys 
import argparse
from models import VGG5_bPC_Paper
from datasets import get_fmnist_dataloaders,get_CIFAR10_dataloaders
import torch.nn.functional as F


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def main(cf):
    # --- NOMMAGE DU RUN WANDB ---
    model_name = f"{cf.dataset}"
    if cf.subset_size is not None:
        model_name += f"-subset_size={cf.subset_size}"
    model_name += f"-latent={cf.rep_neurons}-lr={cf.lr}-steps={cf.infer_steps}-epochs={cf.n_epochs}"

    # --- INITIALISATION WANDB ---
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    wandb.login()
    wandb.init(project="mon-projet-pcn", config=cf, name=model_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Exécution sur : {device}")

    # --- CHARGEMENT DES DONNÉES ---
    # Le dataset est fixé à fMNIST pour l'instant via dataset.py, mais on le rend paramétrable dans la config
    if cf.dataset == "fmnist":
        datasets = get_fmnist_dataloaders(batch_size=cf.batch_size, subset_size=cf.subset_size)
    elif cf.dataset == "CIFAR10":
        datasets = get_CIFAR10_dataloaders(batch_size=cf.batch_size, subset_size=cf.subset_size)
    
    # --- INITIALISATION DU MODÈLE ---
    bpc_model = VGG5_bPC_Paper(
        num_labels=cf.num_labels, 
        rep_neurons=cf.rep_neurons, 
        alpha_gen=cf.alpha_gen, 
        alpha_disc=cf.alpha_disc,
        cifar=True if cf.dataset == "CIFAR10" else False
    ).to(device)
   

    # Taux d'apprentissage distincts (lr_theta_gen et lr_theta_disc)[cite: 2]
    optimizer_theta= optim.Adam(bpc_model.parameters(), lr=cf.lr,weight_decay=cf.weight_decay)
    # --- 2. SCHEDULER COSINE ANNEALING[cite: 2] ---
    import math
    from torch.optim.lr_scheduler import LambdaLR

    # 1. Définir la fonction mathématique de la courbe de la Table 23
    def custom_lr_multiplier(epoch, total_epochs):
        warmup_epochs = max(1, int(0.1 * total_epochs)) # 10% des epochs
        
        if epoch < warmup_epochs:
            # Phase 1 : Augmentation de 1.0 à 1.1
            return 1.0 + 0.1 * (epoch / warmup_epochs)
        else:
            # Phase 2 : Chute cosinusoïdale de 1.1 à 0.1
            progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
            return 0.1 + 0.5 * (1.1 - 0.1) * (1.0 + math.cos(math.pi * progress))

    # 2. Créer l'unique scheduler en passant cette fonction via un lambda
    scheduler_theta = LambdaLR(
        optimizer_theta, 
        lr_lambda=lambda e: custom_lr_multiplier(e, cf.n_epochs)
    )
    # Masque pour le clampage partiel (fige les 'num_labels' premières composantes)
    latent_dim = cf.num_labels + cf.rep_neurons
    latent_mask = torch.zeros(latent_dim, device=device)
    latent_mask[:cf.num_labels] = 1.0 

    print("Début de l'entraînement bPC...")

    # --- BOUCLE D'ENTRAÎNEMENT ---
    for epoch in range(cf.n_epochs):
        bpc_model.train()
        total_energy = 0.0
        
        for batch_idx, (images, labels) in enumerate(datasets["train"]):
            x1_input = images.to(device)
            current_batch_size = x1_input.size(0)
            
            # Tenseur latent complet : one-hot pour les labels, 0 pour les neurones libres
            xL_label = torch.zeros((current_batch_size, latent_dim), device=device)
            xL_label[:, :cf.num_labels] = F.one_hot(labels, num_classes=cf.num_labels).float()
            
            # 1. Initialisation par balayage ascendant
            x_init = bpc_model.bottom_up_sweep(x1_input)
            x_init[0] = x1_input
            x_init[-1][:, :cf.num_labels] = xL_label[:, :cf.num_labels]
            
            # 2. Inférence itérative
            x_inferred = bpc_model.infer_error_optim(
                x_init, 
                clamped_indices=[0], 
                steps=cf.infer_steps, 
                lr_x=cf.e_lr, 
                lr_x_free=cf.e_lr,
                partial_clamp=(5, latent_mask.unsqueeze(0)),
                activity_decay=cf.activity_decay
            )
            
            optimizer_theta.zero_grad()
            
            loss = bpc_model.compute_energy(x_inferred)
            loss.backward()
            
            optimizer_theta.step()
            total_energy += loss.item()
            
            # Log par batch (optionnel, peut être lourd)
            if batch_idx % cf.log_freq == 0:
                 wandb.log({"batch_loss": loss.item()})

        # --- LOGGING WANDB FIN D'EPOCH ---
        avg_energy = total_energy / len(datasets["train"])
        # Mise à jour des taux d'apprentissage à chaque époque[cite: 2]
        scheduler_theta.step()
        wandb.log({
            "epoch": epoch,
            "train_energy_avg": avg_energy,
            "lr": optimizer_theta.param_groups[0]["lr"],
        })
        
        print(f"Epoch [{epoch+1}/{cf.n_epochs}] - Énergie moyenne: {avg_energy:.4f}")

    wandb.finish()

    # --- SAUVEGARDE DU MODÈLE ---
    if not os.path.exists("models"):
        os.makedirs("models")
    torch.save(bpc_model.state_dict(), f"models/bpc-{model_name}.pt")
    print(f"Modèle sauvegardé sous models/bpc-{model_name}.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script d'entraînement du modèle bPC")
    
    # --- ARGUMENTS LIGNE DE COMMANDE ---
    parser.add_argument("--dataset", choices=['fmnist','CIFAR10'], default='CIFAR10', help="Nom du dataset")
    parser.add_argument("--subset_size", type=int, default=None, help="Taille du sous-ensemble (pour tests locaux)")
    parser.add_argument("--n_epochs", type=int, default=25, help="Nombre d'époques")
    parser.add_argument("--batch_size", type=int, default=1024, help="Taille des batchs")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate (AdamW)")
    parser.add_argument("--rep_neurons", type=int, default=256, help="Nombre de neurones de représentation (libres)")
    parser.add_argument("--infer_steps", type=int, default=1, help="Nombre de pas d'inférence (T)")
    parser.add_argument("--e_lr", type=float, default=0.001, help="Learning rate de l'inférence (SGD sur x)")
    args = parser.parse_args()

    # --- DICTIONNAIRE DE CONFIGURATION ---
    cf = AttrDict()

    # Paramètres généraux
    cf.dataset = args.dataset
    cf.subset_size = args.subset_size
    cf.n_epochs = args.n_epochs
    cf.batch_size = args.batch_size
    cf.log_freq = 10 # Log la loss tous les 10 batchs
    
    # Paramètres d'optimisation (theta)
    cf.lr = args.lr
    cf.weight_decay = 1e-4
    cf.warmup_epochs = round(0.1*cf.n_epochs)
    
    # Paramètres du Modèle bPC
    cf.num_labels = 10
    cf.rep_neurons = args.rep_neurons
    cf.alpha_gen = 1e-5
    cf.alpha_disc = 1.0
    
    # Paramètres d'inférence (x)
    cf.infer_steps = args.infer_steps
    cf.e_lr = args.e_lr
    cf.activity_decay = 1e-3

    main(cf)
