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
    model_name += f"-latent={cf.rep_neurons}-lr_p={cf.lr_p}-steps={cf.infer_steps}-epochs={cf.n_epochs}"

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
   

    # 1. Isoler UNIQUEMENT les couches linéaires des 256 neurones libres
    params_gen = list(bpc_model.V_linear_free.parameters()) # Ajoute la couche W correspondante si elle est séparée
    
    # 2. Tout le reste du réseau (Convolutions V et W, et logits)
    params_disc = list(bpc_model.V_convs.parameters()) + \
                  list(bpc_model.W_convs.parameters()) + \
                  list(bpc_model.V_linear_labels.parameters()) + \
                  list(bpc_model.W_linear.parameters()) # Couche W des labels

    # 3. Application stricte des hyperparamètres du sweep
    optimizer_gen = optim.Adam(params_gen, lr=0.001555, weight_decay=0.000349)
    optimizer_disc = optim.Adam(params_disc, lr=0.000141, weight_decay=0.000349)
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
    """ scheduler_theta = LambdaLR(
        optimizer_theta, 
        lr_lambda=lambda e: custom_lr_multiplier(e, cf.n_epochs)
    ) """
    scheduler_gen= torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_gen, T_max=cf.n_epochs)
    scheduler_disc = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_disc, T_max=cf.n_epochs)
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
            x_inferred = bpc_model.infer(
                x_init, 
                clamped_indices=[0,bpc_model.L-1], 
                steps=cf.infer_steps, 
                lr_x=cf.lr_x,
                lr_x_free=cf.lr_x_free,
            )
            
            optimizer_gen.zero_grad()
            optimizer_disc.zero_grad()
            
            energy_gen,energy_disc = bpc_model.compute_raw_energies(x_inferred)
            loss=energy_disc+energy_gen
            (loss/current_batch_size).backward()
            
            optimizer_gen.step()
            optimizer_disc.step()
            total_energy += loss.item()
            
            # Log par batch (optionnel, peut être lourd)
            if batch_idx % cf.log_freq == 0:
                 wandb.log({"batch_loss": loss.item()})

        # --- LOGGING WANDB FIN D'EPOCH ---
        avg_energy = total_energy / len(datasets["train"])
        # Mise à jour des taux d'apprentissage à chaque époque[cite: 2]
        scheduler_gen.step()
        scheduler_disc.step()
        wandb.log({
            "epoch": epoch,
            "train_energy_avg": avg_energy,
            "lr_gen": optimizer_gen.param_groups[0]["lr"],
            "lr_disc":optimizer_disc.param_groups[0]["lr"]
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
    parser.add_argument("--batch_size", type=int, default=256, help="Taille des batchs")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate (AdamW)")
    parser.add_argument("--rep_neurons", type=int, default=256, help="Nombre de neurones de représentation (libres)")
    parser.add_argument("--infer_steps", type=int, default=32, help="Nombre de pas d'inférence (T)")
    parser.add_argument("--e_lr", type=float, default=0.001, help="Learning rate de l'inférence (SGD sur x)")
    parser.add_argument("--activity_decay", type=float, default=1e-3, help="Décroissance de l'activité")
    parser.add_argument("--lr_x_free", type=float, default=0.1, help="Learning rate pour les neurones libres (x)")
    parser.add_argument("--lr_x", type=float, default=0.01, help="Learning rate pour les neurones liés (x)")
    args = parser.parse_args()

    # --- DICTIONNAIRE DE CONFIGURATION (Valeurs exactes de Bogacz) ---
    cf = AttrDict()

    cf.dataset = args.dataset
    cf.subset_size = args.subset_size
    cf.n_epochs = 50 # Le fichier indique 50 époques
    cf.batch_size = 256 # Batch size réduit pour plus de stochasticité
    cf.log_freq = 10 
    
    # Paramètres d'optimisation des POIDS (Dual Optimizer)
    cf.lr_p = 0.0001415926 # Learning rate voie discriminative
    cf.lr_p_latent = 0.0015553778 # Learning rate voie générative/latente (10x plus grand !)[cite: 5]
    cf.weight_decay = 0.0003497999 # Weight decay très précis[cite: 5]
    
    # Paramètres du Modèle bPC
    cf.num_labels = 10 #[cite: 5]
    cf.rep_neurons = 256 # Les 256 neurones libres sont de retour[cite: 5]
    
    # LE SECRET EST ICI : L'énergie générative est infime !
    cf.alpha_gen = 0.0000001 # 1e-7[cite: 5]
    cf.alpha_disc = 1.0 #[cite: 5]
    
    # Paramètres d'inférence des ÉTATS (x)
    cf.infer_steps = 32 # T = 32 itérations pour l'entraînement[cite: 5]
    cf.lr_x = 0.00192827 # Vitesse de relaxation discriminative[cite: 5]
    cf.lr_x_free = 0.00316244 # Vitesse de relaxation des 256 neurones[cite: 5]
    cf.activity_decay = 0.0

    main(cf)
