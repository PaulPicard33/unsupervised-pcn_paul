import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np
import wandb
from sklearn.manifold import TSNE

# Import de ta nouvelle architecture et des dataloaders
from train_claude import bPC_VGG, AttrDict
from pcn.datasets import get_CIFAR10_dataloaders, get_fmnist_dataloaders

def evaluate_discrimination(model, dataloader, device, cf):
    """
    Évalue la classification en mode Inférence.
    L'image est figée, le Vode du label est libéré et optimisé pour minimiser l'énergie.
    """
    correct = 0
    total = 0
    
    print("\n--- Évaluation de la capacité de discrimination (Inférence T=100) ---")
    
    # Même en évaluation, on ne met PAS torch.no_grad() globalement
    # car l'inférence PC a besoin des gradients pour optimiser les états (x)
    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        batch_size = images.size(0)
        
        # 1. Préparation des états
        x_label_dummy = torch.zeros((batch_size, cf.num_labels), device=device)
        
        # 2. Configuration stricte des "Frozen" pour la classification
        model.vodes[-1].frozen = True  # L'image est une observation fixe
        model.vodes[0].frozen = False  # Le label est inconnu, on le laisse libre !
        
        # 3. Initialisation Bottom-Up
        model.vodes[-1].h = images
        model.init_ff(x_label_dummy, images, is_up=True)
        
        # Astuce : On donne au label un bon point de départ en utilisant la prédiction feedforward
        with torch.no_grad():
            model.compute_energy(x_label_dummy, images, weighted=True)
            model.vodes[0].h = model.vodes[0].u.clone().detach().requires_grad_(True)
            
        # 4. Inférence (Descente d'énergie sur les états)
        model.infer(
            x_label=model.vodes[0].h, 
            y_image=images,
            T=cf.infer_steps_eval,
            lr_h=cf.lr_x_eval,
            lr_h_latent=cf.lr_x_latent,
            alpha_up=cf.alpha_disc,
            alpha_down=cf.alpha_gen
        )
        
        # 5. Lecture de la prédiction finale
        preds = model.vodes[0].h.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += batch_size
        
        # Verrouiller à nouveau le label par sécurité
        model.vodes[0].frozen = True
        
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}/{len(dataloader)} - Accuracy partielle : {100 * correct / total:.2f}%")
            
    accuracy = 100 * correct / total
    print(f"\n=> Précision (Accuracy) finale sur le set de validation : {accuracy:.2f}%")
    wandb.log({"eval/accuracy": accuracy})

    return accuracy

def plot_tsne_layers(model, dataloader, device):
    """
    Extrait les représentations de chaque Vode et projette l'espace latent en 2D via t-SNE.
    """
    print("\n--- Calcul des projections t-SNE pour l'analyse des Vodes ---")
    
    # On récupère un gros batch (ex: 1000 images)
    images, labels = next(iter(dataloader))
    images = images.to(device)
    labels_np = labels.numpy()
    
    # Passe ascendante pour remplir les Vodes
    x_label_dummy = torch.zeros((images.size(0), model.output_size), device=device)
    model.vodes[-1].h = images
    
    with torch.no_grad():
        model.init_ff(x_label_dummy, images, is_up=True)
        
        # Dictionnaire des couches à visualiser
        layers_data = {
            'Latent 256 (Libre)': model.latent_vode.h.view(images.size(0), -1).cpu().numpy(),
            'Vode 1 (Flatten 512)': model.vodes[1].h.view(images.size(0), -1).cpu().numpy(),
            'Vode 2 (Post Conv4)': model.vodes[2].h.view(images.size(0), -1).cpu().numpy(),
            'Vode 3 (Post Conv3)': model.vodes[3].h.view(images.size(0), -1).cpu().numpy(),
            'Vode 4 (Post Conv2)': model.vodes[4].h.view(images.size(0), -1).cpu().numpy(),
            'Vode 5 (Post Conv1)': model.vodes[5].h.view(images.size(0), -1).cpu().numpy(),
        }

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, (name, data) in enumerate(layers_data.items()):
        print(f"Génération t-SNE pour {name}...")
        tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
        tsne_results = tsne.fit_transform(data)
        
        scatter = axes[i].scatter(
            tsne_results[:, 0], tsne_results[:, 1], 
            c=labels_np, cmap='tab10', s=15, alpha=0.8
        )
        axes[i].set_title(f"Espace : {name}")
        axes[i].axis('off')
        
    handles, _ = scatter.legend_elements(prop="colors")
    fig.legend(handles, [str(i) for i in range(10)], loc="upper right", title="Classes (0-9)", fontsize=12)
    
    os.makedirs("results", exist_ok=True)
    plt.tight_layout()
    plt.savefig("results/tsne_vodes.png", dpi=150)
    print("Tracés t-SNE sauvegardés avec succès dans 'results/tsne_vodes.png' !")
    wandb.log({"eval/tsne_layers": wandb.Image(fig, caption="Espace Latent (t-SNE) par couche")}) #[cite: 6]
    plt.close(fig)

def main(cf):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initialisation de l'évaluation sur : {device}")
    # --- INITIALISATION WANDB ---
    os.environ["WANDB__SERVICE_WAIT"] = "300" #[cite: 6]
    wandb.login() #[cite: 6]
    run_name = "eval-" + os.path.basename(cf.model_path) if cf.model_path else "eval-random-weights" #[cite: 6]
    wandb.init(project="mon-projet-pcn", config=cf, name=run_name, job_type="evaluation") #[cite: 6]
    # Chargement des données
    if cf.dataset == "fmnist":
        datasets = get_fmnist_dataloaders(batch_size=cf.batch_size, subset_size=cf.subset_size)
        input_channels, input_size = 1, (28, 28)
    elif cf.dataset == "CIFAR10":
        datasets = get_CIFAR10_dataloaders(batch_size=cf.batch_size, subset_size=cf.subset_size)
        input_channels, input_size = 3, (32, 32)

    val_loader = datasets["val"]

    # Instanciation du modèle avec les mêmes paramètres que l'entraînement
    bpc_model = bPC_VGG(
        input_channels=input_channels,
        input_size=input_size,
        output_size=cf.num_labels,
        latent_dim=cf.latent_dim,
        latent_var=cf.alpha_disc / cf.alpha_gen, # 10^7
        device=str(device),
    ).to(device)
    
    # Chargement des poids
    if cf.model_path and os.path.exists(cf.model_path):
        bpc_model.load_state_dict(torch.load(cf.model_path, map_location=device))
        print(f"Poids chargés avec succès depuis : {cf.model_path}")
    else:
        print("ATTENTION: Aucun chemin valide fourni, évaluation sur poids aléatoires !")

    # 1. Évaluation de l'accuracy
    evaluate_discrimination(bpc_model, val_loader, device, cf)
    
    # 2. t-SNE
    # On crée un petit dataloader spécifique pour le t-SNE (1000 images d'un coup)
    if cf.dataset == "fmnist":
        tsne_dataset = get_fmnist_dataloaders(batch_size=1000, subset_size=1000)
    else:
        tsne_dataset = get_CIFAR10_dataloaders(batch_size=1000, subset_size=1000)
        
    plot_tsne_layers(bpc_model, tsne_dataset["val"], device)
    wandb.finish()  # Clôture de la session WandB

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Évaluation bPC (Inférence Classification & t-SNE)")
    parser.add_argument("--model_path", type=str, required=True, help="Chemin vers le fichier .pt du modèle")
    parser.add_argument("--dataset", choices=["fmnist", "CIFAR10"], default="CIFAR10")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--subset_size", type=int, default=None)
    args = parser.parse_args()

    cf = AttrDict()
    cf.model_path   = args.model_path
    cf.dataset      = args.dataset
    cf.batch_size   = args.batch_size
    cf.subset_size  = args.subset_size
    cf.num_labels   = 10
    cf.latent_dim   = 256

    # Hyperparamètres d'inférence pour l'évaluation (cf yaml: T_eval=100)
    cf.infer_steps_eval = 100
    cf.lr_x_eval        = 0.00192827
    cf.lr_x_latent      = 0.00316244
    
    # Scalings
    cf.alpha_gen  = 1e-7
    cf.alpha_disc = 1.0

    main(cf)