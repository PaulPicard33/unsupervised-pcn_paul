import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np
from sklearn.manifold import TSNE

from model import VGG5_bPC_Paper
from dataset import get_fmnist_dataloaders

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def evaluate_discrimination(model, dataloader, device, cf):
    model.eval()
    correct = 0
    total = 0
    
    print("--- Évaluation de la capacité de discrimination ---")
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        batch_size = images.size(0)
        
        # Initialisation par balayage ascendant
        x_init = model.bottom_up_sweep(images)
        x_init[0] = images
        
        # Inférence avec l'image d'entrée figée (index 0)
        x_inferred = model.infer(
            x_init,
            clamped_indices=[0],
            steps=cf.infer_steps_eval,
            lr_x=cf.lr_x_eval
        )
        
        # Extraction de la prédiction depuis les 10 premiers neurones de la couche latente
        preds = x_inferred[-1][:, :cf.num_labels].argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += batch_size
        
    accuracy = 100 * correct / total
    print(f"Précision (Accuracy) sur le set de validation : {accuracy:.2f}%
")
    return accuracy

def evaluate_generation(model, device, cf):
    model.eval()
    print("--- Évaluation de la capacité de génération ---")
    
    # On veut générer une image pour chaque classe (0 à 9)
    labels = torch.arange(10).to(device)
    batch_size = 10
    latent_dim = cf.num_labels + cf.rep_neurons
    
    # Création d'une entrée aléatoire pour déduire les dimensions via un balayage
    dummy_input = torch.randn(batch_size, 1, 32, 32, device=device)
    x_init = model.bottom_up_sweep(dummy_input)
    
    # Remplacement des activations par du bruit pur (initialisation de la génération)
    x_init = [torch.randn_like(tensor) * 0.1 for tensor in x_init]
    
    # Préparation du masque de clampage partiel (on ne fige QUE le label)
    latent_mask = torch.zeros(latent_dim, device=device)
    latent_mask[:cf.num_labels] = 1.0
    
    # Injection du label one-hot dans la couche latente
    x_label = torch.zeros((batch_size, latent_dim), device=device)
    x_label[:, :cf.num_labels] = F.one_hot(labels, num_classes=cf.num_labels).float()
    x_init[-1][:, :cf.num_labels] = x_label[:, :cf.num_labels]
    
    # Inférence : AUCUNE couche entièrement figée. Seul le label est maintenu constant.
    x_inferred = model.infer(
        x_init,
        clamped_indices=[], 
        steps=cf.infer_steps_gen,
        lr_x=cf.lr_x_gen,
        partial_clamp=(model.L - 1, latent_mask.unsqueeze(0))
    )
    
    generated_images = x_inferred[0].detach().cpu()
    
    # Sauvegarde des images
    os.makedirs("results", exist_ok=True)
    fig, axes = plt.subplots(1, 10, figsize=(15, 2))
    for i in range(10):
        img = generated_images[i].squeeze().numpy()
        # Dénormalisation rudimentaire pour l'affichage [-1, 1] -> [0, 1]
        img = (img + 1.0) / 2.0
        img = np.clip(img, 0, 1)
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"Label {i}")
        axes[i].axis('off')
    plt.tight_layout()
    plt.savefig("results/generated_images.png")
    print("Images générées avec succès et sauvegardées dans 'results/generated_images.png'
")

def plot_tsne_layers(model, dataloader, device):
    model.eval()
    print("--- Calcul des projections t-SNE pour l'analyse des représentations ---")
    
    # On récupère un seul batch (ex: 500 images) pour que le calcul t-SNE reste rapide
    images, labels = next(iter(dataloader))
    images = images.to(device)
    
    # On extrait les activations du modèle via le balayage ascendant
    with torch.no_grad():
        x_layers = model.bottom_up_sweep(images)
        
    labels_np = labels.numpy()
    num_layers = len(x_layers)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i in range(num_layers):
        print(f"Projection t-SNE de la couche x_{i} en cours...")
        # Aplatir les activations : (Batch, Channels, H, W) -> (Batch, Features)
        layer_data = x_layers[i].view(images.size(0), -1).cpu().numpy()
        
        # Calcul de la t-SNE
        tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
        tsne_results = tsne.fit_transform(layer_data)
        
        # Tracé
        scatter = axes[i].scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels_np, cmap='tab10', s=15, alpha=0.8)
        axes[i].set_title(f"Espace Latent - Couche x_{i}")
        axes[i].axis('off')
        
    # Légende globale
    handles, _ = scatter.legend_elements(prop="colors")
    fig.legend(handles, [str(i) for i in range(10)], loc="upper right", title="Classes (0-9)", fontsize=12)
    
    os.makedirs("results", exist_ok=True)
    plt.tight_layout()
    plt.savefig("results/tsne_layers.png")
    print("Tracés t-SNE sauvegardés dans 'results/tsne_layers.png'
")

def main(cf):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Évaluation sur : {device}
")

    # Chargement des données (on utilise un subset_size pour la t-SNE si besoin)
    datasets = get_fmnist_dataloaders(batch_size=cf.batch_size, subset_size=cf.subset_size)
    val_loader = datasets["val"]
    
    # Initialisation du modèle
    bpc_model = VGG5_bPC_Paper(
        num_labels=cf.num_labels, 
        rep_neurons=cf.rep_neurons, 
        alpha_gen=cf.alpha_gen, 
        alpha_disc=cf.alpha_disc
    ).to(device)
    
    # Chargement des poids si fournis
    if cf.model_path and os.path.exists(cf.model_path):
        bpc_model.load_state_dict(torch.load(cf.model_path, map_location=device))
        print(f"Poids du modèle chargés depuis {cf.model_path}
")
    else:
        print("ATTENTION: Aucun chemin de modèle fourni ou fichier introuvable. Évaluation avec des poids aléatoires !
")

    # 1. Évaluation de la Discrimination
    evaluate_discrimination(bpc_model, val_loader, device, cf)
    
    # 2. Évaluation de la Génération
    evaluate_generation(bpc_model, device, cf)
    
    # 3. Tracé des t-SNE
    # On crée un petit DataLoader spécifique pour le t-SNE pour ne pas saturer la RAM (ex: 1000 échantillons)
    tsne_dataset = get_fmnist_dataloaders(batch_size=1000, subset_size=1000)
    plot_tsne_layers(bpc_model, tsne_dataset["val"], device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script d'évaluation bPC (Discrimination, Génération, t-SNE)")
    
    parser.add_argument("--model_path", type=str, default="", help="Chemin vers le fichier .pt du modèle entraîné")
    parser.add_argument("--batch_size", type=int, default=512, help="Taille de batch pour l'évaluation")
    parser.add_argument("--subset_size", type=int, default=None, help="Taille du sous-ensemble pour test rapide")
    parser.add_argument("--rep_neurons", type=int, default=256, help="Nombre de neurones de représentation")
    
    args = parser.parse_args()

    cf = AttrDict()
    cf.model_path = args.model_path
    cf.batch_size = args.batch_size
    cf.subset_size = args.subset_size
    
    # Paramètres d'architecture
    cf.num_labels = 10
    cf.rep_neurons = args.rep_neurons
    cf.alpha_gen = 1e-4
    cf.alpha_disc = 1.0
    
    # Paramètres d'inférence spécifiques à l'évaluation
    cf.infer_steps_eval = 20  # Inférence courte pour la discrimination
    cf.lr_x_eval = 0.01
    
    cf.infer_steps_gen = 100  # Inférence plus longue pour la génération pure
    cf.lr_x_gen = 0.05

    main(cf)
