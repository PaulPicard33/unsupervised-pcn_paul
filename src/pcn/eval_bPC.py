import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np
import wandb
from sklearn.manifold import TSNE

from models import VGG5_bPC_Paper
from datasets import get_fmnist_dataloaders, get_CIFAR10_dataloaders

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
        
        x_init = model.bottom_up_sweep(images)
        x_init[0] = images
        
        x_inferred = model.infer(
            x_init,
            clamped_indices=[0],
            steps=cf.infer_steps_eval,
            lr_x=cf.lr_x_eval
        )
        
        preds = x_inferred[-1][:, :cf.num_labels].argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += batch_size
        
    accuracy = 100 * correct / total
    print(f"Précision (Accuracy) sur le set de validation : {accuracy:.2f}%")
    
    # Envoi de la métrique sur WandB[cite: 6]
    wandb.log({"eval/accuracy": accuracy}) #[cite: 6]
    
    return accuracy

# À ajouter dans eval.py
def external_top_down_sweep(model, xL):
    x = [None] * model.L
    x[-1] = xL.clone()
    with torch.no_grad():
        for i in range(model.L - 2, -1, -1):
            x[i] = model._forward_W(x[i+1], i)
    return x

# Remplacement dans la fonction evaluate_generation
def evaluate_generation(model, device, cf):
    model.eval()
    print("--- Évaluation de la capacité de génération ---")
    
    labels = torch.arange(10).to(device)
    batch_size = 10
    latent_dim = cf.num_labels + cf.rep_neurons
    
    # 1. Préparation de la cible latente complète
    x_label = torch.zeros((batch_size, latent_dim), device=device)
    x_label[:, :cf.num_labels] = F.one_hot(labels, num_classes=cf.num_labels).float()
    
    # 2. LA CORRECTION : Initialisation Top-Down (On dessine à partir du label)
    x_init = external_top_down_sweep(model, x_label)
    
    # 3. LA CORRECTION : Couper l'énergie discriminative
    original_alpha_disc = model.alpha_disc
    model.alpha_disc = 0.0
    
    # Inférence : On fige toute la couche latente (indice model.L - 1)
    x_inferred = model.infer(
        x_init,
        clamped_indices=[model.L - 1], 
        steps=cf.infer_steps_gen,
        lr_x=cf.lr_x_gen
    )
    
    # Restauration de l'alpha
    model.alpha_disc = original_alpha_disc
    
    generated_images = x_inferred[0].detach().cpu()
    
    # ... (le reste de ton code d'affichage avec matplotlib reste identique)
    os.makedirs("results", exist_ok=True)
    fig, axes = plt.subplots(1, 10, figsize=(15, 2))
    for i in range(10):
        img = generated_images[i].squeeze().numpy()
        img = (img + 1.0) / 2.0
        img = np.clip(img, 0, 1)
        axes[i].imshow(np.transpose(img,(1,2,0)), cmap='gray')
        axes[i].set_title(f"Label {i}")
        axes[i].axis('off')
    plt.tight_layout()
    
    # Sauvegarde locale ET envoi sur WandB[cite: 6]
    plt.savefig("results/generated_images.png") #[cite: 6]
    wandb.log({"eval/generated_images": wandb.Image(fig, caption="Images générées (Classes 0-9)")}) #[cite: 6]
    print("Images générées avec succès et envoyées sur W&B !") #[cite: 6]
    plt.close(fig) #[cite: 6]
def evaluate_reconstruction(model, dataloader, device, cf):
    model.eval()
    print("\n--- Évaluation de la Reconstruction (256 neurones libres + Label) ---")
    
    images, labels = next(iter(dataloader))
    images, labels = images[:10].to(device), labels[:10].to(device)
    
    # --- PHASE 1 : ENCODAGE ---
    x_init = model.bottom_up_sweep(images)
    x_init[0] = images
    
    latent_mask = torch.zeros(cf.num_labels + cf.rep_neurons, device=device)
    latent_mask[:cf.num_labels] = 1.0 # On fige le label
    
    xL_label = torch.zeros((10, cf.num_labels + cf.rep_neurons), device=device)
    xL_label[:, :cf.num_labels] = F.one_hot(labels, num_classes=cf.num_labels).float()
    x_init[-1][:, :cf.num_labels] = xL_label[:, :cf.num_labels]
    
    # On laisse le réseau inférer les 256 neurones libres
    x_encoded = model.infer(
        x_init,
        clamped_indices=[0], 
        steps=cf.infer_steps_eval,
        lr_x=cf.lr_x_eval,
        partial_clamp=(model.L - 1, latent_mask.unsqueeze(0))
    )
    
    # --- PHASE 2 : DÉCODAGE (Génération) ---
    latent_state = x_encoded[-1].detach()
    latent_state[:, :cf.num_labels] = xL_label[:, :cf.num_labels] # Force le label parfait
    
    x_init_dec = external_top_down_sweep(model, latent_state)
    
    # On coupe l'énergie discriminative pour laisser le générateur s'exprimer
    original_alpha_disc = model.alpha_disc
    model.alpha_disc = 0.0
    
    x_decoded = model.infer(
        x_init_dec,
        clamped_indices=[model.L - 1], 
        steps=cf.infer_steps_gen,
        lr_x=cf.lr_x_gen
    )
    
    model.alpha_disc = original_alpha_disc # Restauration
    
    # --- AFFICHAGE ---
    reconstructed_images = x_decoded[0].detach().cpu()
    original_images = images.cpu()
    
    fig, axes = plt.subplots(2, 10, figsize=(15, 4))
    fig.suptitle("Reconstruction bPC (Haut: Original | Bas: Reconstruit)", fontsize=14)
    
    for i in range(10):
        # Original
        img_orig = np.clip((original_images[i].numpy() + 1.0) / 2.0, 0, 1)
        axes[0, i].imshow(np.transpose(img_orig, (1, 2, 0)))
        axes[0, i].axis('off')
        # Reconstruit
        img_recon = np.clip((reconstructed_images[i].numpy() + 1.0) / 2.0, 0, 1)
        axes[1, i].imshow(np.transpose(img_recon, (1, 2, 0)))
        axes[1, i].axis('off')
        
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/reconstruction.png")
    wandb.log({"eval/reconstruction": wandb.Image(fig, caption="Original vs Reconstruction")})
    plt.close(fig)
    print("Reconstructions générées avec succès et envoyées sur W&B !")
def evaluate_inpainting(model, dataloader, device, cf, missing_ratio=0.5):
    model.eval()
    print(f"\n--- Évaluation de l'Inpainting ({missing_ratio*100}% de pixels manquants) ---")
    
    images, labels = next(iter(dataloader))
    images, labels = images[:10].to(device), labels[:10].to(device)
    batch_size = images.size(0)
    
    # Création du masque binaire sur tous les canaux[cite: 1]
    mask_2d = (torch.rand(batch_size, 1, 32, 32, device=device) > missing_ratio).float()
    mask = mask_2d.expand(-1, 3, -1, -1)
    
    # Les pixels manquants sont initialisés à zéro[cite: 1]
    masked_images = images * mask 
    
    x_init = model.bottom_up_sweep(masked_images)
    x_init[0] = masked_images.clone()
    
    # Utilisation du partial_clamp sur la couche 0 :
    # Les gradients des pixels connus (mask=1) sont annulés (figés).
    # Les gradients des pixels manquants (mask=0) sont actifs (le modèle les infère).
    x_inferred = model.infer(
        x_init,
        clamped_indices=[], # x_0 n'est PAS dans les indices totalement figés
        steps=cf.infer_steps_gen * 2, # On donne plus de temps pour l'inpainting
        lr_x=cf.lr_x_gen,
        partial_clamp=(0, mask) 
    )
    
    inpainted_images = x_inferred[0].detach().cpu()
    preds = x_inferred[-1][:, :cf.num_labels].argmax(dim=1)
    
    # --- AFFICHAGE ---
    fig, axes = plt.subplots(3, 10, figsize=(15, 6))
    fig.suptitle(f"Inpainting bPC - {missing_ratio*100}% occulté (Haut: Original | Milieu: Masqué | Bas: Inferred)", fontsize=14)
    
    for i in range(10):
        # Original
        img_orig = np.clip((images[i].cpu().numpy() + 1.0) / 2.0, 0, 1)
        axes[0, i].imshow(np.transpose(img_orig, (1, 2, 0)))
        axes[0, i].set_title(f"Vrai: {labels[i].item()}")
        axes[0, i].axis('off')
        
        # Masqué
        img_masked = np.clip((masked_images[i].cpu().numpy() + 1.0) / 2.0, 0, 1)
        axes[1, i].imshow(np.transpose(img_masked, (1, 2, 0)))
        axes[1, i].axis('off')
        
        # Reconstruit
        img_inp = np.clip((inpainted_images[i].numpy() + 1.0) / 2.0, 0, 1)
        axes[2, i].imshow(np.transpose(img_inp, (1, 2, 0)))
        axes[2, i].set_title(f"Prédit: {preds[i].item()}")
        axes[2, i].axis('off')
        
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig(f"results/inpainting_{missing_ratio}.png")
    wandb.log({f"eval/inpainting_{missing_ratio}": wandb.Image(fig, caption="Inpainting bPC")})
    plt.close(fig)
    print("Inpainting généré avec succès et envoyé sur W&B !")

def plot_tsne_layers(model, dataloader, device):
    model.eval()
    print("--- Calcul des projections t-SNE pour l'analyse des représentations ---")
    
    images, labels = next(iter(dataloader))
    images = images.to(device)
    
    with torch.no_grad():
        x_layers = model.bottom_up_sweep(images)
        
    labels_np = labels.numpy()
    num_layers = len(x_layers)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i in range(num_layers):
        print(f"Projection t-SNE de la couche x_{i} en cours...")
        layer_data = x_layers[i].view(images.size(0), -1).cpu().numpy()
        
        tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
        tsne_results = tsne.fit_transform(layer_data)
        
        scatter = axes[i].scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels_np, cmap='tab10', s=15, alpha=0.8)
        axes[i].set_title(f"Espace Latent - Couche x_{i}")
        axes[i].axis('off')
        
    handles, _ = scatter.legend_elements(prop="colors")
    fig.legend(handles, [str(i) for i in range(10)], loc="upper right", title="Classes (0-9)", fontsize=12)
    
    os.makedirs("results", exist_ok=True)
    plt.tight_layout()
    
    # Sauvegarde locale ET envoi sur WandB[cite: 6]
    plt.savefig("results/tsne_layers.png") #[cite: 6]
    wandb.log({"eval/tsne_layers": wandb.Image(fig, caption="Espace Latent (t-SNE) par couche")}) #[cite: 6]
    print("Tracés t-SNE envoyés sur W&B !") #[cite: 6]
    plt.close(fig) #[cite: 6]ef

def main(cf):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Évaluation sur : {device}")
    
    # --- INITIALISATION WANDB ---
    os.environ["WANDB__SERVICE_WAIT"] = "300" #[cite: 6]
    wandb.login() #[cite: 6]
    run_name = "eval-" + os.path.basename(cf.model_path) if cf.model_path else "eval-random-weights" #[cite: 6]
    wandb.init(project="mon-projet-pcn", config=cf, name=run_name, job_type="evaluation") #[cite: 6]

    datasets = get_CIFAR10_dataloaders(batch_size=cf.batch_size, subset_size=cf.subset_size) if 'CIFAR10' in cf.model_path else get_fmnist_dataloaders(batch_size=cf.batch_size, subset_size=cf.subset_size)
    train_loader = datasets["train"]
    val_loader = datasets["val"]
    
    bpc_model = VGG5_bPC_Paper(
        num_labels=cf.num_labels, 
        rep_neurons=cf.rep_neurons, 
        alpha_gen=cf.alpha_gen, 
        alpha_disc=cf.alpha_disc
    ).to(device)
    
    if cf.model_path and os.path.exists(cf.model_path):
        bpc_model.load_state_dict(torch.load(cf.model_path, map_location=device))
        print(f"Poids du modèle chargés depuis {cf.model_path}")
    else:
        print("ATTENTION: Évaluation avec des poids aléatoires !")

    evaluate_generation(bpc_model, device, cf)
    evaluate_discrimination(bpc_model, val_loader, device, cf)
    # Nouveaux tests génératifs !
    evaluate_reconstruction(bpc_model, val_loader, device, cf)
    evaluate_inpainting(bpc_model, val_loader, device, cf, missing_ratio=0.3)
    evaluate_inpainting(bpc_model, val_loader, device, cf, missing_ratio=0.5)
    
    tsne_dataset = get_CIFAR10_dataloaders(batch_size=1000, subset_size=1000) if 'CIFAR10' in cf.model_path else get_fmnist_dataloaders(batch_size=1000, subset_size=1000)
    plot_tsne_layers(bpc_model, tsne_dataset["val"], device)
    
    wandb.finish() #[cite: 6]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script d'évaluation bPC (Discrimination, Génération, t-SNE)")
    parser.add_argument("--model_path", type=str, default="", help="Chemin vers le fichier .pt du modèle entraîné")
    parser.add_argument("--batch_size", type=int, default=512, help="Taille de batch pour l'évaluation")
    parser.add_argument("--subset_size", type=int, default=None, help="Taille du sous-ensemble")
    parser.add_argument("--rep_neurons", type=int, default=256, help="Nombre de neurones de représentation")
    
    args = parser.parse_args()

    cf = AttrDict()
    cf.model_path = args.model_path
    cf.batch_size = args.batch_size
    cf.subset_size = args.subset_size
    cf.num_labels = 10
    cf.rep_neurons = 256
    # On réutilise les mêmes vitesses de relaxation optimales
    cf.lr_x_eval = 0.00192827 #[cite: 5]
    cf.lr_x_gen = 0.00316244 #[cite: 5]
    
    cf.rep_neurons = 256 # Toujours actif
    cf.alpha_gen = 0.0000001 #[cite: 5]
    cf.alpha_disc = 1.0 #[cite: 5]
    
    cf.infer_steps_eval = 100 # T_eval = 100 itérations au lieu de 20[cite: 5]
    cf.infer_steps_gen = 100 # Idem pour la génération[cite: 5]
    main(cf)