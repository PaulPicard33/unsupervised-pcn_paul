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

def evaluate_generation(model, device, cf):
    model.eval()
    print("--- Évaluation de la capacité de génération ---")
    
    labels = torch.arange(10).to(device)
    batch_size = 10
    latent_dim = cf.num_labels + cf.rep_neurons
    
    dummy_input = torch.randn(batch_size, 3, 32, 32, device=device)
    x_init = model.bottom_up_sweep(dummy_input)
    x_init = [torch.randn_like(tensor) * 0.1 for tensor in x_init]
    
    latent_mask = torch.zeros(latent_dim, device=device)
    latent_mask[:cf.num_labels] = 1.0
    
    x_label = torch.zeros((batch_size, latent_dim), device=device)
    x_label[:, :cf.num_labels] = F.one_hot(labels, num_classes=cf.num_labels).float()
    x_init[-1][:, :cf.num_labels] = x_label[:, :cf.num_labels]
    
    x_inferred = model.infer(
        x_init,
        clamped_indices=[], 
        steps=cf.infer_steps_gen,
        lr_x=cf.lr_x_gen,
        partial_clamp=(model.L - 1, latent_mask.unsqueeze(0))
    )
    
    generated_images = x_inferred[0].detach().cpu()
    
    os.makedirs("results", exist_ok=True)
    fig, axes = plt.subplots(1, 10, figsize=(15, 2))
    for i in range(10):
        img = generated_images[i].squeeze().numpy()
        img = (img + 1.0) / 2.0
        img = np.clip(img, 0, 1)
        axes[i].set_title(f"Label {i}")
        axes[i].axis('off')
    plt.tight_layout()
    
    # Sauvegarde locale ET envoi sur WandB[cite: 6]
    plt.savefig("results/generated_images.png") #[cite: 6]
    wandb.log({"eval/generated_images": wandb.Image(fig, caption="Images générées (Classes 0-9)")}) #[cite: 6]
    print("Images générées avec succès et envoyées sur W&B !") #[cite: 6]
    plt.close(fig) #[cite: 6]

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

    evaluate_discrimination(bpc_model, val_loader, device, cf)
    evaluate_generation(bpc_model, device, cf)
    
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
    cf.rep_neurons = args.rep_neurons
    cf.alpha_gen = 1e-4
    cf.alpha_disc = 1.0
    cf.infer_steps_eval = 20
    cf.lr_x_eval = 0.01
    cf.infer_steps_gen = 100
    cf.lr_x_gen = 0.05

    main(cf)