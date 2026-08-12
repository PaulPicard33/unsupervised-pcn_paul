import torch
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np
import wandb
from sklearn.manifold import TSNE

# Import de ta nouvelle architecture et des dataloaders
from pcn import optim
from train_claude import bPC_VGG, AttrDict
from pcn.datasets import get_CIFAR10_dataloaders, get_fmnist_dataloaders

def evaluate_discrimination(model, dataloader, device, cf):
    correct = 0
    total = 0
    
    print("\n--- Évaluation de la capacité de discrimination (Inférence T=100) ---")
    
    for batch_idx, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        batch_size = images.size(0)
        
        # 1. CRITIQUE : Créer le tenseur avec la dimension de batch ! [batch_size, 10]
        x_label_dummy = torch.zeros((batch_size, cf.num_labels), device=device)
        
        # 2. Configuration des "Frozen"
        model.vodes[-1].frozen = True  # L'image est une observation fixe
        model.vodes[0].frozen = False  # Le label est inconnu, on le laisse libre
        
        # 3. Assigner les tenseurs AVEC LEUR DIMENSION DE BATCH aux Vodes
        model.vodes[-1].h = images
        model.vodes[0].h = x_label_dummy  # <-- C'est cette ligne qui manquait !
        
        # 4. Initialisation Bottom-Up
        model.init_ff(x_label_dummy, images, is_up=True)
        
        # 5. Astuce : On donne au label la prédiction initiale feedforward
        with torch.no_grad():
            pred_logits = model.unified_up.fc_label(model.vodes[1].h.flatten(start_dim=1))
            # On écrase le Vode par les vrais logits (taille [batch_size, 10])
            model.vodes[0].h = pred_logits.clone().detach() 
            
        # 6. Inférence (Descente d'énergie sur les états)
        model.infer(
            x_label=model.vodes[0].h, 
            y_image=images,
            T=cf.infer_steps_eval,
            lr_h=cf.lr_x_eval,
            lr_h_latent=cf.lr_x_latent,
            alpha_up=cf.alpha_disc,
            alpha_down=cf.alpha_gen
        )
        
        # 7. Lecture de la prédiction finale (le tenseur est bien 2D maintenant)
        preds = model.vodes[0].h.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += batch_size
        
        # Verrouiller à nouveau le label par sécurité pour le prochain batch
        model.vodes[0].frozen = True
        
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx}/{len(dataloader)} - Accuracy partielle : {100 * correct / total:.2f}%")
            
    accuracy = 100 * correct / total
    print(f"\n=> Précision (Accuracy) finale sur le set de validation : {accuracy:.2f}%")
    return accuracy

def plot_tsne_layers(model, dataloader, device, num_samples=1000):
    print(f"\n--- Calcul des projections t-SNE pour l'analyse des Vodes ({num_samples} images) ---")
    model.eval()
    
    # 1. Extraction d'un subset pour le t-SNE
    images_list, labels_list = [], []
    samples_collected = 0
    for x, y in dataloader:
        images_list.append(x)
        labels_list.append(y)
        samples_collected += x.size(0)
        if samples_collected >= num_samples:
            break
            
    images = torch.cat(images_list)[:num_samples].to(device)
    labels = torch.cat(labels_list)[:num_samples].cpu().numpy()
    
    # 2. HARD RESET des mémoires du modèle à la taille num_samples
    for v in model.vodes:
        v.h = torch.zeros((num_samples, *v.h.shape[1:]), device=device)
        v.u = torch.zeros((num_samples, *v.u.shape[1:]), device=device)
        
    model.latent_vode.h = torch.zeros((num_samples, model.latent_dim), device=device)
    model.latent_vode.u = torch.zeros((num_samples, model.latent_dim), device=device)

    # 3. Inférence pour extraire les représentations
    x_label_dummy = torch.zeros((num_samples, model.output_size), device=device)
    model.vodes[-1].frozen = True
    model.vodes[0].frozen = False
    
    model.vodes[-1].h = images
    model.vodes[0].h = x_label_dummy
    
    model.init_ff(x_label_dummy, images, is_up=True)
    
    # On utilise infer avec les hyperparamètres standard (à ajuster si besoin)
    model.infer(
        x_label=model.vodes[0].h, y_image=images,
        T=32, lr_h=0.0019, lr_h_latent=0.0031,
        alpha_up=1.0, alpha_down=1e-7
    )
    
    # 4. Le bon dictionnaire avec les indices décalés (1 à 4)
    layers_dict = {
        'Espace Latent 256': model.latent_vode.h.detach().cpu().numpy(),
        'Vode 1 (Post Conv4)': model.vodes[1].h.flatten(start_dim=1).detach().cpu().numpy(),
        'Vode 2 (Post Conv3)': model.vodes[2].h.flatten(start_dim=1).detach().cpu().numpy(),
        'Vode 3 (Post Conv2)': model.vodes[3].h.flatten(start_dim=1).detach().cpu().numpy(),
        'Vode 4 (Post Conv1)': model.vodes[4].h.flatten(start_dim=1).detach().cpu().numpy(),
    }
    
    # ... (Suite du code avec TSNE(n_components=2) et l'affichage matplotlib)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, (name, data) in enumerate(layers_dict.items()):
        print(f"Génération t-SNE pour {name}...")
        tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
        tsne_results = tsne.fit_transform(data)
        
        scatter = axes[i].scatter(
            tsne_results[:, 0], tsne_results[:, 1], 
            c=labels, cmap='tab10', s=15, alpha=0.8
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
def evaluate_generation(model, device, cf, nm_classes=10):
    print("\n--- Évaluation de la Génération Conditionnelle ---")
    model.eval()
    
    labels = torch.arange(nm_classes, device=device)
    x_label = F.one_hot(labels, num_classes=nm_classes).float()
    y_image_dummy = torch.zeros((nm_classes, 3, 32, 32), device=device)

    # 1. HARD RESET : On force tout le réseau à la taille nm_classes (10)
    for v in model.vodes:
        v.h = torch.zeros((nm_classes, *v.h.shape[1:]), device=device)
        v.u = torch.zeros((nm_classes, *v.u.shape[1:]), device=device)
        
    model.latent_vode.h = torch.zeros((nm_classes, cf.latent_dim), device=device)
    model.latent_vode.u = torch.zeros((nm_classes, cf.latent_dim), device=device)

    # 2. Assignation
    model.vodes[0].h = x_label
    model.vodes[-1].h = y_image_dummy
    model.vodes[0].frozen = True
    model.vodes[-1].frozen = False
    
    model.init_ff(x_label, y_image_dummy, is_up=False)

    # 3. Inférence générative (L'énergie descendante est débridée !)
    model.infer(
        x_label=x_label, 
        y_image=y_image_dummy,
        T=cf.infer_steps_eval,
        lr_h=cf.lr_x_eval,
        lr_h_latent=cf.lr_x_latent,
        alpha_up=cf.alpha_disc,
        alpha_down=1.0  # <--- CHANGEMENT CRITIQUE : Remplace cf.alpha_gen pour libérer les pixels
    )
    
    generated_images = model.vodes[-1].h.detach()
    
    fig, axs = plt.subplots(1, nm_classes, figsize=(15, 2))
    imgs = generated_images.cpu().numpy() / 2 + 0.5
    imgs = np.clip(np.transpose(imgs, (0, 2, 3, 1)), 0, 1)
    
    for i in range(nm_classes):
        axs[i].imshow(imgs[i])
        axs[i].set_title(f"Classe {i}")
        axs[i].axis("off")
    plt.savefig("results/conditional_generation.png")
    wandb.log({"eval/conditional_generation": wandb.Image(plt, caption="Génération Conditionnelle par Classe")}) #[cite: 6]
    plt.close()
def evaluate_reconstruction(model, dataloader, device, cf):
    print("\n--- Évaluation de la Reconstruction (MSE) ---")
    mse_total = 0.0
    total_images = 0
    
    for x_images, _ in dataloader:
        x_images = x_images.to(device)
        batch_size = x_images.size(0)
        
        # 1. Inférence UP : Trouver l'état latent de l'image
        x_label_dummy = torch.zeros((batch_size, cf.num_labels), device=device)
        model.vodes[-1].frozen = True
        model.vodes[0].frozen = False
        
        model.vodes[-1].h = x_images
        model.vodes[0].h = x_label_dummy
        model.init_ff(x_label_dummy, x_images, is_up=True)
        
        model.infer(
            x_label=model.vodes[0].h, y_image=x_images,
            T=cf.infer_steps_eval, lr_h=cf.lr_x_eval, lr_h_latent=cf.lr_x_latent,
            alpha_up=cf.alpha_disc, alpha_down=cf.alpha_disc
        )
        
        # 2. Inférence DOWN : Régénérer l'image depuis l'état latent figé
        inferred_latent = model.latent_vode.h.detach()
        inferred_label = model.vodes[0].h.detach()
        
        model.vodes[0].frozen = True
        model.latent_vode.frozen = True
        model.vodes[-1].frozen = False
        
        dummy_reconstruction = torch.zeros_like(x_images)
        model.vodes[-1].h = dummy_reconstruction
        
        model.infer(
            x_label=inferred_label, y_image=dummy_reconstruction,
            T=cf.infer_steps_eval, lr_h=cf.lr_x_eval, lr_h_latent=0.0, # Latent figé
            alpha_up=cf.alpha_disc, alpha_down=cf.alpha_gen
        )
        
        reconstructed_images = model.vodes[-1].h.detach()
        mse_total += F.mse_loss(reconstructed_images, x_images, reduction='sum').item()
        total_images += batch_size
        
        # Rétablir les états
        model.latent_vode.frozen = False
        
    mse_final = mse_total / (total_images * 3 * 32 * 32)
    print(f"MSE de reconstruction latente : {mse_final:.5f}")
    wandb.log({"eval/reconstruction_mse": mse_final}) #[cite: 6]
    return mse_final
class LinearProbe(nn.Module):
    def __init__(self, rep_size, n_classes):
        super().__init__()
        self.lin = nn.Linear(rep_size, n_classes)

    def forward(self, x):
        return self.lin(x)

def evaluate_linear_probing(model, dataloader, device, cf):
    print("\n--- Entraînement de la Sonde Linéaire (Linear Probing) ---")
    
    latents_list, labels_list = [], []
    
    # 1. Extraction des représentations (sans gradient)
    for x_images, y_labels in dataloader:
        x_images = x_images.to(device)
        batch_size = x_images.size(0)
        
        x_label_dummy = torch.zeros((batch_size, cf.num_labels), device=device)
        model.vodes[-1].frozen = True
        model.vodes[0].frozen = False
        
        model.vodes[-1].h = x_images
        model.vodes[0].h = x_label_dummy
        model.init_ff(x_label_dummy, x_images, is_up=True)
        
        model.infer(
            x_label=model.vodes[0].h, y_image=x_images,
            T=cf.infer_steps_eval, lr_h=cf.lr_x_eval, lr_h_latent=cf.lr_x_latent,
            alpha_up=cf.alpha_disc, alpha_down=cf.alpha_gen
        )
        
        # On extrait l'avant-dernière couche (flatten_size) pour l'évaluer
        latents_list.append(model.vodes[1].h.flatten(start_dim=1).detach().cpu())
        labels_list.append(y_labels.cpu())

    X = torch.cat(latents_list)
    Y = torch.cat(labels_list)
    
    # 2. Entraînement du classifieur
    dataset = torch.utils.data.TensorDataset(X, Y)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(0.8 * len(dataset)), len(dataset) - int(0.8 * len(dataset))])
    probe_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True)
    probe_val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=128, shuffle=False)
    probe = LinearProbe(model.flatten_size, cf.num_labels).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    best_acc = 0.0
    for epoch in range(10): # 10 époques suffisent pour un modèle linéaire
        probe.train()
        for data, target in probe_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = probe(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
        # Évaluation rapide
        probe.eval()
        correct = 0
        with torch.no_grad():
            for data, target in probe_val_loader:
                data, target = data.to(device), target.to(device)
                pred = probe(data).argmax(dim=1)
                correct += (pred == target).sum().item()
        acc = correct / len(val_dataset)
        if acc > best_acc:
            best_acc = acc
            
    print(f"Accuracy de décodage latent (Sonde linéaire) : {best_acc * 100:.2f}%")
    wandb.log({"eval/linear_probe_accuracy": best_acc})
    return best_acc

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
    evaluate_generation(bpc_model, device, cf)
    evaluate_reconstruction(bpc_model, val_loader, device, cf)
    evaluate_linear_probing(bpc_model, val_loader, device, cf)
    
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