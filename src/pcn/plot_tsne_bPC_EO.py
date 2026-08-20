import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import wandb

# Importations spécifiques à ton projet
from bpc_e import PCE, PC_States, PCESkipConnection
from datamodules import CIFAR10 # Remplace si tu utilises CIFAR100 ou TinyImageNet
from get_arch import get_architecture_bpc

def extract_and_plot_tsne(model, dataloader, device, num_samples=1000, save_path="results/tsne_pce_layers.png"):
    print(f"\n--- Extraction des représentations ({num_samples} images) ---")
    model.eval()
    model.to(device)
    
    # Récupération du nombre de couches montantes
    num_layers = len(model.layers_up)
    layers_data = {f"Couche_UP_{i+1}": [] for i in range(num_layers)}
    labels_list = []
    
    samples_collected = 0

    # 1. Extraction par mini-batchs (Protection Anti-OOM)
    for x, y in dataloader:
        if samples_collected >= num_samples:
            break
            
        batch_size = x.size(0)
        # Couper le batch si on dépasse num_samples
        if samples_collected + batch_size > num_samples:
            x = x[:num_samples - samples_collected]
            y = y[:num_samples - samples_collected]
            batch_size = x.size(0)
            
        x = x.to(device)
        
        with torch.no_grad():
            # Passe Feedforward manuelle pour capturer chaque état intermédiaire
            current_state = x
            for i, layer in enumerate(model.layers_up):
                current_state = layer(current_state)
                # On aplatit le tenseur (start_dim=1) et on l'envoie direct sur le CPU
                layers_data[f"Couche_UP_{i+1}"].append(current_state.flatten(start_dim=1).cpu().numpy())
                
        labels_list.append(y.cpu().numpy())
        samples_collected += batch_size

    # 2. Concaténation des listes en gros tableaux Numpy
    print("--- Concaténation des données ---")
    layers_dict = {k: np.concatenate(v, axis=0) for k, v in layers_data.items()}
    labels = np.concatenate(labels_list, axis=0)
    
    # 3. Calcul du t-SNE et affichage (Totalement sur CPU)
    print("--- Calcul des projections t-SNE (Cela peut prendre quelques minutes) ---")
    fig, axs = plt.subplots(1, num_layers, figsize=(5 * num_layers, 5))
    
    # Sécurité si une seule couche
    if num_layers == 1:
        axs = [axs]
        
    fig.suptitle("Évolution des Représentations (Passe Montante PCE)", fontsize=16)

    for i, (layer_name, features) in enumerate(layers_dict.items()):
        print(f"Calcul t-SNE pour {layer_name} (Dimensions: {features.shape})...")
        # Paramétrage standard robuste pour le t-SNE
        tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
        reduced_features = tsne.fit_transform(features)
        
        scatter = axs[i].scatter(
            reduced_features[:, 0], reduced_features[:, 1], 
            c=labels, cmap='tab10', s=10, alpha=0.7
        )
        axs[i].set_title(layer_name)
        axs[i].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    wandb.log({"t-SNE Plot": wandb.Image(save_path)})  # Log vers Weights & Biases
    plt.close()
    print(f"\nSuccès ! Graphique t-SNE sauvegardé sous : {save_path}")
import torchvision.utils as vutils

def generate_and_plot_classes(model, device, num_classes=10, img_shape=(3, 32, 32), save_path="results/generated_classes.png"):
    """
    Génère une image à partir de zéro pour chaque classe (0 à 9) en minimisant
    l'énergie du réseau selon la méthode PCE (optimisation des erreurs).
    """
    print(f"\n--- Génération des images par classe ({num_classes} classes) ---")
    model.eval()
    
    # 1. Préparation des cibles (y) : matrice identité pour du one-hot encoding
    # (Adapte si tes labels y dans bpc_e sont formatés différemment)
    y_target = torch.eye(num_classes).to(device)
    
    # 2. Initialisation de la "toile vierge" (x) avec un léger bruit pour briser la symétrie
    x_gen = (torch.randn((num_classes, *img_shape), device=device) * 0.1).requires_grad_(True)
    
    # On fige les poids du modèle (inférence uniquement)
    for p in model.parameters():
        p.requires_grad_(False)
        
    # 3. Initialisation des erreurs via la fonction interne de bpc_e
    model.init_zero_errors(x_gen)
    
    # 4. Configuration de l'optimiseur
    # On optimise simultanément les pixels de l'image ET les erreurs latentes
    optimizer = torch.optim.Adam([x_gen] + model.errors, lr=0.05)
    
    # Sauvegarde des alphas d'origine pour la phase 2
    orig_alpha_up = model.alpha_up
    orig_alpha_down = model.alpha_down
    
    # --- PHASE 1 : Voie descendante stricte ---
    # On force le label à écraser le bruit de l'image (dictature du label)
    model.alpha_up = 0.0
    model.alpha_down = 1.0
    
    print("Phase 1 : Inférence descendante (T=1000)...")
    for _ in range(1000):
        optimizer.zero_grad()
        E = model.E(x_gen, y_target)
        E.backward()
        optimizer.step()
        
        # Borner l'image pour rester dans l'espace colorimétrique valide
        with torch.no_grad():
            x_gen.clamp_(-1.0, 1.0)
            
    # --- PHASE 2 : Relaxation et harmonisation ---
    # On réactive la voie montante pour lisser et affiner les détails
    model.alpha_up = orig_alpha_up
    model.alpha_down = orig_alpha_down
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.01  # Réduction du pas pour le fignolage
        
    print("Phase 2 : Harmonisation bidirectionnelle (T=500)...")
    for _ in range(500):
        optimizer.zero_grad()
        E = model.E(x_gen, y_target)
        E.backward()
        optimizer.step()
        
        with torch.no_grad():
            x_gen.clamp_(-1.0, 1.0)
            
    # Rétablir les gradients des paramètres pour ne pas bloquer les fonctions suivantes
    for p in model.parameters():
        p.requires_grad_(True)
        
    # 5. Affichage et sauvegarde
    print(f"Sauvegarde des images générées sous : {save_path}")
    
    # Dé-normalisation pour l'affichage (si tes images sont entraînées entre -1 et 1)
    imgs_to_plot = (x_gen.detach().clone() + 1) / 2.0 
    imgs_to_plot = imgs_to_plot.clamp(0, 1)
    
    grid = vutils.make_grid(imgs_to_plot, nrow=5, padding=2, normalize=False)
    
    plt.figure(figsize=(10, 4))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.title("Génération par Inférence d'Énergie (PCE)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    wandb.log({"Generated Classes": wandb.Image(save_path)})  # Log vers Weights & Biases
    plt.close()
    
    return x_gen.detach()

if __name__ == "__main__":
    # --- Configuration ---
    # Aligne ces paramètres avec ceux qui ont servi à générer ton .pt
    DATASET_NAME = "CIFAR10"
    MODEL_NAME = "VGG5"
    ACT_FN = "gelu"
    BATCH_SIZE = 256
    WEIGHTS_PATH = "models/bpc_eo0.001.pt" # Modifie avec ton chemin
    OPTIM_MODE = "errors" # "errors" (PCE), "states" (PC_States), ou "skip" (PCESkipConnection)
    NUM_IMAGES_TSNE = 1000
    wandb.init(project="unsupervised-pcn", name=f"t-SNE_{MODEL_NAME}_{DATASET_NAME}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Appareil détecté : {device}")

    # 1. Chargement des données
    datamodule = CIFAR10(BATCH_SIZE, is_test=False) # Remplacer par la bonne classe si besoin
    datamodule.setup("test")
    val_loader = datamodule.test_édataloader() # On utilise le set de validation pour le t-SNE

    # 2. Instanciation de l'architecture
    architecture = get_architecture_bpc(dataset=DATASET_NAME, model_name=MODEL_NAME, activation=ACT_FN)
    
    # 3. Sélection dynamique de la classe
    if OPTIM_MODE == "states":
        ModelClass = PC_States
    elif OPTIM_MODE == "skip":
        ModelClass = PCESkipConnection
    else:
        ModelClass = PCE

    # Initialisation du module Lightning[cite: 1, 2]
    model = ModelClass(
        architecture=architecture,
        iters=5, e_lr=0.001, w_lr=0.0002, alpha_up=1.0, alpha_down=1e-8 # Paramètres dummy pour l'eval
    )
    
    # 4. Chargement des poids depuis le .pt
    # (Si c'est un state_dict brut PyTorch)
    state_dict = torch.load(WEIGHTS_PATH, map_location=device, weights_only=True)
    
    # Si le fichier .pt contient des clés avec un préfixe inattendu (ex: 'model.layers...'), 
    # le paramètre strict=False évite les crashs bloquants
    model.load_state_dict(state_dict, strict=False) 
    
    # Lancement du plot
    extract_and_plot_tsne(model, val_loader, device, num_samples=NUM_IMAGES_TSNE)
    # 2. Génération des images par classe
    generate_and_plot_classes(
        model=model, 
        device=device, 
        num_classes=10, 
        img_shape=(3, 32, 32), # Ajuste si tu passes sur TinyImageNet (3, 64, 64)
        save_path="results/generated_classes_pce.png"
    )
    wandb.finish()