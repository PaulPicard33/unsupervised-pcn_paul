import torch
import torch.optim as optim
import sys 
import os 
HOME = os.path.join(os.path.expanduser("~"), 'Desktop/unsupervised-pcn')
SRCDIR = os.path.join(HOME, 'src')
if SRCDIR not in sys.path:
    sys.path.insert(0, SRCDIR)

from models import VGG5_bPC_Paper
from datasets import get_fmnist_dataloaders


def main():
    # --- CONFIGURATION LOCALE DE TEST ---
    # Pour un test rapide sur ton ordi, mets test_local = True
    test_local = True
    
    epochs = 2 if test_local else 25 #[cite: 1]
    batch_size = 128 if test_local else 1024 #[cite: 1]
    subset_size = 500 if test_local else None # Limite le dataset à 500 images pour tester
    # ------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #[cite: 1]
    print(f"Exécution sur : {device}")

    # Chargement des données
    datasets = get_fmnist_dataloaders(batch_size=batch_size, subset_size=subset_size)
    
    # Instanciation du modèle avec 266 neurones (10 + 256)[cite: 1]
    bpc_model = VGG5_bPC_Paper(num_labels=10, rep_neurons=256, alpha_gen=1e-4, alpha_disc=1.0).to(device) #[cite: 1]
    optimizer_theta = optim.AdamW(bpc_model.parameters(), lr=1e-4, weight_decay=1e-4) #[cite: 1]

    # Masque pour le clampage partiel (fige les 10 premières composantes)[cite: 1]
    latent_mask = torch.zeros(266, device=device) #[cite: 1]
    latent_mask[:10] = 1.0 #[cite: 1]

    print("Début de l'entraînement bPC (Algorithme exact du papier)...") #[cite: 1]

    for epoch in range(epochs):
        bpc_model.train() #[cite: 1]
        total_energy = 0.0
        
        for batch_idx, (images, labels) in enumerate(datasets["train"]):
            x1_input = images.to(device) #[cite: 1]
            current_batch_size = x1_input.size(0)
            
            # Tenseur latent complet (266) : one-hot pour les 10 premiers[cite: 1]
            xL_label = torch.zeros((current_batch_size, 266), device=device) #[cite: 1]
            xL_label[:, :10] = torch.nn.functional.one_hot(labels, num_classes=10).float() #[cite: 1]
            
            # 1. Initialisation par balayage ascendant[cite: 1]
            x_init = bpc_model.bottom_up_sweep(x1_input) #[cite: 1]
            x_init[0] = x1_input #[cite: 1]
            x_init[-1][:, :10] = xL_label[:, :10] #[cite: 1]
            
            # 2. Inférence itérative avec T=32[cite: 1]
            x_inferred = bpc_model.infer(
                x_init, 
                clamped_indices=[0], 
                steps=32, 
                lr_x=0.01, 
                partial_clamp=(5, latent_mask.unsqueeze(0)), #[cite: 1]
                activity_decay=1e-3 #[cite: 1]
            )
            
            # 3. Mise à jour des poids[cite: 1]
            optimizer_theta.zero_grad() #[cite: 1]
            loss = bpc_model.compute_energy(x_inferred) #[cite: 1]
            loss.backward() #[cite: 1]
            optimizer_theta.step() #[cite: 1]
            
            total_energy += loss.item()
            
            if test_local:
                print(f"  Batch {batch_idx+1} - Loss: {loss.item():.4f}")

        print(f"Epoch [{epoch+1}/{epochs}] - Énergie totale: {total_energy:.4f}")

if __name__ == "__main__":
    main()