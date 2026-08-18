import os
# Force XLA/JAX à ne pas pré-allouer 90% de la VRAM
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
import os
import argparse

from pcn.datasets import get_CIFAR10_dataloaders, get_fmnist_dataloaders


# =============================================================================
# VODE
# =============================================================================

class Vode:
    """
    Nœud du graphe de croyances.
      h       : état courant (optimisé pendant l'inférence E-step)
      u       : prédiction reçue (calculée par les couches voisines)
      frozen  : si True, h est fixé et n'est jamais mis à jour
      is_latent : True pour le latent_vode (neurones libres) → lr différent
    """
    def __init__(self, shape, frozen=False, is_latent=False, device="cpu"):
        self.h = torch.zeros(shape, device=device)
        self.u = torch.zeros(shape, device=device)
        self.frozen    = frozen
        self.is_latent = is_latent

    def to(self, device):
        self.h = self.h.to(device)
        self.u = self.u.to(device)
        return self
class UnifiedUp(nn.Module):
    def __init__(self, in_features, size_label, size_latent):
        super().__init__()
        # Deux matrices séparées pour préserver tes deux learning rates
        self.fc_label = nn.Linear(in_features, size_label)
        self.fc_latent = nn.Linear(in_features, size_latent)

    def forward(self, x):
        # Le graphe génère une sortie 1D à population unique (taille 266)
        out_label = self.fc_label(x)
        out_latent = self.fc_latent(x)
        return torch.cat([out_label, out_latent], dim=1)


class UnifiedDown(nn.Module):
    def __init__(self, size_label, size_latent, out_features):
        super().__init__()
        self.size_label = size_label
        # Deux matrices séparées (un seul biais suffit pour la somme)
        self.fc_label = nn.Linear(size_label, out_features)
        self.fc_latent = nn.Linear(size_latent, out_features, bias=False)

    def forward(self, pop_1d):
        # Le graphe reçoit une entrée 1D à population unique (taille 266)
        x_label = pop_1d[:, :self.size_label]
        x_latent = pop_1d[:, self.size_label:]
        return self.fc_label(x_label) + self.fc_latent(x_latent)

# =============================================================================
# MODÈLE bPC VGG5 avec neurones libres (AddLatent)
# =============================================================================

class bPC_VGG(nn.Module):
    """
    Architecture VGG5 bidirectionnelle avec deux espaces latents distincts :

      vode[0]      → 10 neurones SUPERVISÉS (labels, frozen pendant l'entraînement)
      latent_vode  → 256 neurones LIBRES (non supervisés, optimisés séparément)
      vodes[1..5]  → états intermédiaires des blocs conv
      vode[-1]     → image (frozen)

    La passe DOWN injecte le latent via :
        fc_down(label) + latent_layer_down(latent)   → vode[2]
    ce qui est la `combination_fn_pre` de AddLatent :
        l(input + ld(latent))   avec combination_idx=2 (pre-activation)

    La passe UP extrait le latent en parallèle du label :
        latent = latent_layer_up(hidden_representation)
    depuis le même espace intermédiaire que la prédiction du label.
    """

    def __init__(self, input_channels=1, input_size=(28, 28),
                 output_size=10, latent_dim=256,
                 latent_var=None,     # = alpha_up/alpha_down dans train_cnn_pcax
                 device="cpu"):
        super().__init__()
        self.latent_dim  = latent_dim
        self.output_size = output_size
        self.latent_var  = latent_var  # variance du prior sur le latent
        self.eval_mode    = False  # True pendant l'évaluation pour figer les vodes
        # ── Pipeline UP (image → label) ──────────────────────────────────────
        self.conv1 = nn.Conv2d(input_channels, 128, kernel_size=3, padding=1, stride=1)
        self.pool1 = nn.MaxPool2d(2, 2)      # 28→14
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1)
        self.pool2 = nn.MaxPool2d(2, 2)      # 14→7
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1)
        self.pool3 = nn.MaxPool2d(2, 2)      # 7→3
        self.conv4 = nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1)
        self.pool4 = nn.MaxPool2d(2, 2)      # 3→1
        # flatten_size = 512 * 1 * 1 = 512
        # Dans __init__, après les définitions de pool :
        h, w = input_size
        for _ in range(4):          # 4 blocs MaxPool(2,2)
            h, w = h // 2, w // 2
        self.final_h    = h
        self.final_w    = w
        self.flatten_size = 512 * h * w   # 512 pour fMNIST, 2048 pour CIFAR10        
        # ── Pipeline UP & DOWN Unifiés ──
        self.unified_up = UnifiedUp(self.flatten_size, output_size, latent_dim)
        self.unified_down = UnifiedDown(output_size, latent_dim, self.flatten_size)

        # On remplace les ConvTranspose2d par des blocs Resize + Conv2d
        # Note : on conserve les mêmes dimensions de canaux pour correspondre à tes Vodes
        self.up_sample = nn.Upsample(scale_factor=2, mode='nearest')

        self.deconv4 = nn.Sequential(
            self.up_sample,
            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1)
        )
        self.deconv3 = nn.Sequential(
            self.up_sample,
            nn.Conv2d(512, 256, kernel_size=3, padding=1, stride=1)
        )
        self.deconv2 = nn.Sequential(
            self.up_sample,
            nn.Conv2d(256, 128, kernel_size=3, padding=1, stride=1)
        )
        self.deconv1 = nn.Sequential(
            self.up_sample,
            nn.Conv2d(128, input_channels, kernel_size=3, padding=1, stride=1)
        )

        self.act          = nn.GELU()
        self.out_act_down = nn.Tanh()
        # Recalculer les dims spatiales à chaque pool
        h, w = input_size
        pool_shapes = []
        for channels in [128, 256, 512, 512]:
            h, w = h // 2, w // 2
            pool_shapes.append((channels, h, w))
        # pool_shapes[0] = après pool1, ..., pool_shapes[3] = après pool4

        self.vodes = [
            Vode((output_size,),              frozen=True,  device=device),  # [0] label
            Vode((self.flatten_size,),        frozen=False, device=device),  # [1] flatten
            Vode(pool_shapes[3],              frozen=False, device=device),  # [2] après pool4
            Vode(pool_shapes[2],              frozen=False, device=device),  # [3] après pool3
            Vode(pool_shapes[1],              frozen=False, device=device),  # [4] après pool2
            Vode(pool_shapes[0],              frozen=False, device=device),  # [5] après pool1
            Vode((input_channels, *input_size), frozen=True, device=device), # [-1] image
        ]
        # ── Vode latent (neurones libres) ─────────────────────────────────────
        # Séparé des vodes principaux, optimisé avec un lr différent (lr_x_latent)
        # et des poids dédiés (latent_layer_up/down) mis à jour avec lr_p_latent.
        # Son énergie est 0.5*(h - u)²  ou  0.5*(h - u)²/latent_var si latent_var fourni.
        # u est mis à jour par la passe UP : u = latent_layer_up(hidden_flatten)
        # u est mis à zéro pendant la passe DOWN (bias_latent = zeros → u ≈ 0)
        # → pendant la passe DOWN, l'énergie du latent_vode force h → 0 (prior gaussien)
        self.latent_vode = Vode((latent_dim,), frozen=False, is_latent=True, device=device)

    def to(self, device, **kwargs):
        super().to(device, **kwargs)
        for v in self.vodes:
            v.to(device)
        self.latent_vode.to(device)
        return self
    def eval(self,**kwargs):
        super().eval(**kwargs)
        self.eval_mode = True


    # ── Énergie du vode latent ────────────────────────────────────────────────

    def _latent_energy(self):
        """
        Reproduit se_energy_latent de AddLatent.
        - Si u == 0 (passe DOWN, prior actif) : énergie normalisée par 1.0
        - Si u != 0 (passe UP, latent prédit)  : énergie normalisée par latent_var
        Le latent_var = alpha_up / alpha_down dans train_cnn_pcax.py.
        Quand alpha_gen=1e-7, alpha_disc=1.0 → latent_var = 1e-7 : prior très faible,
        le latent est quasi-libre (peu pénalisé).
        """
        h, u = self.latent_vode.h, self.latent_vode.u
        e = h - u
        if self.latent_var is not None:
            # Même logique que le jax.lax.cond : si u≈0 → var=1.0, sinon → latent_var
            var = 1.0 if u.abs().sum().item() == 0 else self.latent_var
        else:
            var = 1.0
        return (0.5 * (e * e) / var).sum()

    # ── Passe feedforward (initialisation) ────────────────────────────────────

    def init_ff(self, x_label, y_image, is_up=True):
        with torch.no_grad():
            if is_up:
                # Passe UP complète avec les bons indices (4 à 1)
                z = self.pool1(self.act(self.conv1(y_image)))
                self.vodes[4].h = z.clone()
                
                z = self.pool2(self.act(self.conv2(z)))
                self.vodes[3].h = z.clone()
                
                z = self.pool3(self.act(self.conv3(z)))
                self.vodes[2].h = z.clone()
                
                z = self.pool4(self.act(self.conv4(z)))
                self.vodes[1].h = z.clone()
                
                z_flat = z.flatten(start_dim=1)
                
                # Initialisation unifiée du sommet
                u_top = self.unified_up(z_flat)
                latent_u = u_top[:, self.output_size:]
                self.latent_vode.u = latent_u.clone()
                self.latent_vode.h = latent_u.clone()
            else:
                # Passe DOWN unifiée avec les bons indices (1 à 4)
                top_population = torch.cat([x_label, self.latent_vode.h], dim=1)
                z = self.act(self.unified_down(top_population)).reshape(-1, 512, self.final_h, self.final_w)
                self.vodes[1].h = z.clone()
                
                z = self.act(self.deconv4(z))
                self.vodes[2].h = z.clone()
                
                z = self.act(self.deconv3(z))
                self.vodes[3].h = z.clone()
                
                z = self.act(self.deconv2(z))
                self.vodes[4].h = z.clone()
                # --- LA CORRECTION : Calcul de l'image ---
                z_img = self.out_act_down(self.deconv1(z))
                self.vodes[-1].h = z_img.clone()

    # ── Calcul d'énergie bPC ──────────────────────────────────────────────────

    def compute_energy(self, x_label, y_image, alpha_up=1.0, alpha_down=1.0, weighted=True):
        
        # 1. Création de la population 1D unique à la volée
        top_population = torch.cat([self.vodes[0].h, self.latent_vode.h], dim=1)
        
        # ── PASSE UP ──
        u_up_4 = self.pool1(self.act(self.conv1(self.vodes[-1].h)))
        u_up_3 = self.pool2(self.act(self.conv2(self.vodes[4].h)))
        u_up_2 = self.pool3(self.act(self.conv3(self.vodes[3].h)))
        u_up_1_flat = self.pool4(self.act(self.conv4(self.vodes[2].h))).flatten(start_dim=1)
        
        # Prédiction unifiée séparée à la volée
        u_up_top = self.unified_up(u_up_1_flat)
        u_up_label = u_up_top[:, :self.output_size]
        u_up_latent = u_up_top[:, self.output_size:]

        e_up = 0.0
        e_up += 0.5 * ((self.vodes[4].h - u_up_4) ** 2).sum()
        e_up += 0.5 * ((self.vodes[3].h - u_up_3) ** 2).sum()
        e_up += 0.5 * ((self.vodes[2].h - u_up_2) ** 2).sum()
        
        # Énergie des labels
        e_up += 0.5 * ((self.vodes[0].h - u_up_label) ** 2).sum() 
        # Énergie du latent (pondérée par la variance)
        e_up += 0.5 * ((self.latent_vode.h - u_up_latent) ** 2).sum() / self.latent_var

        # ── PASSE DOWN ──
        # La passe descendante exploite enfin la synergie totale !
        u_down_1_flat = self.act(self.unified_down(top_population))
        u_down_1 = u_down_1_flat.reshape(-1, 512, self.final_h, self.final_w)
        
        u_down_2 = self.act(self.deconv4(self.vodes[1].h))
        u_down_3 = self.act(self.deconv3(self.vodes[2].h))
        u_down_4 = self.act(self.deconv2(self.vodes[3].h))
        u_down_img = self.out_act_down(self.deconv1(self.vodes[4].h))

        e_down = 0.0
        e_down += 0.5 * ((self.vodes[1].h - u_down_1) ** 2).sum()
        e_down += 0.5 * ((self.vodes[2].h - u_down_2) ** 2).sum()
        e_down += 0.5 * ((self.vodes[3].h - u_down_3) ** 2).sum()
        e_down += 0.5 * ((self.vodes[4].h - u_down_4) ** 2).sum()
        e_down += 0.5 * ((self.vodes[-1].h - u_down_img) ** 2).sum() 
        
        e_down += 0.5 * ((self.latent_vode.h - 0.0) ** 2).sum() / 1.0

        if weighted:
            return alpha_up * e_up + alpha_down * e_down
        else:
            return e_up + e_down

    # ── E-step : inférence ────────────────────────────────────────────────────

    def infer(self, x_label, y_image, T=32,
              lr_h=0.001, lr_h_latent=None,
              alpha_up=1.0, alpha_down=1.0):
        """
        Optimise les h des Vodes intermédiaires ET du latent_vode par T pas de SGD.

        Deux optimiseurs distincts, comme dans infer_on_batch_with_free_latent :
          optim_h       : SGD(lr_h)        pour les Vodes intermédiaires normaux
          optim_h_latent: SGD(lr_h_latent) pour le latent_vode (lr_x_latent dans pcax)

        Le lr du latent_vode est mis à l'échelle par 1/alpha_down dans pcax
        (sgd_scaled avec scale=1/alpha_down). On reproduit ce scaling ici.
        """
        if lr_h_latent is None:
            lr_h_latent = lr_h  # fallback si non fourni

        # Activer requires_grad sur les états à optimiser
        for v in self.vodes:
            if not v.frozen:
                v.h = v.h.detach().requires_grad_(True)
        self.latent_vode.h = self.latent_vode.h.detach().requires_grad_(True)

        # Deux optimiseurs séparés (comme optim_h et optim_h_latent dans pcax)
        optimizer_h = torch.optim.SGD(
            [v.h for v in self.vodes if not v.frozen],
            lr=lr_h, momentum=0.0
        )
        optimizer_h_latent = torch.optim.SGD(
            [self.latent_vode.h],
            lr=lr_h_latent/alpha_down ,   # scaling par 1/alpha_down (sgd_scaled)
            momentum=0.0
        )

        for _ in range(T):
            optimizer_h.zero_grad()
            optimizer_h_latent.zero_grad()

            E = self.compute_energy(x_label, y_image, alpha_up, alpha_down, weighted=True)
            E.backward()

            # Sécurité contre les falaises d'énergie
            torch.nn.utils.clip_grad_norm_([v.h for v in self.vodes if not v.frozen], max_norm=1.0)
            torch.nn.utils.clip_grad_norm_([self.latent_vode.h], max_norm=1.0)
            # Sécurité contre les falaises d'énergie
            optimizer_h.step()
            optimizer_h_latent.step()
            if self.eval_mode:
                if not self.vodes[-1].frozen:
                    self.vodes[-1].h.data.clamp_(-1.0, 1.0) # Ajuste à [0, 1] si tu n'utilises pas la normalisation -1/1
            
        # Détacher proprement avant le W-step
        for v in self.vodes:
            if not v.frozen:
                v.h = v.h.detach()
        self.latent_vode.h = self.latent_vode.h.detach()

    # ── W-step ────────────────────────────────────────────────────────────────

    def w_step(self, x_label, y_image, optimizer_w, optimizer_w_latent):
        """
        Met à jour les poids après l'inférence.

        Deux optimiseurs distincts pour les poids, comme dans train_on_batch_free :
          optimizer_w       : AdamW(lr_p)        pour toutes les couches sauf latent
          optimizer_w_latent: AdamW(lr_p_latent) pour latent_layer_up/down uniquement

        energy_weights n'applique PAS les alphas (weighted=False).
        Les h sont déjà détachés par infer() → le gradient ne remonte que dans les poids.
        """
        optimizer_w.zero_grad()
        optimizer_w_latent.zero_grad()

        E = self.compute_energy(x_label, y_image, weighted=False)
        (E/y_image.size(0)).backward()

        optimizer_w.step()
        optimizer_w_latent.step()


# =============================================================================
# CONFIGURATION
# =============================================================================

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


# =============================================================================
# BOUCLE D'ENTRAÎNEMENT
# =============================================================================

def main(cf):
    run_name = f"{cf.dataset}"
    if cf.subset_size is not None:
        run_name += f"-subset={cf.subset_size}"
    run_name += f"-T={cf.infer_steps}-ep={cf.n_epochs}"

    os.environ["WANDB__SERVICE_WAIT"] = "300"
    wandb.login()
    wandb.init(project="bpc-vgg", config=dict(cf), name=run_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    if cf.dataset == "fmnist":
        datasets      = get_fmnist_dataloaders(batch_size=cf.batch_size, subset_size=cf.subset_size)
        input_channels = 1
        input_size     = (28, 28)
    elif cf.dataset == "CIFAR10":
        datasets      = get_CIFAR10_dataloaders(batch_size=cf.batch_size, subset_size=cf.subset_size)
        input_channels = 3
        input_size     = (32, 32)

    # ── Modèle ────────────────────────────────────────────────────────────────
    # latent_var = alpha_up / alpha_down, comme dans train_cnn_pcax.py ligne 243
    bpc_model = bPC_VGG(
        input_channels=input_channels,
        input_size=input_size,
        output_size=cf.num_labels,
        latent_dim=cf.latent_dim,
        latent_var=cf.alpha_disc / cf.alpha_gen,   # = 1e-7 → prior très faible
        device=str(device),
    ).to(device)

    # ── Optimiseurs des POIDS ─────────────────────────────────────────────────
    # Même séparation que train_cnn_pcax.py :
    #   optim_w       → toutes les couches SAUF latent_layer_up/down  (lr_p)
    #   optim_w_latent→ latent_layer_up + latent_layer_down           (lr_p_latent)
    # ── Optimiseurs des POIDS ──
    latent_param_ids = {
        id(p) for p in list(bpc_model.unified_up.fc_latent.parameters())
                      + list(bpc_model.unified_down.fc_latent.parameters())
    }
    params_main   = [p for p in bpc_model.parameters() if id(p) not in latent_param_ids]
    params_latent = [p for p in bpc_model.parameters() if id(p)     in latent_param_ids]
    
    # Dans main(), lors de la création des optimiseurs :
    decay_params_main = []
    no_decay_params_main = []
    
    for name, param in bpc_model.named_parameters():
        if id(param) not in latent_param_ids:
            if 'bias' in name:
                no_decay_params_main.append(param)
            else:
                decay_params_main.append(param)

    # On retire le weight decay des biais
    optimizer_w = torch.optim.AdamW([
        {'params': decay_params_main, 'weight_decay': cf.weight_decay},
        {'params': no_decay_params_main, 'weight_decay': 0.0}
    ], lr=cf.lr_p)
    #optimizer_w = torch.optim.AdamW(params_main, lr=cf.lr_p, weight_decay=cf.weight_decay)     
    optimizer_w_latent = torch.optim.AdamW(
        params_latent, lr=cf.lr_p_latent, weight_decay=cf.weight_decay
    )
    # Dans main(), juste après la création de optimizer_w et optimizer_w_latent :
    
    total_steps = len(datasets["train"]) * cf.n_epochs
    
    # Scheduler pour les poids principaux
    scheduler_w = torch.optim.lr_scheduler.OneCycleLR(
        optimizer_w,
        max_lr=cf.lr_p * 1.1,                  # Le pic à 1.1 * lr_p
        total_steps=total_steps,               # Mise à jour à chaque batch
        pct_start=0.1,                         # Warmup sur 10% du temps
        div_factor=1.1,                        # Départ à (max_lr / 1.1) = lr_p
        final_div_factor=11.0,                 # Fin à (max_lr / 11.0) = 0.1 * lr_p
        cycle_momentum=False                   # Requis car on utilise AdamW
    )

    # Scheduler pour les poids latents
    scheduler_w_latent = torch.optim.lr_scheduler.OneCycleLR(
        optimizer_w_latent,
        max_lr=cf.lr_p_latent * 1.1,
        total_steps=total_steps,
        pct_start=0.1,
        div_factor=1.1,
        final_div_factor=11.0,
        cycle_momentum=False
    )

    print("Début de l'entraînement bPC...")

    for epoch in range(cf.n_epochs):
        bpc_model.train()
        total_energy = 0.0

        for batch_idx, (y_image, label_int) in enumerate(datasets["train"]):
            y_image= y_image.to(device)
            if label_int.ndim == 1:
                x_label = F.one_hot(label_int, num_classes=cf.num_labels).float().to(device)
            else:
                x_label = label_int.float().to(device)

            # Fixer les extrémités
            bpc_model.vodes[0].h  = x_label
            bpc_model.vodes[-1].h = y_image

            # 1. Init feedforward (passe UP par défaut)
            bpc_model.init_ff(x_label, y_image, is_up=True)

            # 2. E-step : deux lr distincts pour h normal et h latent
            bpc_model.infer(
                x_label, y_image,
                T=cf.infer_steps,
                lr_h=cf.lr_x,
                lr_h_latent=cf.lr_x_latent,
                alpha_up=cf.alpha_disc,
                alpha_down=cf.alpha_gen,
            )

            # 3. W-step : deux optimiseurs distincts pour poids normaux et latents
            bpc_model.w_step(x_label, y_image, optimizer_w, optimizer_w_latent)

            with torch.no_grad():
                batch_e = bpc_model.compute_energy(
                    x_label, y_image,
                    alpha_up=cf.alpha_disc, alpha_down=cf.alpha_gen,
                    weighted=True
                ).item()
            total_energy += batch_e

            if batch_idx % cf.log_freq == 0:
                wandb.log({"batch_energy": batch_e, "epoch": epoch})
            scheduler_w.step()
            scheduler_w_latent.step()
        avg_energy = total_energy / len(datasets["train"])
        
        
        if batch_idx % cf.log_freq == 0:
            wandb.log({
                "batch_energy": batch_e, 
                "lr_w": optimizer_w.param_groups[0]["lr"],
                "lr_w_latent": optimizer_w_latent.param_groups[0]["lr"]
            })
        print(f"Epoch [{epoch+1}/{cf.n_epochs}] — Énergie moy. : {avg_energy:.4f}")

    wandb.finish()
    os.makedirs("models", exist_ok=True)
    torch.save(bpc_model.state_dict(), f"models/bpc-{run_name}.pt")
    print(f"Modèle sauvegardé sous models/bpc-{run_name}.pt")


# =============================================================================
# POINT D'ENTRÉE
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entraînement bPC VGG5 avec neurones libres")
    parser.add_argument("--dataset",     choices=["fmnist", "CIFAR10"], default="CIFAR10", help="Nom du dataset ")
    parser.add_argument("--subset_size", type=int,   default=None)
    parser.add_argument("--n_epochs",    type=int,   default=50)
    parser.add_argument("--batch_size",  type=int,   default=256)
    parser.add_argument("--infer_steps", type=int,   default=32)
    args = parser.parse_args()

    cf = AttrDict()
    cf.dataset      = args.dataset
    cf.subset_size  = args.subset_size
    cf.n_epochs     = args.n_epochs
    cf.batch_size   = args.batch_size
    cf.log_freq     = 10
    cf.num_labels   = 10
    cf.latent_dim   = 256   # neurones libres

    # Poids principaux (toutes couches sauf latent_layer_up/down)
    cf.lr_p         = 0.0001415926
    cf.weight_decay = 0.0003497999

    # Poids latents (latent_layer_up/down, lr 10x plus grand)
    cf.lr_p_latent  = 0.0015553778

    # Scalings bPC
    cf.alpha_gen    = 1e-7   # passe DOWN (génération) quasi-inactive
    cf.alpha_disc   = 1.0    # passe UP  (discrimination) dominante

    # Inférence des états h
    cf.infer_steps  = args.infer_steps
    cf.lr_x         = 0.00192827    # SGD pour les Vodes intermédiaires normaux
    cf.lr_x_latent  = 0.00316244    # SGD pour le latent_vode (lr_x_free dans pcax)

    main(cf)