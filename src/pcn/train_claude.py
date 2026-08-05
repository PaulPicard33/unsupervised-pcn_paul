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
        self.fc_up = nn.Linear(self.flatten_size, output_size)

        # ── Couches du latent ─────────────────────────────────────────────────
        # Dans AddLatent :
        #   hidden_dim = np.prod(vodes[1].shape) = flatten_size = 512
        #   latent_layer_up   : Linear(hidden_dim → latent_dim)  [512 → 256]
        #   latent_layer_down : Linear(latent_dim → hidden_dim)  [256 → 512]
        self.latent_layer_up   = nn.Linear(self.flatten_size, latent_dim)
        self.latent_layer_down = nn.Linear(latent_dim, self.flatten_size)

        # ── Pipeline DOWN (label → image) ─────────────────────────────────────
        # combination_fn_pre : fc_down(label) + latent_layer_down(latent) → vode[2]
        # Autrement dit : fc_down prend le label, latent_layer_down injecte le latent,
        # et leur somme entre dans relu → reshape → deconvolutions.
        self.fc_down = nn.Linear(output_size, self.flatten_size)
        self.deconv4 = nn.ConvTranspose2d(512, 512, kernel_size=3, padding=1, stride=2, output_padding=1)  # 1→3
        self.deconv3 = nn.ConvTranspose2d(512, 256, kernel_size=3, padding=1, stride=2, output_padding=1)  # 3→7
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1, stride=2, output_padding=1)  # 7→14
        self.deconv1 = nn.ConvTranspose2d(128, input_channels, kernel_size=3, padding=1, stride=2, output_padding=1)  # 14→28

        self.act          = nn.LeakyReLU(negative_slope=0.01)
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
        """
        Initialise h de tous les Vodes intermédiaires + latent_vode.
        is_up=True (défaut pcax) : image → label → latent extrait en parallèle.
        """
        with torch.no_grad():
            if is_up:
                # Passe UP complète
                z = self.act(self.pool1(self.conv1(y_image)))
                self.vodes[5].h = z.clone()
                z = self.act(self.pool2(self.conv2(z)))
                self.vodes[4].h = z.clone()
                z = self.act(self.pool3(self.conv3(z)))
                self.vodes[3].h = z.clone()
                z = self.act(self.pool4(self.conv4(z)))
                self.vodes[2].h = z.clone()
                z_flat = z.flatten(start_dim=1)
                self.vodes[1].h = z_flat.clone()
                # vode[0] est frozen (label), on ne l'initialise pas

                # Initialisation du latent_vode par la passe UP
                # latent_vode.u = latent_layer_up(hidden_flatten)
                # latent_vode.h = u  (règle "ff" : h ← u)
                latent_u = self.latent_layer_up(z_flat)
                self.latent_vode.u = latent_u.clone()
                self.latent_vode.h = latent_u.clone()
            else:
                # Passe DOWN : label + latent → image
                # combination_fn_pre : fc_down(label) + latent_layer_down(latent)
                latent = self.latent_vode.h
                z = self.act(
                    self.fc_down(x_label) + self.latent_layer_down(latent)
                ).reshape(-1, 512, self.final_h, self.final_w)
                self.vodes[2].h = z.clone()
                z = self.act(self.deconv4(z))
                self.vodes[3].h = z.clone()
                z = self.act(self.deconv3(z))
                self.vodes[4].h = z.clone()
                z = self.act(self.deconv2(z))
                self.vodes[5].h = z.clone()
                # vode[-1] est frozen (image)

    # ── Calcul d'énergie bPC ──────────────────────────────────────────────────

    def compute_energy(self, x_label, y_image,
                       alpha_up=1.0, alpha_down=1.0, weighted=True):
        """
        Reproduit energy() de models.py + la logique AddLatent.

        PASSE UP :
          Chaque vode[i].u est mis à jour par les conv ascendantes.
          latent_vode.u = latent_layer_up(hidden_flatten)
          → latent_vode contribue à e_up avec sa propre variance (latent_var)

        PASSE DOWN :
          latent_vode.u = 0  (bias_latent = zeros → prior gaussien centré)
          → la passe down pénalise h_latent ≠ 0 (regularisation L2)
          La combination_fn_pre injecte latent dans la passe down :
              fc_down(label) + latent_layer_down(latent_vode.h) → vode[2].u

        Les deux énergies sont calculées sur leurs Vodes respectifs
        (frozen exclus), puis l'énergie du latent_vode est ajoutée aux deux.
        """

        # ── PASSE UP ─────────────────────────────────────────────────────────
        z = self.act(self.conv1(self.vodes[-1].h))
        z = self.pool1(z)
        self.vodes[5].u = z

        z = self.act(self.conv2(self.vodes[5].h))
        z = self.pool2(z)
        self.vodes[4].u = z

        z = self.act(self.conv3(self.vodes[4].h))
        z = self.pool3(z)
        self.vodes[3].u = z

        z = self.act(self.conv4(self.vodes[3].h))
        z = self.pool4(z)
        self.vodes[2].u = z

        z_flat = z.flatten(start_dim=1)
        self.vodes[1].u = z_flat

        # Prédiction label (vode[0] est frozen, son énergie n'est pas comptée)
        self.vodes[0].u = self.fc_up(self.vodes[1].h)

        # Prédiction latent depuis le même hidden_flatten
        self.latent_vode.u = self.latent_layer_up(self.vodes[1].h)

        e_up = sum(
            0.5 * ((v.h - v.u) ** 2).sum()
            for v in self.vodes if not v.frozen
        ) + self._latent_energy()  # latent contribue à l'énergie UP

        # ── PASSE DOWN ───────────────────────────────────────────────────────
        # Prior latent : u = 0 (bias_latent dans AddLatent)
        self.latent_vode.u = torch.zeros_like(self.latent_vode.h)

        # combination_fn_pre : fc_down(label) + latent_layer_down(latent_vode.h)
        # puis activation → reshape → déconvolutions
        latent = self.latent_vode.h
        z = self.act(
            self.fc_down(self.vodes[0].h) + self.latent_layer_down(latent)
        ).reshape(-1, 512, self.final_h, self.final_w)
        self.vodes[2].u = z      # écrase le .u de la passe UP (intentionnel)

        z = self.act(self.deconv4(self.vodes[2].h))
        self.vodes[3].u = z

        z = self.act(self.deconv3(self.vodes[3].h))
        self.vodes[4].u = z

        z = self.act(self.deconv2(self.vodes[4].h))
        self.vodes[5].u = z

        self.vodes[-1].u = self.out_act_down(self.deconv1(self.vodes[5].h))

        e_down = sum(
            0.5 * ((v.h - v.u) ** 2).sum()
            for v in self.vodes if not v.frozen
        ) + self._latent_energy()  # latent contribue aussi à l'énergie DOWN (prior)

        return (alpha_up * e_up + alpha_down * e_down) if weighted else (e_up + e_down)

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
            lr=lr_h_latent / alpha_down,   # scaling par 1/alpha_down (sgd_scaled)
            momentum=0.0
        )

        for _ in range(T):
            optimizer_h.zero_grad()
            optimizer_h_latent.zero_grad()

            E = self.compute_energy(x_label, y_image, alpha_up, alpha_down, weighted=True)
            E.backward()

            optimizer_h.step()
            optimizer_h_latent.step()

            
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
    latent_param_ids = {
        id(p) for p in list(bpc_model.latent_layer_up.parameters())
                      + list(bpc_model.latent_layer_down.parameters())
    }
    params_main   = [p for p in bpc_model.parameters() if id(p) not in latent_param_ids]
    params_latent = [p for p in bpc_model.parameters() if id(p)     in latent_param_ids]

    optimizer_w = torch.optim.AdamW(
        params_main, lr=cf.lr_p, weight_decay=cf.weight_decay
    )
    optimizer_w_latent = torch.optim.AdamW(
        params_latent, lr=cf.lr_p_latent, weight_decay=cf.weight_decay
    )
    scheduler_w        = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_w, T_max=cf.n_epochs, eta_min=cf.lr_p * 0.1
    )
    scheduler_w_latent = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_w_latent, T_max=cf.n_epochs, eta_min=cf.lr_p_latent * 0.1
    )

    print("Début de l'entraînement bPC...")

    for epoch in range(cf.n_epochs):
        bpc_model.train()
        total_energy = 0.0

        for batch_idx, (y_image, label_int) in enumerate(datasets["train"]):
            y_image = y_image.to(device)
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

        avg_energy = total_energy / len(datasets["train"])
        scheduler_w.step()
        scheduler_w_latent.step()

        wandb.log({
            "epoch":              epoch,
            "train_energy_avg":   avg_energy,
            "lr_w":               optimizer_w.param_groups[0]["lr"],
            "lr_w_latent":        optimizer_w_latent.param_groups[0]["lr"],
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