from pcn import utils
from pcn.layers import FCLayer, FCPlusLayer
import torch
from torch import nn
import numpy as np
import wandb
import torch.optim as optim

class PCModel(nn.Module):
    def __init__(self, nodes, mu_dt, act_fn, use_bias=False, kaiming_init=False, positive=False, device=utils.DEVICE):
        super().__init__()
        self.nodes = nodes
        self.mu_dt = mu_dt
        self.act_fn = act_fn
        self.n_nodes = len(nodes)
        self.n_layers = len(nodes) - 1
        self.device = device
        
        if not positive:
            self.layers = []
            for l in range(self.n_layers):
                _act_fn = utils.Linear() if (l == self.n_layers - 1) else self.act_func(self.act_fn)
                _use_bias = False if (l == self.n_layers - 1) else use_bias

                layer = FCLayer(
                    in_size=nodes[l],
                    out_size=nodes[l + 1],
                    act_fn=_act_fn,
                    use_bias=_use_bias,
                    kaiming_init=kaiming_init,
                    device=device
                )
                self.layers.append(layer)
        else:
            self.layers = [
                FCPlusLayer(
                    in_size=nodes[l],
                    out_size=nodes[l + 1],
                    act_fn=self.act_func(self.act_fn),
                    use_bias=use_bias,
                    kaiming_init=kaiming_init,
                    device=device
                )
                for l in range(self.n_layers)
            ]
        self.layers = nn.ModuleList(self.layers)

    def act_func(self, act_fn):
        if act_fn == 'sigmoid':
            return utils.Sigmoid()
        elif act_fn == 'tanh':
            return utils.Tanh()
        elif act_fn == 'relu':
            return utils.ReLU()
        elif act_fn == 'linear':
            return utils.Linear()
        else:
            raise ValueError(f'Unsupported activation function: {act_fn}')

    def reset(self):
        self.preds = [[] for _ in range(self.n_nodes)]
        self.errs = [[] for _ in range(self.n_nodes)]
        self.mus = [[] for _ in range(self.n_nodes)]

    def reset_mus(self, batch_size, init_std):
        for l in range(self.n_layers):
            self.mus[l] = utils.set_tensor(
                torch.empty(batch_size, self.layers[l].in_size).normal_(mean=0, std=init_std), self.device
            )

    def set_target(self, target):
        self.mus[-1] = target.clone()
    
    def set_input(self, inp):
        self.mus[0] = inp.clone()

    def train_batch(self, img_batch, n_iters, init_std=0.05, fixed_preds=False):
        batch_size = img_batch.size(0)
        self.reset()
        self.reset_mus(batch_size, init_std)
        self.set_target(img_batch)
        self.updates(n_iters, fixed_preds=fixed_preds)
        self.update_grads()
    
    def eval_batch(self, img_batch, n_iters, init_std=0.05, fixed_preds=False):
        batch_size = img_batch.size(0)
        self.reset()
        self.reset_mus(batch_size, init_std)
        self.set_target(img_batch)
        self.updates(n_iters, fixed_preds=fixed_preds)

    def test_batch(self, img_batch, n_iters, step_tolerance=1e-5, init_std=0.05, fixed_preds=False):
        batch_size = img_batch.size(0)
        self.reset()
        self.reset_mus(batch_size, init_std)
        self.set_target(img_batch)
        self.test_updates(n_iters, step_tolerance, fixed_preds=fixed_preds)

    def replay_batch(self, label_batch, n_iters, step_tolerance=1e-5, init_std=0.05, fixed_preds=False):
        batch_size = label_batch.size(0)
        self.reset()
        self.reset_mus(batch_size, init_std)
        self.set_input(label_batch)
        self.replay_updates(n_iters, step_tolerance, fixed_preds)

    def generate_batch(self, label_batch, n_iters, step_tolerance=1e-5, init_std=0.05, fixed_preds=False):
        batch_size = label_batch.size(0)
        self.reset()
        self.reset_mus(batch_size, init_std)
        self.set_input(label_batch)
        self.generation_updates(n_iters, step_tolerance, fixed_preds)

    def recall_batch(self, img_batch_corrupt, n_iters, indices, step_tolerance=1e-5, init_std=0.05, fixed_preds=False):
        batch_size = img_batch_corrupt.size(0)
        self.reset()
        self.reset_mus(batch_size, init_std)
        self.set_target(img_batch_corrupt)
        self.recall_updates(n_iters, step_tolerance, indices, fixed_preds=fixed_preds)

    def precision_recall_batch(self, img_batch_corrupt, n_iters, n_cut, step_tolerance=1e-5, init_std=0.05, fixed_preds=False):
        batch_size = img_batch_corrupt.size(0)
        self.reset()
        self.reset_mus(batch_size, init_std)
        self.set_target(img_batch_corrupt)
        self.precision_recall_updates(n_iters, step_tolerance, n_cut, fixed_preds=fixed_preds)

    def updates(self, n_iters, fixed_preds=False):
        self.preds[0] = utils.set_tensor(torch.zeros(self.mus[0].shape), self.device)
        self.errs[0] = self.mus[0] - self.preds[0]
        for n in range(1, self.n_nodes):
            self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1])
            self.errs[n] = self.mus[n] - self.preds[n]

        for itr in range(n_iters):
            for l in range(self.n_layers): # mus[-1] is fixed to the image
                delta = self.layers[l].backward(self.errs[l + 1]) - self.errs[l]
                self.mus[l] = self.mus[l] + self.mu_dt * delta

            for n in range(1, self.n_nodes):
                if not fixed_preds:
                    self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1])
                self.errs[n] = self.mus[n] - self.preds[n]

    def replay_updates(self, n_iters, step_tolerance, fixed_preds=False):
        batch_size = self.mus[0].shape[0]
        self.plot_batch_errors = [[[] for n in range(self.n_nodes)] for m in range(batch_size)]
        self.preds[0] = utils.set_tensor(torch.zeros(self.mus[0].shape), self.device)
        self.errs[0] = self.mus[0] - self.preds[0]
        for n in range(1, self.n_nodes - 1):
            self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1])
            self.errs[n] = self.mus[n] - self.preds[n]
        self.preds[-1] = self.layers[-1].forward(self.mus[-2])
        self.errs[-1] = utils.set_tensor(torch.zeros(batch_size, self.layers[-1].out_size))        
        relative_diff = torch.empty(self.n_layers - 1, batch_size)
        for itr in range(n_iters): 
            for l in range(1, self.n_layers): # mus[-1] and mus[0] are fixed
                delta = self.layers[l].backward(self.errs[l + 1]) - self.errs[l]
                relative_diff[l-1] = delta.norm(dim=1)/self.mus[l].norm(dim=1)
                self.mus[l] = self.mus[l] + self.mu_dt * delta

            for n in range(1, self.n_nodes - 1):
                if not fixed_preds:
                    self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1])
                self.errs[n] = self.mus[n] - self.preds[n]

            for n in range(self.n_nodes):
                errors = self.get_errors(n)/self.nodes[n]
                for m in range(batch_size):
                    self.plot_batch_errors[m][n].append(errors[m])        

            if (relative_diff < step_tolerance).sum().item():
                # Replay
                n = self.n_nodes - 1
                if not fixed_preds: 
                    self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1]) 
                break

    def precision_recall_updates(self, n_iters, step_tolerance, n_cut, fixed_preds=False):
        batch_size = self.mus[0].shape[0]
        self.plot_batch_errors = [[[] for n in range(self.n_nodes)] for m in range(batch_size)]
        self.preds[0] = utils.set_tensor(torch.zeros(self.mus[0].shape), self.device)
        self.errs[0] = self.mus[0] - self.preds[0]
        for n in range(1, self.n_nodes):
            self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1])
            self.errs[n] = self.mus[n] - self.preds[n]
        self.errs[-1][:, n_cut:] = utils.set_tensor(torch.zeros_like(self.errs[-1][:, n_cut:]))
        relative_diff = torch.empty(self.n_layers - 1, batch_size)
        for itr in range(n_iters):           
            for l in range(1, self.n_layers): 
                delta = self.layers[l].backward(self.errs[l + 1]) - self.errs[l]
                relative_diff[l-1] = delta.norm(dim=1)/self.mus[l].norm(dim=1)
                self.mus[l] = self.mus[l] + self.mu_dt * delta

            for n in range(1, self.n_nodes):
                if not fixed_preds:
                    self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1])
                self.errs[n] = self.mus[n] - self.preds[n] 
            self.errs[-1][:, n_cut:] = utils.set_tensor(torch.zeros_like(self.errs[-1][:, n_cut:]))
            
            for n in range(self.n_nodes):
                errors = self.get_errors(n)/self.nodes[n]
                for m in range(batch_size):
                    self.plot_batch_errors[m][n].append(errors[m])

            if (relative_diff < step_tolerance).sum().item():
                break  

    def generation_updates(self, n_iters, step_tolerance, fixed_preds=False):
        batch_size = self.mus[0].shape[0]
        self.plot_batch_errors = [[[] for n in range(self.n_nodes)] for m in range(batch_size)]
        self.preds[0] = utils.set_tensor(torch.zeros(self.mus[0].shape), self.device)
        self.errs[0] = self.mus[0] - self.preds[0]
        for n in range(1, self.n_nodes - 1):
            self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1])
            self.errs[n] = self.mus[n] - self.preds[n]
        self.preds[-1] = self.layers[-1].forward(self.mus[-2])
        self.errs[-1] = utils.set_tensor(torch.zeros(batch_size, self.layers[-1].out_size))        
        relative_diff = torch.empty(self.n_layers - 1, batch_size)
        for itr in range(n_iters): 
            for l in range(self.n_layers): # mus[-1] is fixed
                delta = self.layers[l].backward(self.errs[l + 1]) - self.errs[l]
                relative_diff[l-1] = delta.norm(dim=1)/self.mus[l].norm(dim=1)
                self.mus[l] = self.mus[l] + self.mu_dt * delta

            for n in range(1, self.n_nodes - 1):
                if not fixed_preds:
                    self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1])
                self.errs[n] = self.mus[n] - self.preds[n]

            for n in range(self.n_nodes):
                errors = self.get_errors(n)/self.nodes[n]
                for m in range(batch_size):
                    self.plot_batch_errors[m][n].append(errors[m])        

            if (relative_diff < step_tolerance).sum().item():
                # Replay
                n = self.n_nodes - 1
                if not fixed_preds: 
                    self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1]) 
                break
        
    def test_updates(self, n_iters, step_tolerance, fixed_preds=False):
        batch_size = self.mus[0].shape[0]
        self.plot_batch_errors = [[[] for n in range(self.n_nodes)] for m in range(batch_size)]
        self.preds[0] = utils.set_tensor(torch.zeros(self.mus[0].shape), self.device)
        self.errs[0] = self.mus[0] - self.preds[0]
        for n in range(1, self.n_nodes):
            self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1])
            self.errs[n] = self.mus[n] - self.preds[n]
        relative_diff = torch.empty(self.n_layers, batch_size)
        for itr in range(n_iters):           
            for l in range(self.n_layers): # mus[-1] is fixed to the image
                delta = self.layers[l].backward(self.errs[l + 1]) - self.errs[l]
                relative_diff[l] = delta.norm(dim=1)/self.mus[l].norm(dim=1)
                self.mus[l] = self.mus[l] + self.mu_dt * delta                

            for n in range(1, self.n_nodes):
                if not fixed_preds:
                    self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1])
                self.errs[n] = self.mus[n] - self.preds[n]
            
            for n in range(self.n_nodes):
                errors = self.get_errors(n)/self.nodes[n]
                for m in range(batch_size):
                    self.plot_batch_errors[m][n].append(errors[m])

            if (relative_diff < step_tolerance).sum().item():
                break
        
    def recall_updates(self, n_iters, step_tolerance, indices, fixed_preds=False):
        batch_size = self.mus[0].shape[0]
        self.plot_batch_errors = [[[] for n in range(self.n_nodes)] for m in range(batch_size)]
        self.preds[0] = utils.set_tensor(torch.zeros(self.mus[0].shape), self.device)
        self.errs[0] = self.mus[0] - self.preds[0]
        for n in range(1, self.n_nodes):
            self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1])
            self.errs[n] = self.mus[n] - self.preds[n]
        relative_diff = torch.empty(self.n_layers, batch_size)
        for itr in range(n_iters):           
            for l in range(self.n_layers): 
                delta = self.layers[l].backward(self.errs[l + 1]) - self.errs[l]
                relative_diff[l] = delta.norm(dim=1)/self.mus[l].norm(dim=1)
                self.mus[l] = self.mus[l] + self.mu_dt * delta       
            # Recall pixels
            delta = - self.errs[-1]
            self.mus[-1][:, indices] = self.mus[-1][:, indices] + self.mu_dt * delta[:, indices]

            for n in range(1, self.n_nodes):
                if not fixed_preds:
                    self.preds[n] = self.layers[n - 1].forward(self.mus[n - 1])
                self.errs[n] = self.mus[n] - self.preds[n]            
            
            for n in range(self.n_nodes):
                errors = self.get_errors(n)/self.nodes[n]
                for m in range(batch_size):
                    self.plot_batch_errors[m][n].append(errors[m])

            if (relative_diff < step_tolerance).sum().item():
                break 

    def update_grads(self):
        for l in range(self.n_layers):
            self.layers[l].update_gradient(self.errs[l + 1])
    
    def get_errors(self, n): # losses 
        return torch.sum(self.errs[n] ** 2, dim=1).cpu()
    



class bPCModel(nn.Module):
    def __init__(self, nodes, mu_dt, act_fn, use_bias=False, kaiming_init=False, positive=False, device=utils.DEVICE,alpha_up=0,alpha_down=1.0):
        super().__init__()
        self.nodes = nodes
        self.mu_dt = mu_dt
        self.act_fn = act_fn
        self.n_nodes = len(nodes)
        self.n_layers = len(nodes) - 1
        self.device = device
        self.alpha_up= alpha_up
        self.alpha_down = alpha_down
        
        if not positive:
            self.uplayers = []
            self.downlayers = []
            for l in range(self.n_layers):
                _act_fn = utils.Linear() if (l == self.n_layers - 1) else self.act_func(self.act_fn)
                _use_bias = False if (l == self.n_layers - 1) else use_bias

                uplayer = FCLayer(
                    in_size=nodes[l],
                    out_size=nodes[l + 1],
                    act_fn=_act_fn,
                    use_bias=_use_bias,
                    kaiming_init=kaiming_init,
                    device=device
                )
                downlayer = FCLayer(
                    in_size=nodes[l+1],
                    out_size=nodes[l],
                    act_fn=_act_fn,
                    use_bias=_use_bias,
                    kaiming_init=kaiming_init,
                    device=device
                )
                self.uplayers.append(uplayer)
                self.downlayers.append(downlayer)
        else:
            self.uplayers = [
                FCPlusLayer(
                    in_size=nodes[l],
                    out_size=nodes[l + 1],
                    act_fn=self.act_func(self.act_fn),
                    use_bias=use_bias,
                    kaiming_init=kaiming_init,
                    device=device
                )
                for l in range(self.n_layers)
            ]
            self.downlayers = [FCPlusLayer(
                    in_size=nodes[l+1],
                    out_size=nodes[l],
                    act_fn=self.act_func(self.act_fn),
                    use_bias=use_bias,
                    kaiming_init=kaiming_init,
                    device=device
                )
                for l in range(self.n_layers)
            ]
        self.uplayers = nn.ModuleList(self.uplayers)
        self.downlayers = nn.ModuleList(self.downlayers)
    def reset(self):
        # Séparation des buffers pour les deux directions
        self.preds_up = [[] for _ in range(self.n_nodes)]
        self.preds_down = [[] for _ in range(self.n_nodes)]
        self.errs_up = [[] for _ in range(self.n_nodes)]
        self.errs_down = [[] for _ in range(self.n_nodes)]
        self.mus = [[] for _ in range(self.n_nodes)]
    def act_func(self, act_fn):
        if act_fn == 'sigmoid':
            return utils.Sigmoid()
        elif act_fn == 'tanh':
            return utils.Tanh()
        elif act_fn == 'relu':
            return utils.ReLU()
        elif act_fn == 'linear':
            return utils.Linear()
        else:
            raise ValueError(f'Unsupported activation function: {act_fn}')
    def reset_mus(self, batch_size, init_std):
        for l in range(self.n_layers):
            self.mus[l] = utils.set_tensor(
                torch.empty(batch_size, self.layers[l].in_size).normal_(mean=0, std=init_std), self.device
            )

    def compute_preds_and_errs(self, fixed_preds=False):
        # Flux "Up" (ex: Label vers Image dans l'indexation)
        for l in range(self.n_layers):
            if not fixed_preds or len(self.preds_up[l+1]) == 0:
                self.preds_up[l+1] = self.uplayers[l].forward(self.mus[l])
            # Multiplié par alpha_up selon l'équation d'énergie du papier
            self.errs_up[l+1] = self.alpha_up * (self.mus[l+1] - self.preds_up[l+1])
            
        # Flux "Down" (ex: Image vers Label)
        for l in range(self.n_layers):
            if not fixed_preds or len(self.preds_down[l]) == 0:
                self.preds_down[l] = self.downlayers[l].forward(self.mus[l+1])
            # Multiplié par alpha_down
            self.errs_down[l] = self.alpha_down * (self.mus[l] - self.preds_down[l])
    def updates(self, n_iters, clamped_nodes):
        """
        clamped_nodes : liste des indices des couches figées. 
        Ex: [0] pour l'image (classification), [-1] pour le label (génération), [0, -1] (entraînement)
        """
        with torch.no_grad():
            self.compute_preds_and_errs() # Calcul initial

            for itr in range(n_iters):
                for l in range(self.n_nodes):
                    
                    # --- L'astuce est ici : on saute la mise à jour si le noeud est figé ---
                    # On gère les indices négatifs (ex: -1 pour la dernière couche)
                    actual_l = l if l not in clamped_nodes and (l - self.n_nodes) not in clamped_nodes else None
                    if actual_l is None:
                        continue 

                    # Initialisation de la force (delta) qui va bouger l'activité du noeud l
                    delta = utils.set_tensor(torch.zeros_like(self.mus[l]), self.device)
                    
                    # --- 1. Forces directes (Le noeud l est la CIBLE des prédictions) ---
                    if l < self.n_nodes - 1:
                        # L'erreur venant d'en haut tire le noeud vers le haut
                        delta -= self.errs_down[l] 
                    if l > 0:
                        # L'erreur venant d'en bas tire le noeud vers le bas
                        delta -= self.errs_up[l]
                    
                    # --- 2. Forces de rétroaction (Le noeud l est la SOURCE des prédictions) ---
                    if l > 0:
                        # Rétroaction de l'erreur causée à la couche l-1
                        delta += self.downlayers[l-1].backward(self.errs_down[l-1])
                    if l < self.n_nodes - 1:
                        # Rétroaction de l'erreur causée à la couche l+1
                        delta += self.uplayers[l].backward(self.errs_up[l+1])

                    # Application du gradient (Descente d'énergie)
                    self.mus[l] = self.mus[l] + self.mu_dt * delta

                # Recalcul des erreurs avec les nouvelles activités
                self.compute_preds_and_errs()
    def update_grads(self):
        # Mise à jour purement locale
        for l in range(self.n_layers):
            # Le gradient pour uplayers[l] dépend de l'erreur post-synaptique générée en l+1
            self.uplayers[l].update_gradient(self.errs_up[l+1])
            
            # Le gradient pour downlayers[l] dépend de l'erreur post-synaptique générée en l
            self.downlayers[l].update_gradient(self.errs_down[l])
            
    def train_step(self, img_batch, label_batch, n_iters, clamped_nodes):
        """
        Effectue une seule itération d'inférence + mise à jour des poids pour un batch.
        """
        self.reset_mus(batch_size=img_batch.size(0), init_std=0.1)
        self.mus[0] = img_batch
        self.mus[-1] = label_batch
        
        # Inférence : on minimise l'énergie
        self.updates(n_iters, clamped_nodes=clamped_nodes)
        
        # Apprentissage : mise à jour des gradients
        self.update_grads()
        
        # Retourne les erreurs pour le logging (optionnel)
        return self.get_errors() 

    def get_errors(self):
        # Somme des erreurs pour le suivi
        return {n: torch.sum(self.errs[n] ** 2, dim=1).mean().item() for n in range(self.n_nodes)}
    
class bPCTrainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
    
    def train_epoch(self, data_loader, n_iters, clamped_nodes=[0, -1]):
        self.model.train() # Mode entraînement
        epoch_errors = []
        
        for batch_id, (img_batch, label_batch) in enumerate(data_loader):
            # 1. Le Trainer appelle le Moteur pour faire le travail sur le batch
            batch_errors = self.model.train_step(img_batch, label_batch, n_iters, clamped_nodes)
            
            # 2. Le Trainer gère l'optimisation
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # 3. Le Trainer gère le logging (WandB, console, etc.)
            epoch_errors.append(batch_errors)
            if batch_id % 100 == 0:
                print(f"Batch {batch_id}: {batch_errors}")
                
        return epoch_errors

    def infer_classification(self, img_batch, n_iters):
        """
        Mode Classification : On donne l'image, on laisse le label fluctuer.
        """
        self.model.eval() # Mode évaluation
        self.model.reset_mus(batch_size=img_batch.size(0), init_std=0.1)
        self.model.mus[0] = img_batch
        # On initialise le label avec du bruit
        self.model.mus[-1] = torch.randn_like(self.model.mus[-1])
        
        # On ne bloque QUE l'image
        self.model.updates(n_iters, clamped_nodes=[0])
        return self.model.mus[-1] # Retourne la prédiction du label

    def infer_reconstruction(self, label_batch, n_iters):
        """
        Mode Génération : On donne le label, on laisse l'image fluctuer.
        """
        self.model.eval()
        self.model.reset_mus(batch_size=label_batch.size(0), init_std=0.1)
        self.model.mus[-1] = label_batch
        # On initialise l'image avec du bruit
        self.model.mus[0] = torch.randn_like(self.model.mus[0])
        
        # On ne bloque QUE le label
        self.model.updates(n_iters, clamped_nodes=[-1])
        return self.model.mus[0] # Retourne l'image générée





class PCTrainer(object):
    def __init__(self, model, optimizer=None):
        self.model = model
        self.optimizer = optimizer

    def train(self, data_loader, epoch, n_iters, fixed_preds, log=True, log_freq=1000):
        """
        Return errors (losses weighted by the inverse of the number of nodes) in all layers averaged over the 
        training dataset
        """
        train_epoch_errors = [[] for _ in range(self.model.n_nodes)]
        n_batches = len(data_loader)
        for batch_id, (img_batch, label_batch) in enumerate(data_loader):   
            self.model.train_batch(img_batch, n_iters, fixed_preds=fixed_preds)

            t = epoch * n_batches + batch_id        
            self.optimizer.step(
                curr_epoch=epoch,
                curr_batch=batch_id,
                n_batches=n_batches,
                batch_size=img_batch.size(0),
                log=log and t%log_freq == 0,
            )
        
            # gather data for the current batch
            for n in range(self.model.n_nodes):
                errors = self.model.get_errors(n)/self.model.nodes[n]
                train_epoch_errors[n] += [errors.mean().item()]
            
            # log layer activations (except input) and weights            
            if log and t%log_freq == 0:
                for n in range(self.model.n_nodes - 1):
                    wandb.log({f'latents_{n}': wandb.Histogram(self.model.mus[n].cpu().detach())})
                    wandb.log({f'weights_{n}': wandb.Histogram(self.model.layers[n].weights.cpu().detach())})
                    if self.model.layers[n].use_bias:
                        wandb.log({f'bias_{n}': wandb.Histogram(self.model.layers[n].bias.cpu().detach())})

        # gather data for the full epoch
        train_errors = []
        for n in range(self.model.n_nodes):
            error = np.mean(train_epoch_errors[n])
            train_errors.append(error)
        return train_errors 
    
    def eval(self, img_batch, n_iters, fixed_preds):
        """
        Return errors in all layers averaged over an input validation batch
        """
        self.model.eval_batch(img_batch, n_iters, fixed_preds=fixed_preds)
        valid_errors = []
        for n in range(self.model.n_nodes):
            errors = self.model.get_errors(n)/self.model.nodes[n]
            valid_errors.append(errors.mean().item())
        return valid_errors
    
    def test(self, data_loader, n_iters, fixed_preds) -> float:
        """
        Return MSE between original and reconstructed images averaged over the testing dataset
        """
        test_mse = 0
        for img_batch, label_batch in data_loader:  
            self.model.test_batch(img_batch, n_iters, fixed_preds=fixed_preds)
            errors = self.model.get_errors(-1)/self.model.nodes[-1] # MSE
            test_mse += torch.sum(errors).item()
        n_batches = len(data_loader)
        batch_size = img_batch.shape[0]
        test_mse = test_mse/(n_batches*batch_size)

        return float(test_mse)


import torch
import torch.nn as nn
import torch.optim as optim

class VGG5_bPC_Paper(nn.Module):
    def __init__(self, num_labels=10, rep_neurons=256, alpha_gen=1e-4, alpha_disc=1.0,cifar=True):
        super().__init__()
        self.L = 6
        self.alpha_gen = alpha_gen
        self.alpha_disc = alpha_disc
        self.latent_dim = num_labels + rep_neurons 
        self.num_labels = num_labels
        self.cifar=cifar
        
        self.activation = nn.GELU() #

        self.V_convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1), nn.MaxPool2d(2, 2)),
            nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), nn.MaxPool2d(2, 2)),
            nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), nn.MaxPool2d(2, 2)),
            nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1), nn.MaxPool2d(2, 2))
        ])
        
        # SÉPARATION PHYSIQUE pour le Dual Optimizer
        self.V_linear_labels = nn.Sequential(nn.Flatten(), nn.Linear(2048, self.num_labels), nn.Identity())
        self.V_linear_free = nn.Sequential(nn.Flatten(), nn.Linear(2048, rep_neurons), nn.Identity())

        self.W_linear = nn.Sequential(nn.Linear(self.latent_dim, 2048), nn.Unflatten(1, (512, 2, 2)))
        self.W_convs = nn.ModuleList([
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sequential(nn.ConvTranspose2d(128, 3, kernel_size=3, stride=2, padding=1, output_padding=1), nn.Tanh())
        ])

        alpha_disc_L = torch.ones(self.latent_dim) * alpha_disc
        alpha_disc_L[num_labels:] = alpha_gen 
        self.register_buffer('alpha_disc_L', alpha_disc_L)

    def _forward_V(self, x, layer_idx):
        if layer_idx == 0:
            return self.V_convs[0](x)
        elif layer_idx < 4: 
            return self.V_convs[layer_idx](self.activation(x))
        # Reconstruction du tenseur complet (B, 266) à partir des deux flux
        logits = self.V_linear_labels(self.activation(x))
        free_rep = self.V_linear_free(self.activation(x))
        return torch.cat([logits, free_rep], dim=1)

    def _forward_W(self, x, layer_idx):
        if layer_idx == 4: return self.W_linear(self.activation(x))
        return self.W_convs[3 - layer_idx](self.activation(x))

    def compute_energy(self, x):
        energy_gen = 0.0
        energy_disc = 0.0
        for i in range(self.L - 1):
            pred_gen = self._forward_W(x[i+1], i)
            energy_gen += torch.sum((x[i] - pred_gen) ** 2) * (self.alpha_gen / 2)
            
            pred_disc = self._forward_V(x[i], i)
            diff_sq_disc = (x[i+1] - pred_disc) ** 2
            
            if i == self.L - 2:
                energy_disc += torch.sum(diff_sq_disc * self.alpha_disc_L) / 2
            else:
                energy_disc += torch.sum(diff_sq_disc) * (self.alpha_disc / 2)
        return energy_gen + energy_disc
    def compute_raw_energies(self, x):
        energy_disc_base = 0.0
        energy_gen = 0.0
        
        # 1. Énergie Discriminative (Bottom-Up)
        # Couches 1 à L-2 (On s'arrête avant la couche latente)
        for i in range(1, self.L - 1):
            pred_v = self._forward_V(x[i-1], i-1)
            energy_disc_base += torch.sum((x[i] - pred_v) ** 2) / 2
            
        # Couche L-1 (Latente : Labels + Variables Libres)
        pred_v_last = self._forward_V(x[-2], self.L - 2)
        pred_logits = pred_v_last[:, :self.num_labels]
        pred_free = pred_v_last[:, self.num_labels:]
        
        labels = x[-1][:, :self.num_labels]
        free_latents = x[-1][:, self.num_labels:]
        
        # Séparation des deux énergies
        energy_disc_labels = torch.sum((labels - pred_logits) ** 2) / 2
        energy_disc_free = torch.sum((free_latents - pred_free) ** 2) / 2
        
        # 2. Énergie Générative (Top-Down)
        for i in range(self.L - 1):
            pred_w = self._forward_W(x[i+1], i)
            energy_gen += torch.sum((x[i] - pred_w) ** 2) / 2
            
        return energy_disc_base, energy_disc_labels, energy_disc_free, energy_gen
    def bottom_up_sweep(self, x1):
        x = [x1.clone()]
        with torch.no_grad():
            for i in range(self.L - 1):
                x.append(self._forward_V(x[-1], i))
        return x

    def infer(self, x_init, clamped_indices, steps=32, lr_x=0.001928, lr_x_free=0.003162):
        """
        Phase de relaxation : trouve l'état d'équilibre des neurones (x) 
        en minimisant l'énergie locale, tout en gardant les poids figés.
        """
        # 1. Détacher les états initiaux du graphe précédent
        x = [tensor.clone().detach() for tensor in x_init]
        
        # 2. Activer le calcul de gradient uniquement pour les couches cachées libres
        for i in range(self.L - 1):
            if i not in clamped_indices:
                x[i].requires_grad = True
                
        # 3. Traitement spécial pour la couche latente (Indice L - 1)
        # Les 10 premiers neurones (Labels) sont figés.
        labels = x[-1][:, :self.num_labels].clone().detach()
        
        # Les 256 neurones suivants (Latents libres) sont activés et cherchent un équilibre.
        free_latents = x[-1][:, self.num_labels:].clone().detach()
        free_latents.requires_grad = True
        
        # La liste des tenseurs d'états que l'on va optimiser
        states_to_optimize = [xi for xi in x[:-1] if xi.requires_grad] + [free_latents]

        # --- BOUCLE DE RELAXATION ---
        for step in range(steps):
            
            # Reconstruire la couche latente complète pour les calculs d'énergie
            x_L_full = torch.cat([labels, free_latents], dim=1)
            x_current = x[:-1] + [x_L_full]
            
            # Calcul des énergies pures
            energy_disc_base, energy_disc_labels, energy_disc_free, energy_gen = self.compute_raw_energies(x_current)
            
            # LA CORRECTION : L'énergie discriminative des neurones libres est pondérée par alpha_gen !
            total_energy = (energy_disc_base * self.alpha_disc) + \
                           (energy_disc_labels * self.alpha_disc) + \
                           (energy_disc_free * self.alpha_gen) + \
                           (energy_gen * self.alpha_gen)
            
            # Calcul des gradients de l'énergie par rapport aux états x UNIQUEMENT
            grads = torch.autograd.grad(total_energy, states_to_optimize)
            
            # --- DESCENTE DE GRADIENT MANUELLE (SGD) ---
            with torch.no_grad():
                grad_idx = 0
                
                # Mise à jour des couches standard (V et W)
                for i in range(self.L - 1):
                    if x[i].requires_grad:
                        x[i] -= lr_x * grads[grad_idx]
                        grad_idx += 1
                
                # LA CORRECTION CRITIQUE : Mise à jour des 256 neurones libres.
                # On divise lr_x_free par alpha_gen (1e-7) pour compenser l'écrasement de l'énergie !
                effective_lr_free = lr_x_free / self.alpha_gen
                free_latents -= effective_lr_free * grads[grad_idx]
                
        # --- FIN DE LA BOUCLE ---
        
        # Reconstruire la liste finale détachée
        x_final = [xi.detach() for xi in x[:-1]]
        x_final.append(torch.cat([labels, free_latents], dim=1).detach())
        
        return x_final
    def infer_error_optim(self, x_init, clamped_indices, steps=5, lr_e=0.001):
        # 1. Initialisation des paramètres libres (les tenseurs d'erreur eps)
        eps_params = []
        for i in range(1, self.L):
            if i not in clamped_indices:
                # L'erreur est la variable que l'on optimise (requires_grad=True)
                e = torch.zeros_like(x_init[i], requires_grad=True)
                eps_params.append(e)
            else:
                eps_params.append(None) # Placeholder pour les couches figées
                
        # L'optimiseur SGD agit désormais directement sur les erreurs, avec le lr=0.001 de la Table 23
        optimizer_e = optim.SGD([e for e in eps_params if e is not None], lr=lr_e)
        
        for _ in range(steps):
            optimizer_e.zero_grad()
            
            x = [x_init[0]] # La couche 0 (l'image) est toujours initialisée
            energy_disc = 0.0
            
            # 2. Construction dynamique du graphe et calcul de l'énergie discriminative
            for i in range(1, self.L):
                pred_v = self._forward_V(x[-1], i-1)
                
                if i in clamped_indices:
                    # Si la couche est figée (ex: le label x_L), on déduit l'erreur
                    current_x = x_init[i]
                    current_eps = current_x - pred_v
                else:
                    # Si la couche est libre, on déduit l'état x à partir de l'erreur optimisée
                    current_eps = eps_params[i-1]
                    current_x = pred_v + current_eps
                    
                x.append(current_x)
                
                # Le calcul de l'énergie discriminative est trivial et stable 
                if i == self.L - 1:
                    energy_disc += torch.sum((current_eps ** 2) * self.alpha_disc_L) / 2
                else:
                    energy_disc += torch.sum(current_eps ** 2) * (self.alpha_disc / 2)
                    
            # 3. Calcul de l'énergie générative (inchangée selon l'article)
            energy_gen = 0.0
            for i in range(self.L - 1):
                pred_gen = self._forward_W(x[i+1], i)
                energy_gen += torch.sum((x[i] - pred_gen) ** 2) * (self.alpha_gen / 2)
                
            # Descente de gradient
            energy = energy_disc + energy_gen
            energy.backward()
            optimizer_e.step()
            
        return [tensor.detach() for tensor in x]