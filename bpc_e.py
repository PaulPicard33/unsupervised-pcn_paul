from typing import Optional
import itertools
import torch
import torch.nn.functional as F
from lightning import LightningModule
import wandb
from torch.optim.lr_scheduler import LambdaLR
import math

class PCE(LightningModule):
    def __init__(
        self,
        architecture: list[torch.nn.Sequential],
        iters: int,
        e_lr: float,
        w_lr: float,
        alpha_up: float,
        alpha_down: float,
        output_loss_scale: float = 1.0,
        weight_decay: float = 0.0,
        nm_batches: int = 50000//256,  
        nm_epochs: int = 50,    
    ):
        super().__init__()

        self.save_hyperparameters()

        # Store all layers and register them properly as parameters
        self.layers_up = torch.nn.ModuleList(architecture[0])
        self.layers_down = torch.nn.ModuleList(architecture[1])

        self.errors = None  # Needs to be initialized with an input x
        self.states = None  # Needs to be set by errors

        self.alpha_up = alpha_up
        self.alpha_down = alpha_down

        self.iters = iters
        self.e_lr = e_lr
        self.w_lr = w_lr
        self.weight_decay = weight_decay

        self.nm_epochs = nm_epochs
        self.nm_batches = nm_batches

        self.output_loss_scale = output_loss_scale

        self.mask = None  # for masked input inference
        self.mask_30 = None  # for masked input inference
        self.mask_50 = None  # for masked input inference

        self.energy_scale = min([1.0, e_lr * iters]) # to avoid tiny errors from inference


    def y_pred(self, x: torch.Tensor):
        self.states = [
            x := e_i + layer_i(x) for e_i, layer_i in zip(self.errors, self.layers_up[:-1])
        ]
        return self.layers_up[-1](x)

    def y_pred_up(self, x: torch.Tensor):
        for layer_i in self.layers_up:
            x = layer_i(x)
        return x

    def y_pred_down(self, y: torch.Tensor):
        for layer_i in self.layers_down:
            y = layer_i(y)
        return y

    def class_loss(self, y_pred: torch.Tensor, y: torch.Tensor):
        return 0.5 * F.mse_loss(y_pred, y, reduction="sum") * self.output_loss_scale

    def configure_optimizers(self):
        base_lr = self.w_lr
        peak_lr = 1.1 * base_lr
        end_lr = 0.1 * base_lr

        total_steps = self.nm_batches * self.nm_epochs
        warmup_steps = int(0.1 * total_steps)

        optimizer = torch.optim.Adam(itertools.chain(self.layers_up.parameters(), self.layers_down.parameters()), lr=1.0, weight_decay=self.weight_decay)

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # Linear warmup from base_lr to peak_lr
                return base_lr + (peak_lr - base_lr) * (current_step / warmup_steps)
            else:
                # Cosine decay from peak_lr to end_lr
                progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                decayed = end_lr + (peak_lr - end_lr) * cosine_decay
                return decayed

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda) # acts as multiplier of base lr set to 1.0
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step", 
                "frequency": 1
            }
        }
        return torch.optim.Adam(itertools.chain(self.layers_up.parameters(), self.layers_down.parameters()), lr=self.w_lr, weight_decay=self.weight_decay)


    def E(self, x: torch.Tensor, y: torch.Tensor):
        """
        Calculates the energy using only the errors

        DANGER: don't use this E to train the params, or you'll be backpropping!
        """
        E_up = 0.5 * sum(torch.linalg.vector_norm(e, ord=2, dim=None) ** 2 for e in self.errors)
        pred = self.y_pred(x)   # this sets self.states

        e_down = [  
            (s_i - layer_i(s_i_plus_1)) for layer_i, s_i, s_i_plus_1 in zip(self.layers_down[::-1], [x] + self.states, self.states + [y])
        ]
        E_down = 0.5 * sum(torch.linalg.vector_norm(e, ord=2, dim=None) ** 2 for e in e_down)

        return self.alpha_up * (E_up + self.class_loss(pred, y))  + self.alpha_down * E_down

    def E_local(self, x: torch.Tensor, y: torch.Tensor):
        """
        Calculates the energy using only local interactions (no backprop!)
        Specifically, it infers the states from the errors and returns the states-based energy.

        By construction, the value is exactly equal to the energy using only errors,
        but its computational graph is different and enforces local weight updates.
        """
        E_up = 0.0
        E_down = 0.0

        # Up pass
        s_i = x
        states = []  # Store states for the down pass
        for e_i, layer_i in zip(self.errors, self.layers_up[:-1]):
            s_i_pred = layer_i(s_i)  # tracking the computational graph...
            s_i = (e_i + s_i_pred).detach()  # detach => no backprop!
            E_up += 0.5 * F.mse_loss(s_i_pred, s_i, reduction="sum") * self.alpha_up
            states.append(s_i)
        y_pred = self.layers_up[-1](s_i)
        E_up += self.class_loss(y_pred, y) * self.alpha_up

        # Down pass - no detach needed since all s_i are already detached
        for layer_i, s_i, s_i_plus_1 in zip(self.layers_down[::-1], [x] + states, states + [y]):
            s_i_pred = layer_i(s_i_plus_1)
            E_down += 0.5 * F.mse_loss(s_i_pred, s_i, reduction="sum") * self.alpha_down

        return E_up + E_down, E_up, E_down


    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None):
        if y is None:
            # Inference is easy: all errors are zero
            self.errors = [0.0] * (len(self.layers_up) - 1)

        else:  # Training is more difficult
            self.minimize_error_energy(x, y)

        # We don't need to return anything during training.
        # At inference, we can easily access the error values through self.errors

    def minimize_error_energy(self, x: torch.Tensor, y: torch.Tensor):
        """Novel PC energy minimization, using errors instead of states"""

        # Deactivate autograd on params
        for p in self.layers_up.parameters():
            p.requires_grad_(False)
        for p in self.layers_down.parameters():
            p.requires_grad_(False)


        # Initialize self.errors to the right shape using a forward pass
        self.init_zero_errors(x)

        # Minimize energy via the errors
        error_optim = torch.optim.SGD(self.errors, lr=self.e_lr)
        for _ in range(self.iters):
            error_optim.zero_grad()
            E = self.E(x, y)
            E.backward()
            error_optim.step()

        # Log final energy
        self.log("E_errors", E, prog_bar=True)

        # Re-activate autograd on params
        for p in self.layers_up.parameters():
            p.requires_grad_(True)
        for p in self.layers_down.parameters():
            p.requires_grad_(True)

    @torch.no_grad()
    def init_zero_errors(self, x: torch.Tensor):
        """Creates trainable errors via a feedforward pass"""
        self.errors = [
            torch.zeros_like(x := layer_i(x), requires_grad=True) for layer_i in self.layers_up[:-1]
        ]

    def on_fit_start(self):
        # Store batch_size for easy access
        self.batch_size = self.trainer.datamodule.batch_size

    def training_step(self, batch: dict[str, torch.Tensor], batch_idx):
        self.forward(x=batch["img"], y=batch["y"])

        # IMPORTANT: calculate the energy using the states!
        # (needed for local weight updates + good sanity check)
        E_final, E_up, E_down = self.E_local(x=batch["img"], y=batch["y"])

        self.log_dict({
            "E_up": E_up,
            "E_down": E_down,
            "E_total": E_final,
        }, prog_bar=True)

        # For weight optimization, we must average E over the batch.
        return E_final / (self.batch_size * self.energy_scale)  # = loss function for Lightning to minimize wrt params

    def infer_masked_inputs(self, batch: dict[str, torch.Tensor], mask: torch.Tensor, batch_idx=0, subset="val"):
        x, y = batch["img"], batch["y"]

        mask_bool = (mask == 1) if mask.dtype != torch.bool else mask
        if mask_bool.ndim == 4:
            # add channel dimension
            mask_bool = mask_bool.repeat(1,3,1,1)
        with torch.set_grad_enabled(True):
            # Apply the mask to the input
            x_masked = torch.nn.Parameter(torch.ones_like(x, requires_grad=True) * x * mask)
            y_init = self.y_pred_up(x=x_masked.detach())
            y_state = torch.nn.Parameter(torch.ones_like(y, requires_grad=True) * y_init)

            # Deactivate autograd on params
            for p in self.layers_up.parameters():
                p.requires_grad_(False)
            for p in self.layers_down.parameters():
                p.requires_grad_(False)

            # Initialize self.errors to the right shape using a forward pass
            self.init_zero_errors(x)

            #
            tmp_alphas = (self.alpha_up, self.alpha_down)

            self.alpha_down = 1
            self.alpha_up = 0.
            # alpha_down = torch.logspace(start=torch.log10(torch.tensor(float(self.alpha_down))), end=0, steps=4)

            # Minimize energy via the errors
            input_optim = torch.optim.Adam([x_masked] + self.errors + [y_state], lr=0.1)
            for _ in range(1000):
                input_optim.zero_grad()
                E = self.E(x_masked, y_state)
                E.backward()
                input_optim.step()
                # Reset fixed entries
                with torch.no_grad():
                    x_masked[mask_bool] = x[mask_bool]

            # Restore alphas
            self.alpha_down = tmp_alphas[1]
            self.alpha_up = tmp_alphas[0]
            self.init_zero_errors(x)
            y_init_ = self.y_pred_up(x=x_masked.detach())
            y_state = torch.nn.Parameter(torch.ones_like(y, requires_grad=True) * y_init_)
            input_optim = torch.optim.Adam([x_masked] + self.errors + [y_state], lr=0.01)
            for _ in range(2000):
                input_optim.zero_grad()
                E = self.E(x_masked, y_state)
                E.backward()
                input_optim.step()
                # Reset fixed entries
                with torch.no_grad():
                    x_masked[mask_bool] = x[mask_bool]

            # Re-activate autograd on params
            for p in self.layers_up.parameters():
                p.requires_grad_(True)
            for p in self.layers_down.parameters():
                p.requires_grad_(True)

        # Log original and reconstructed images (5 pairs)
        num_images = min(15, batch["img"].shape[0])  # Ensure we don't exceed batch size
        original_imgs = []
        masked_imgs = []
        reconstructed_imgs = []
        
        for i in range(num_images):
            original_img = batch["img"][i].cpu()
            masked_img = batch["img"][i].cpu() * mask[i].cpu()
            reconstructed_img = x_masked[i].cpu().clip(-1, 1)
            if original_img.ndim == 1:
                original_img = original_img.reshape(28,28).t()  # First image in batch
                masked_img = masked_img.reshape(28,28).t()  # First image in batch
                reconstructed_img = reconstructed_img.reshape(28,28).t()  # First reconstructed image
            original_imgs.append(original_img)
            masked_imgs.append(masked_img)
            reconstructed_imgs.append(reconstructed_img)
        
        # Concatenate all original images horizontally
        original_row = torch.cat(original_imgs, dim=-1)
        masked_row = torch.cat(masked_imgs, dim=-1)
        reconstructed_row = torch.cat(reconstructed_imgs, dim=-1)
        # Stack original and reconstructed rows vertically
        combined_image = torch.cat([original_row, masked_row, reconstructed_row], dim= 0 if original_row.ndim == 2 else -2)

        if combined_image.ndim in (3,):
            mode = "RGB" if combined_image.shape[0] == 3 else "L"
        else:
            mode = "L"
            combined_image = combined_image.unsqueeze(0)  # Add channel dimension
        self.logger.experiment.log({
            subset+"_original_vs_masked_vs_filled": wandb.Image(
                combined_image,
                mode=mode
            )
        })

        # Log the dataset-specific metrics
        node_dict = {
            "y": y_state.detach(),
            "img": x_masked.detach(),
        }

        res_reconstruction = self.trainer.datamodule.metrics_bpc(node_dict, batch, prefix=subset+"_masked_") 

        node_dict = {
            "y": y_init.detach(),
        }
        res_init = self.trainer.datamodule.metrics(node_dict, batch, prefix="masked/"+subset+"_init")

        node_dict = {
            "y": self.y_pred_up(x=x).detach(),
        }
        res = self.trainer.datamodule.metrics(node_dict, batch, prefix=subset+"_masked_")

        out_combined = {
            subset+"_combined_acc": (res[subset+"_masked_acc"] + res_reconstruction[subset+"_masked_acc"]) / 2,
            subset+"_combined_acc_top5": (res[subset+"_masked_acc_top5"] + res_reconstruction[subset+"_masked_acc_top5"]) / 2,
        }
        # Log all metrics into out_combined
        out_combined.update(res_init)
        out_combined.update(res_reconstruction)

        self.log_dict(
            out_combined, prog_bar=True
        )


    def validation_step(self, batch: dict[str, torch.Tensor], batch_idx):
        # Log the dataset-specific metrics
        node_dict = {
            "y": self.y_pred_up(x=batch["img"]),
            "img": self.y_pred_down(y=batch["y"]),
            }
        self.log_dict(
            self.trainer.datamodule.metrics_bpc(node_dict, batch, prefix="val_"), prog_bar=True
        )
        
        # Log wandb image (only for first batch to avoid too many images)
        if batch_idx == 0 and hasattr(self.logger, 'experiment'):
            # Log original and reconstructed images (5 pairs)
            num_images = min(5, batch["img"].shape[0])  # Ensure we don't exceed batch size
            original_imgs = []
            reconstructed_imgs = []
            
            for i in range(num_images):
                original_img = batch["img"][i].cpu()
                reconstructed_img = node_dict["img"][i].cpu()
                if original_img.ndim == 1:
                    original_img = original_img.reshape(28,28).t()  # First image in batch
                    reconstructed_img = reconstructed_img.reshape(28,28).t()  # First reconstructed image
                original_imgs.append(original_img)
                reconstructed_imgs.append(reconstructed_img)
            
            # Concatenate all original images horizontally
            original_row = torch.cat(original_imgs, dim=-1)
            # Concatenate all reconstructed images horizontally  
            reconstructed_row = torch.cat(reconstructed_imgs, dim=-1)
            # Stack original and reconstructed rows vertically
            combined_image = torch.cat([original_row, reconstructed_row], dim= 0 if original_row.ndim == 2 else -2)

            if combined_image.ndim in (3,):
                mode = "RGB" if combined_image.shape[0] == 3 else "L"
            else:
                mode = "L"
                combined_image = combined_image.unsqueeze(0)  # Add channel dimension
            self.logger.experiment.log({
                "val_original_vs_reconstructed": wandb.Image(
                    combined_image,
                    mode=mode
                )
            })


    def test_step(self, batch: dict[str, torch.Tensor], batch_idx):
        # Log the dataset-specific metrics
        node_dict = {
            "y": self.y_pred_up(x=batch["img"]),
            "img": self.y_pred_down(y=batch["y"]),
            }
        self.log_dict(
            self.trainer.datamodule.metrics_bpc(node_dict, batch, prefix="test_"), prog_bar=True
        )
        
        if hasattr(self.logger, 'experiment'):
            if self.mask_30 is None:
                mask = make_mask(0.3, batch["img"].shape[0], batch["img"].shape[1:], patch_size=1).to(batch["img"].device)
                self.mask_30 = mask
            if self.mask_50 is None:
                mask = make_mask(0.5, batch["img"].shape[0], batch["img"].shape[1:], patch_size=1).to(batch["img"].device)
                self.mask_50 = mask
            self.infer_masked_inputs(batch, self.mask_30, batch_idx=batch_idx, subset="test_30")
            self.infer_masked_inputs(batch, self.mask_50, batch_idx=batch_idx, subset="test_50")


    ### State-based methods ###
    def get_states_from_errors(self, x: torch.Tensor):
        """Returns the states corresponding to the errors, including y_pred"""
        return [(x := e_i + layer_i(x)) for e_i, layer_i in zip(self.errors + [0.0], self.layers_up)]

    def E_states_only(self, x: torch.Tensor, y: torch.Tensor, states: list[torch.Tensor]):
        """
        Calculates the energy using only the states, which need to be given as inputs.
        No errors are used here.
        """

        def half_mse_loss(y_pred, y):
            return 0.5 * F.mse_loss(y_pred, y, reduction="sum")

        losses = [half_mse_loss] * len(states) + [self.class_loss]
        states = [x] + states + [y]

        E_up = self.alpha_up * sum(
            loss(layer(s_i), s_ip1)
            for s_i, s_ip1, layer, loss in zip(states[:-1], states[1:], self.layers_up, losses)
        )

        E_down = self.alpha_down * sum(
            half_mse_loss(s_i, layer(s_ip1))
            for s_i, s_ip1, layer in zip(states[:-1], states[1:], self.layers_down[::-1])
        )
        return E_up + E_down

    def minimize_state_energy(self, x: torch.Tensor, y: torch.Tensor, iters: int, s_lr: float):
        """Classical PC energy minimization using states"""

        # Deactivate autograd on params
        for p in self.layers_up.parameters():
            p.requires_grad_(False)
        for p in self.layers_down.parameters():
            p.requires_grad_(False)

        # Initialize states using a feedforward pass
        def ff_init(s):
            return [(s := layer(s).detach().requires_grad_(True)) for layer in self.layers_up[:-1]]

        states = ff_init(x)

        # Minimize energy via the states
        state_optim = torch.optim.SGD(states, lr=s_lr)
        for _ in range(iters):
            state_optim.zero_grad()
            E = self.E_states_only(x, y, states)
            E.backward()
            state_optim.step()

        # Re-activate autograd on params
        for p in self.layers_up.parameters():
            p.requires_grad_(True)
        for p in self.layers_down.parameters():
            p.requires_grad_(True)

        # No need to store in self.states, just return for later use in callbacks
        return states


# Define state optim version of PCE
class PC_States(PCE):
    def minimize_error_energy(self, x, y):
        # Recycle iters and e_lr for state optimization, and store final states
        self.states = super().minimize_state_energy(x, y, self.iters, self.e_lr)

    def E_local(self, x, y):
        return super().E_states_only(x, y, self.states)

    # No need to redefine forward or y_pred:
    # For prediction, they set all errors to zero and simply to the correct prediction.
    # Therefore, we only need to adapt the training procedure.


class PCESkipConnection(PCE):
    def __init__(
        self,
        architecture: list[torch.nn.Sequential],
        iters: int,
        e_lr: float,
        w_lr: float,
        alpha_up: float,
        alpha_down: float,
        output_loss_scale: float = 1.0,
        weight_decay: float = 0.0,
        nm_batches: int = 50000//256,
        nm_epochs: int = 50,
    ):
        super().__init__(architecture, iters, e_lr, w_lr, alpha_up, alpha_down, output_loss_scale, weight_decay, nm_batches, nm_epochs)


    def y_pred(self, x: torch.Tensor):
        s_i = (x, 0.0)  # activity, identity for skip connection
        states = []
        for e_i, layer_i in zip(self.errors + [0.0], self.layers_up):
            s_i = layer_i(s_i)  # layers take care of writing s_i[1] and adding it to s_i[0]
            s_i = (s_i[0] + e_i, s_i[1]) 
            states.append(s_i[0])
        self.states = states[::2]  # all but last and skip states whithin skip connections
        return s_i[0]
    

    @torch.no_grad()
    def init_zero_errors(self, x, y):
        super().init_zero_errors(x)

        # init down errors - only exist within middle of residual block
        self.errors_down = [
            torch.zeros_like(y := layer_i(y), requires_grad=True) for layer_i in self.layers_down[:-1]
        ]
        self.errors_down = self.errors_down[1::2]


    def E(self, x: torch.Tensor, y: torch.Tensor):
        """
        Calculates the energy using only the errors

        DANGER: don't use this E to train the params, or you'll be backpropping!
        """
        E_up = 0.5 * sum(torch.linalg.vector_norm(e, ord=2, dim=None) ** 2 for e in self.errors)
        pred = self.y_pred(x)   # this sets self.states

        E_down = 0.5 * sum(torch.linalg.vector_norm(e, ord=2, dim=None) ** 2 for e in self.errors_down)

        e_down = []
        e_down += [self.states[-1] - self.layers_down[0](y)]  # output layer error
        for i in range(0, (len(self.layers_down)-2)//2):
            s_i_plus_1 = self.states[-1 - i]
            s_i = self.states[-2 - i]

            layer_i_1 = self.layers_down[2*i + 1]
            layer_i_2 = self.layers_down[2*i + 2]

            e_d = self.errors_down[i]

            hidden = layer_i_1((s_i_plus_1, 0))
            hidden = (hidden[0] + e_d, hidden[1])
            hidden = layer_i_2(hidden)

            e = s_i - hidden[0]
            e_down.append(e)
        e_down += [x - self.layers_down[-1](self.states[0])] 
        E_down += 0.5 * sum(torch.linalg.vector_norm(e, ord=2, dim=None) ** 2 for e in e_down)

        return self.alpha_up * (E_up + self.class_loss(pred, y))  + self.alpha_down * E_down

    def minimize_error_energy(self, x: torch.Tensor, y: torch.Tensor):
        """Novel PC energy minimization, using errors instead of states"""

        # Deactivate autograd on params
        for p in self.layers_up.parameters():
            p.requires_grad_(False)
        for p in self.layers_down.parameters():
            p.requires_grad_(False)


        # Initialize self.errors to the right shape using a forward pass
        self.init_zero_errors(x, y)

        # Minimize energy via the errors
        error_optim = torch.optim.SGD(self.errors + self.errors_down, lr=self.e_lr)
        for _ in range(self.iters):
            error_optim.zero_grad()
            E = self.E(x, y)
            E.backward()
            error_optim.step()

        # Log final energy
        self.log("E_errors", E, prog_bar=True)

        # Re-activate autograd on params
        for p in self.layers_up.parameters():
            p.requires_grad_(True)
        for p in self.layers_down.parameters():
            p.requires_grad_(True)


    def E_local(self, x: torch.Tensor, y: torch.Tensor):
        """
        Calculates the energy using only local interactions (no backprop!)
        Specifically, it infers the states from the errors and returns the states-based energy.

        By construction, the value is exactly equal to the energy using only errors,
        but its computational graph is different and enforces local weight updates.
        """
        E_up = 0.0
        E_down = 0.0

        # Up pass
        s_i = (x, 0.0)
        states = []  # Store states for the down pass
        for e_i, layer_i in zip(self.errors, self.layers_up[:-1]):
            s_i_pred = layer_i(s_i)  # tracking the computational graph...
            s_i = (e_i + s_i_pred[0]).detach()  # detach => no backprop!
            E_up += 0.5 * F.mse_loss(s_i_pred[0], s_i, reduction="sum") * self.alpha_up
            s_i = (s_i, s_i_pred[1])
            states.append(s_i[0])
        y_pred = self.layers_up[-1](s_i)[0]
        E_up += self.class_loss(y_pred, y) * self.alpha_up

        states = states[::2]

        E_down += 0.5 * F.mse_loss(states[-1], self.layers_down[0](y), reduction="sum") * self.alpha_down
        for i in range(0, (len(self.layers_down)-2)//2):
            s_i_plus_1 = states[-1 - i]
            s_i_final = states[-2 - i]
            layer_i_1 = self.layers_down[2*i + 1]
            layer_i_2 = self.layers_down[2*i + 2]
            e_d = self.errors_down[i]

            s_i_pred = layer_i_1((s_i_plus_1, 0))
            s_i = (s_i_pred[0] + e_d).detach()
            E_down += 0.5 * F.mse_loss(s_i_pred[0], s_i , reduction="sum") * self.alpha_down

            s_i_pred = layer_i_2((s_i, s_i_pred[1]))
            E_down += 0.5 * F.mse_loss(s_i_pred[0], s_i_final , reduction="sum") * self.alpha_down
        E_down += 0.5 * F.mse_loss(x, self.layers_down[-1](states[0]), reduction="sum") * self.alpha_down

        return E_up + E_down, E_up, E_down



import numpy as np
def make_mask(p, batch_size, shape, patch_size):
    # infer but allow only the wrongly initialised neurons to be updated
    shape_orig = shape
    if len(shape) == 1:
        shape = (int(np.sqrt(shape[0])), int(np.sqrt(shape[0])))
    elif len(shape) == 3:
        shape = shape[1:]  # remove channel dimension
        shape_orig = (1, *shape_orig[1:])  # remove channel dimension and add 1 for multiplying back later
    else:
        raise ValueError("Shape must be 1D or 3D (with channel dimension)")

    # can be done by updating the call function of the input vode
    mask_fixed = torch.ones((batch_size, *shape))  # one if fixed zero if not

    # sample patches, patches are indexed from top left to bottom right
    n_pathches = np.prod(shape) // patch_size**2
    n_patches_per_row = shape[1] // patch_size

    # p gives the proportion of input neurons that should be left uninitialised
    for i in range(batch_size):
        idxs = np.random.choice(n_pathches, int(p * n_pathches), replace=False)

        # fill in the selected patches with zeros
        for idx in idxs:
            row = idx // n_patches_per_row
            col = idx % n_patches_per_row
            mask_fixed[
                i,
                row * patch_size : (row + 1) * patch_size,
                col * patch_size : (col + 1) * patch_size,
            ] = 0        

    return mask_fixed.reshape(batch_size, *shape_orig)  # flatten the image to a vector