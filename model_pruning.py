import torch
import torch.nn as nn
import numpy as np
import os
import math
import time
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

# Pruning the model using our method
def our_pruned_model(model, prune_per):
    prev_out_channels = None  # To track the output channels of the previous layer
    non_zero_indices = None
    k = 1 - prune_per
    temp = 0
    all_losses = []
    for name, layer in model.features.named_children():
        if isinstance(layer, nn.Conv2d):
            weights = layer.weight.data.clone()
            F = weights
            if non_zero_indices is not None:
                F = weights[:, non_zero_indices, :, :]
            st1_opti = time.time()
            X_updated = optimize_frobenius(F.view(F.size(0), -1).T, int(F.size(0)* k[temp]))
            et1_opti = time.time()
            tt1_optimize = et1_opti-st1_opti
            
            print(f"Total time for solving the optimization for the above layer is :{tt1_optimize: .4f} seconds")
            print(f"{temp: .1f} th layer")
            # X_updated = iht_pruning(F.view(F.size(0), -1).T, int(F.size(0) * k[temp]))
            # X_updated = optimize_frobenius(F.view(F.size(0), -1).T, int(F.size(0)/5))
            # all_losses.append(np.array(loss_array, dtype=np.float32))
            temp = temp + 1
            non_zero_indices = (X_updated.diag() != 0).nonzero(as_tuple=True)[0]
            pruned_filters = F[non_zero_indices]
            new_out_channels, new_in_channels, k_h, k_w = pruned_filters.shape
            # Update layer weights and dimensions
            layer.weight.data = pruned_filters
            layer.out_channels = new_out_channels
            layer.in_channels = new_in_channels
            layer.bias.data = layer.bias[non_zero_indices]
            prev_out_channels = new_out_channels

        if isinstance(layer, nn.BatchNorm2d) and non_zero_indices is not None:
            # Select channels using PyTorch indexing (NO numpy)
            weight = layer.weight.data.clone()
            bias = layer.bias.data.clone()
            rm = layer.running_mean.clone()
            rv = layer.running_var.clone()

            pruned_weight = weight[non_zero_indices]
            pruned_bias = bias[non_zero_indices]
            pruned_rm = rm[non_zero_indices]
            pruned_rv = rv[non_zero_indices]

            # Update BN parameters
            layer.weight = nn.Parameter(pruned_weight)
            layer.bias = nn.Parameter(pruned_bias)

            # Update running stats
            layer.running_mean = pruned_rm
            layer.running_var = pruned_rv

            # Update layer metadata
            layer.num_features = pruned_weight.shape[0]
    for name, layer in model.classifier.named_children():
        if isinstance(layer, nn.Linear):
            # clone weight (shape: out_features × in_features)
            W = layer.weight.data.clone()

            # Rearrange to (in_features × out_features)
            W_t = W.t()

            # Prune input channels
            W_t_pruned = W_t[non_zero_indices]

            # Transpose back to (out_features × new_in_features)
            W_pruned = W_t_pruned.t()

            # Update layer
            layer.weight = nn.Parameter(W_pruned)

            # Update metadata
            layer.in_features = W_pruned.size(1)

            break




def optimize_frobenius(F, k, max_iters=3000, lr=0.001, tol=1e-6, eps=1e-8):
    m, n = F.size()
    F = F / torch.norm(F, dim=0, keepdim=True)
    G1 = torch.randn(m,n, device=F.device)
    G = G1 @ G1.t()
    diag = torch.sqrt(torch.diag(G))
    G_normalized = G / (diag[:, None] * diag[None, :])
    temp  = int(k)
    epsilon = math.sqrt((n - temp) / (temp * (n - 1)))
    I = torch.clamp(G_normalized, min=-epsilon, max=epsilon)
    I.fill_diagonal_(1.0)
    x = torch.zeros(n, requires_grad=True, device=F.device)

    # Keep top-k active indices
    with torch.no_grad():
        _, idx = torch.topk(x, k)
        mask = torch.zeros_like(x)
        mask[idx] = 1.0
        x *= mask

    optimizer = Adam([x], lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=50, verbose=False)
    loss_fro = []
    start_time1 = time.time()
    for epoch in range(max_iters):
        optimizer.zero_grad()
        X = torch.diag(x)
        loss = torch.norm(F @ X @ F.T - I, p="fro") ** 2
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            # Ensure non-negativity
            x.clamp_(min=0.0)

            # Re-select the k largest values
            if k < len(x):
                _, idx = torch.topk(x, k)
                mask = torch.zeros_like(x)
                mask[idx] = 1.0
                x *= mask

            # Guarantee exactly k nonzeros by adding small epsilon
            # so that even if any of them hit zero, they're revived
            nonzero_idx = torch.nonzero(x).flatten()
            missing = k - len(nonzero_idx)
            if missing > 0:
                # Add small positive noise to randomly chosen zero entries
                zero_idx = torch.where(x == 0)[0]
                if len(zero_idx) >= missing:
                    extra_idx = zero_idx[torch.randperm(len(zero_idx))[:missing]]
                    x[extra_idx] = eps

        loss_fro.append(loss.item())
        scheduler.step(loss)
        if epoch % 500 == 0:
            current_time1 = time.time()
            elapsed_time1 = current_time1 - start_time1
            print(f"=====> Epoch: {epoch} | Optimizer Loss: {loss_fro[-1]:.8f} | Total Time Elapsed: {elapsed_time1:.4f}s")
        if loss.item() < tol:
            break

    # print(f"Final loss: {loss.item():.6f}, number of nonzeros = {(x>0).sum().item()}")
    return torch.diag(x)

###############################

###################
#Autoencoder pruning
import torch
import torch.nn as nn
import torch.nn.functional as F

class FilterVectorAutoencoder(nn.Module):
    def __init__(self, filter_dim, latent_dim=32):
        super().__init__()
        self.filter_dim = filter_dim
        self.latent_dim = latent_dim

        # Simple MLP AE – you can adjust sizes
        self.enc = nn.Sequential(
            nn.Linear(filter_dim, 4 * latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(4 * latent_dim, latent_dim)
        )

        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 4 * latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(4 * latent_dim, filter_dim)
        )

    def encode(self, x):        # x: [N, filter_dim]
        return self.enc(x)      # -> [N, latent_dim]

    def forward(self, x):
        z = self.enc(x)
        recon = self.dec(z)
        return recon, z

def train_filter_autoencoder(F_flat, latent_dim=32, num_epochs=500, lr=1e-3, device=None):
    """
    F_flat: [num_filters, filter_dim] tensor (each row = one flattened filter)
    Train an AE to reconstruct these filters and return the trained model.
    """
    if device is None:
        device = F_flat.device

    num_filters, filter_dim = F_flat.shape

    ae = FilterVectorAutoencoder(filter_dim=filter_dim, latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(ae.parameters(), lr=lr)
    criterion = nn.MSELoss()

    F_flat = F_flat.to(device)

    ae.train()
    loss_ae = []
    start_time = time.time()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        recon, _ = ae(F_flat)
        loss = criterion(recon, F_flat)
        loss.backward()
        optimizer.step()
        loss_ae.append(loss.detach().cpu().item())
        if epoch % 50 == 0:
        	current_time = time.time()
        	elapsed_time = current_time - start_time
        
        	# Print loss and elapsed time
        	print(f"==> Epoch: {epoch} | Loss: {loss_ae[-1]:.8f} | Total Time Elapsed: {elapsed_time:.4f}s")
        
        # Optional: if you want the time taken SPECIFICALLY for the last 50 epochs, 
        # reset the start_time here:
        # start_time = time.time()
    end_time = time.time()
    total_time = end_time-start_time
    print(f"Total Training Time for training AE: {total_time:.4f} seconds")

        # You can comment this out if too verbose
        # if (epoch + 1) % 50 == 0:
        #     print(f"[Filter AE] Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}")

    return ae

def our_pruned_model_ae(model, prune_per, latent_dim=27,
                     ae_epochs=300, ae_lr=1e-3):
    prev_out_channels = None  # To track the output channels of the previous layer
    non_zero_indices = None
    k = 1 - prune_per
    temp = 0
    all_losses = []

    # infer device
    device = next(model.parameters()).device

    for name, layer in model.features.named_children():
        if isinstance(layer, nn.Conv2d):
            weights = layer.weight.data.clone()      # [out_c, in_c, k_h, k_w]
            F = weights
            if non_zero_indices is not None:
                F = weights[:, non_zero_indices, :, :]  # restrict input channels

            num_filters = F.size(0)
            F_flat = F.view(num_filters, -1)           # [num_filters, filter_dim]

            # ---- NEW PART: train AE on these filters and get latent features ----
            with torch.no_grad():
                F_flat_norm = F_flat.clone()

            # Train AE to reconstruct filters (unsupervised)
            ae = train_filter_autoencoder(
                F_flat_norm.to(device),
                latent_dim=latent_dim,
                num_epochs=ae_epochs,
                lr=ae_lr,
                device=device
            )

            ae.eval()
            with torch.no_grad():
                z = ae.encode(F_flat_norm.to(device))      # [num_filters, latent_dim]
                z = z.cpu()

            # Build feature matrix with columns = filters
            # shape: [latent_dim, num_filters]
            feature_matrix = z.T

            # Use AE features instead of raw flattened filters
            st_opti = time.time()
            X_updated = optimize_frobenius(
                feature_matrix,
                int(num_filters * k[temp])
            #X_updated = iht_pruning(feature_matrix, int(num_filters * k[temp])
            )
            et_opti = time.time()
            tt_optimize = et_opti-st_opti
            print(f"{temp: .1f} th layer")
            print(f"Total time for solving the optimization for the above layer is :{tt_optimize: .4f} seconds")

            temp = temp + 1

            # Select filters to keep
            non_zero_indices = (X_updated.diag() != 0).nonzero(as_tuple=True)[0]  # [kept_filters]

            pruned_filters = F[non_zero_indices]
            new_out_channels, new_in_channels, k_h, k_w = pruned_filters.shape

            # Update layer weights and dimensions
            layer.weight.data = pruned_filters.to(layer.weight.data.device)
            layer.out_channels = new_out_channels
            layer.in_channels = new_in_channels
            layer.bias.data = layer.bias[non_zero_indices]

            prev_out_channels = new_out_channels

        if isinstance(layer, nn.BatchNorm2d) and non_zero_indices is not None:
            weight = layer.weight.data.clone()
            bias = layer.bias.data.clone()
            rm = layer.running_mean.clone()
            rv = layer.running_var.clone()

            pruned_weight = weight[non_zero_indices]
            pruned_bias = bias[non_zero_indices]
            pruned_rm = rm[non_zero_indices]
            pruned_rv = rv[non_zero_indices]

            layer.weight = nn.Parameter(pruned_weight)
            layer.bias = nn.Parameter(pruned_bias)

            layer.running_mean = pruned_rm
            layer.running_var = pruned_rv

            layer.num_features = pruned_weight.shape[0]

    # ----- classifier pruning stays same -----
    for name, layer in model.classifier.named_children():
        if isinstance(layer, nn.Linear):
            W = layer.weight.data.clone()   # [out_features, in_features]
            W_t = W.t()                     # [in_features, out_features]

            W_t_pruned = W_t[non_zero_indices]
            W_pruned = W_t_pruned.t()

            layer.weight = nn.Parameter(W_pruned)
            layer.in_features = W_pruned.size(1)
            break

##################################################