import torch
import torch.nn as nn
import numpy as np
import os
import math
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
            X_updated = optimize_frobenius(F.view(F.size(0), -1).T, int(F.size(0)* k[temp]))
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




def optimize_frobenius(F, k, max_iters=1000, lr=0.001, tol=1e-6, eps=1e-8):
    m, n = F.size()
    F = F / torch.norm(F, dim=0, keepdim=True)
    #Uncomment for ETF code otherwise comment for Identity
    G1 = torch.randn(m,n, device=F.device)
    G = G1 @ G1.t()
    diag = torch.sqrt(torch.diag(G))
    G_normalized = G / (diag[:, None] * diag[None, :])
    temp  = int(k)
    epsilon = math.sqrt((n - temp) / (temp * (n - 1)))
    I = torch.clamp(G_normalized, min=-epsilon, max=epsilon)
    I.fill_diagonal_(1.0)


    #I = torch.eye(m, device=F.device)

    # ---- Initialization: random positive values ----
    # x = torch.rand(n, requires_grad=True, device=F.device)
    x = torch.zeros(n, requires_grad=True, device=F.device)

    # Keep top-k active indices
    with torch.no_grad():
        _, idx = torch.topk(x, k)
        mask = torch.zeros_like(x)
        mask[idx] = 1.0
        x *= mask

    optimizer = Adam([x], lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=50)
    loss_fro = []

    for _ in range(max_iters):
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

        if loss.item() < tol:
            break

    # print(f"Final loss: {loss.item():.6f}, number of nonzeros = {(x>0).sum().item()}")
    return torch.diag(x)

def iht_pruning(F, k, num_iters=1000, lr=1e-3, tol=1e-06):
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m, n = F.size()
    # F = F.to(device)
    F = F / torch.norm(F, dim=0, keepdim=True)
    I = torch.eye(m, device=device)
    loss_iht = []

    # Initialize w (nonnegative)
    w = torch.zeros(n, device=device, requires_grad=False)

    # Precompute columns for gradient evaluation
    cols = [F[:, i] for i in range(n)]

    for t in range(num_iters):
        # Compute A = F diag(w) F^T
        A = F @ torch.diag(w) @ F.T
        R = A - I  # residual

        # Gradient: g_i = 2 f_i^T R f_i
        g = torch.tensor([2 * cols[i].dot(R @ cols[i]) for i in range(n)],
                         device=device)
        loss_iht.append(torch.norm(R, p="fro") ** 2)

        # Gradient step
        w_half = w - lr * g

        # Project to nonnegative
        w_half = torch.clamp(w_half, min=0.0)

        # Keep top-k entries
        if k < n:
            topk_idx = torch.topk(w_half, k).indices
            mask = torch.zeros_like(w_half)
            mask[topk_idx] = 1.0
            w = w_half * mask
        else:
            w = w_half
        if loss_iht[-1] < tol:
            break

    # Final D = diag(sqrt(w)) (since D^2 = diag(w))
    D = torch.diag(w)
    # print(loss_iht)
    return D.to(device)

###############################
#Iterative pruning###

def prune_one_layer(model, prune_ratio, layer_idx):
    """
    Prune only the convolutional layer at `layer_idx` (index among conv layers in model.features).
    prune_ratio: fraction to prune (e.g., 0.25 means keep 75%).
    """
    k_ratio = 1.0 - prune_ratio  # fraction to keep
    conv_indices = []
    for i, (name, layer) in enumerate(model.features.named_children()):
        if isinstance(layer, nn.Conv2d):
            conv_indices.append(i)

    if layer_idx < 0 or layer_idx >= len(conv_indices):
        raise IndexError("layer_idx out of range")

    # Map layer order index to named_children index
    target_child_idx = conv_indices[layer_idx]

    # We'll keep track of the non_zero_indices for the pruned layer (output channels we keep)
    non_zero_indices = None

    # First pass: find and prune the target conv layer
    temp = 0
    for idx, (name, layer) in enumerate(model.features.named_children()):
        if not isinstance(layer, nn.Conv2d):
            continue

        # if temp == layer_idx:
        #     # This is the conv we want to prune
        #     weights = layer.weight.data.clone()  # shape: (out_ch, in_ch, kh, kw)
        #     device = weights.device
        #
        #     # If previous layer was pruned, caller should have updated next conv's input channels.
        #     # Here we simply compute pruning on current weights as-is.
        #
        #     out_ch, in_ch, kh, kw = weights.shape
        #     keep_k = int(max(1, round(out_ch * k_ratio)))  # number of filters to keep
        #
        #     F = weights  # shape (out_ch, in_ch, kh, kw)
        #
        #     # Flatten filters to (out_ch, in_ch*kh*kw) for your optimize_frobenius input format
        #     # You used F.view(F.size(0), -1).T previously; keep same behavior
        #     X_updated = optimize_frobenius(F.view(F.size(0), -1).T.to(device), keep_k)
        if temp == layer_idx:
            # This is the conv we want to prune
            weights = layer.weight.data.clone()  # shape: (out_ch, in_ch, kh, kw)
            device = weights.device

            out_ch, in_ch, kh, kw = weights.shape
            keep_k = int(max(1, round(out_ch * k_ratio)))  # number of filters to keep

            F = weights  # shape (out_ch, in_ch, kh, kw)

            # ---------- AE-BASED PART (INSERTED) ----------
            # Flatten filters: [num_filters, filter_dim]
            F_flat = F.view(out_ch, -1)  # [out_ch, in_ch*kh*kw]

            with torch.no_grad():
                F_flat_norm = F_flat.clone().to(device)

            # latent_dim same idea as in your our_pruned_model_ae
            # latent_dim = max(1, int(F_flat_norm.shape[1] / 4))
            latent_dim = 27

            # Train AE to reconstruct filters (unsupervised)
            ae = train_filter_autoencoder(F_flat_norm, latent_dim=latent_dim, device=device)

            ae.eval()
            with torch.no_grad():
                z = ae.encode(F_flat_norm)  # [out_ch, latent_dim]
                z = z.cpu()

            # Build feature matrix with columns = filters: [latent_dim, out_ch]
            feature_matrix = z.T

            # Use AE features instead of raw flattened filters
            X_updated = optimize_frobenius(feature_matrix, keep_k)
            # ---------- END AE PART ----------

            non_zero_indices = (X_updated.diag() != 0).nonzero(as_tuple=True)[0].to(device)
            # Ensure sorted
            non_zero_indices, _ = torch.sort(non_zero_indices)

            # Select the pruned filters (output channel selection)
            pruned_filters = F[non_zero_indices, :, :, :].contiguous()
            new_out_channels, new_in_channels, k_h, k_w = pruned_filters.shape

            # Update the conv layer parameters
            layer.out_channels = new_out_channels
            layer.in_channels = new_in_channels  # usually same
            layer.weight = nn.Parameter(pruned_filters)
            if layer.bias is not None:
                layer.bias = nn.Parameter(layer.bias.data[non_zero_indices].clone())

            # Update BatchNorm immediately following this conv (if any)
            # Typically pattern: Conv -> BN -> ReLU; find BN in subsequent child indices
            # We search the next few named_children entries
            # Note: need to get names from named_children list again to match indices
            named = list(model.features.named_children())
            # find index of this conv in named list
            named_idx = None
            for j, (n, l) in enumerate(named):
                if n == name and l is layer:
                    named_idx = j
                    break
            # check next entries for BatchNorm2d
            if named_idx is not None:
                for j in range(named_idx+1, min(named_idx+4, len(named))):
                    n2, l2 = named[j]
                    if isinstance(l2, nn.BatchNorm2d):
                        # prune BN params
                        l2.weight = nn.Parameter(l2.weight.data[non_zero_indices].clone())
                        l2.bias = nn.Parameter(l2.bias.data[non_zero_indices].clone())
                        l2.running_mean = l2.running_mean[non_zero_indices].clone()
                        l2.running_var = l2.running_var[non_zero_indices].clone()
                        l2.num_features = non_zero_indices.numel()
                        break
            break
        temp += 1

    if non_zero_indices is None:
        # Should not happen
        print("Warning: non_zero_indices is None after pruning target conv")
        return non_zero_indices

    # Second pass: find the *next* Conv2d after this pruned conv (L+1) and prune its input channels
    # We must reduce the in_channels dimension by selecting those indices.
    # Iterate through model.features named_children in order, find first Conv2d after target_child_idx
    named = list(model.features.named_children())
    next_conv_found = False
    for j in range(target_child_idx+1, len(named)):
        name_j, layer_j = named[j]
        if isinstance(layer_j, nn.Conv2d):
            # layer_j.weight shape: (out_ch_j, in_ch_j, kh, kw)
            wj = layer_j.weight.data.clone()
            # if this conv's in_channels matches original out_channels, we can prune corresponding input channels
            # select input channels by non_zero_indices
            # But make sure non_zero_indices length <= current in_channels
            _, in_ch_j, kh_j, kw_j = wj.shape
            if non_zero_indices.numel() <= in_ch_j:
                # Select input channels -> keep only those input channel slices corresponding to non_zero_indices
                wj_pruned = wj[:, non_zero_indices, :, :].contiguous()
                layer_j.in_channels = wj_pruned.shape[1]
                layer_j.weight = nn.Parameter(wj_pruned)
                # If bias exists, keep same bias (bias is per output channel)
                next_conv_found = True
            else:
                # If mismatch (e.g., due to grouped convs or earlier structure), do not prune input channels automatically.
                # Warn and skip input-channel pruning.
                print(f"Warning: cannot prune input channels of {name_j}; mismatch in channel counts.")
            break

    if not next_conv_found:
        # No next conv in features; often the next structural module is the classifier.
        # You already have code to prune classifier linear using non_zero_indices at the end; we optionally do that here.
        # Find first Linear in model.classifier and prune its input rows corresponding to pruned features.
        for name_c, layer_c in model.classifier.named_children():
            if isinstance(layer_c, nn.Linear):
                W = layer_c.weight.data.clone()  # shape (out_f, in_f)
                W_t = W.t()  # (in_f, out_f)
                # If the input dimension equals the original out_channels * spatial_size, careful: CIFAR VGG flattens features.
                # Here we only handle case where linear input corresponds to channel-wise concatenation (e.g., avg-pooled features).
                # We'll prune columns that correspond directly to channels if possible (simple case).
                # If not safe, skip and leave classifier unchanged.
                if non_zero_indices.numel() <= W_t.shape[0]:
                    W_t_pruned = W_t[non_zero_indices]
                    W_pruned = W_t_pruned.t().contiguous()
                    layer_c.in_features = W_pruned.size(1)
                    layer_c.weight = nn.Parameter(W_pruned)
                    # bias remains same
                else:
                    print("Warning: cannot prune classifier linear input reliably; skipping.")
                break

    return non_zero_indices

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
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        recon, _ = ae(F_flat)
        loss = criterion(recon, F_flat)
        loss.backward()
        optimizer.step()
        loss_ae.append(loss.detach().cpu().item())

        # You can comment this out if too verbose
        # if (epoch + 1) % 50 == 0:
        #     print(f"[Filter AE] Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}")

    return ae

def our_pruned_model_ae(model, prune_per, latent_dim=27,
                     ae_epochs=1000, ae_lr=1e-3):
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
            X_updated = optimize_frobenius(
                feature_matrix,
                int(num_filters * k[temp])
            )

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
##################################################
##################################################
#CBMIR process for filter pruning


import torch
import torch.nn.functional as F

# ---------- K-means in PyTorch ----------
def kmeans_torch(X, n_clusters, n_iters=20, device=None):
    """
    X: [N, D]  (N samples, D-dim features)
    Returns:
        labels: [N] long tensor
        centers: [n_clusters, D]
    """
    if device is None:
        device = X.device
    X = X.to(device)

    N, D = X.shape
    # random initialization from data
    idx = torch.randperm(N, device=device)[:n_clusters]
    centers = X[idx].clone()

    for _ in range(n_iters):
        # Assign
        distances = torch.cdist(X, centers)  # [N, K]
        labels = distances.argmin(dim=1)

        # Update
        for c in range(n_clusters):
            mask = labels == c
            if mask.any():
                centers[c] = X[mask].mean(dim=0)

    return labels, centers


# ---------- OMP for sparse coding (single sample) ----------
def omp_single(D, y, sparsity):
    """
    D: [d, K] dictionary
    y: [d, 1] signal
    sparsity: int (max nonzeros)

    Returns:
        x: [K, 1] sparse code
    """
    d, K = D.shape
    y = y.to(D.device)

    residual = y.clone()
    idxs = []
    x = torch.zeros(K, 1, device=D.device)

    for t in range(sparsity):
        # correlation with residual
        corr = (D.t() @ residual).abs().squeeze(1)  # [K]
        k = int(corr.argmax().item())
        if k in idxs:
            break

        idxs.append(k)
        D_sub = D[:, idxs]  # [d, t+1]

        # least squares: min ||y - D_sub * a||
        # solution: a = argmin ||y - D_sub a||
        # using torch.linalg.lstsq
        sol = torch.linalg.lstsq(D_sub, y).solution  # [t+1, 1]
        residual = y - D_sub @ sol

    if len(idxs) > 0:
        x[idxs, 0:1] = sol

    return x


# ---------- K-SVD dictionary learning ----------
def ksvd(Y, n_atoms, sparsity, n_iter=5):
    """
    Y: [d, N]  data matrix (columns are signals)
    n_atoms: number of dictionary atoms
    sparsity: max nonzeros per code
    n_iter: number of K-SVD iterations

    Returns:
        D: [d, n_atoms] learned dictionary
        X: [n_atoms, N] sparse codes
    """
    device = Y.device
    d, N = Y.shape

    # Initialize D by selecting random columns of Y
    rand_idx = torch.randperm(N, device=device)[:n_atoms]
    D = Y[:, rand_idx].clone()  # [d, n_atoms]
    D = F.normalize(D, dim=0)

    X = torch.zeros(n_atoms, N, device=device)

    for it in range(n_iter):
        # --- Sparse coding step (OMP per column) ---
        for i in range(N):
            y_i = Y[:, i:i+1]  # [d,1]
            x_i = omp_single(D, y_i, sparsity=min(sparsity, n_atoms))
            X[:, i:i+1] = x_i

        # --- Dictionary update step ---
        for k in range(n_atoms):
            omega = (X[k, :] != 0).nonzero(as_tuple=True)[0]
            if omega.numel() == 0:
                continue

            # E_k = Y_omega - sum_{j!=k} D_j x_j
            D_except = D.clone()
            D_except[:, k] = 0
            E_k = Y[:, omega] - D_except @ X[:, omega]  # [d, |omega|]

            # SVD of E_k
            U, S, Vh = torch.linalg.svd(E_k, full_matrices=False)
            # Update atom k and its coefficients
            D[:, k] = U[:, 0]
            X[k, omega] = S[0] * Vh[0, :]

        # Normalize dictionary atoms
        D = F.normalize(D, dim=0)

    return D, X


# ---------- Cluster refinement via K-SVD ----------
# def refine_clusters_ksvd(Y, labels, n_clusters,
#                          sparsity=2, n_ksvd_iter=3, n_global_iter=1):
#     """
#     Y: [d, N] feature matrix (columns are data points)
#     labels: [N] initial cluster labels (from K-means)
#     n_clusters: number of clusters
#     sparsity: sparsity level in OMP
#     n_ksvd_iter: iterations per K-SVD call
#     n_global_iter: number of global cluster refinement iterations
#
#     Returns:
#         labels_refined: [N] refined labels
#     """
#     device = Y.device
#     d, N = Y.shape
#     labels = labels.clone().to(device)
#
#     for _ in range(n_global_iter):
#         dicts = [None for _ in range(n_clusters)]
#
#         # 1) Learn dictionary per cluster using K-SVD
#         for c in range(n_clusters):
#             idx_c = (labels == c).nonzero(as_tuple=True)[0]
#             if idx_c.numel() == 0:
#                 continue
#
#             Y_c = Y[:, idx_c]  # [d, Nc]
#             # choose atoms <= min(d, Nc)
#             n_atoms_c = int(min(Y_c.shape[1], Y_c.shape[0]))
#             if n_atoms_c == 0:
#                 continue
#
#             D_c, _ = ksvd(
#                 Y_c,
#                 n_atoms=n_atoms_c,
#                 sparsity=min(sparsity, n_atoms_c),
#                 n_iter=n_ksvd_iter
#             )
#             dicts[c] = D_c
#
#         # 2) Reassign each sample to the best dictionary (smallest recon error)
#         new_labels = labels.clone()
#         for i in range(N):
#             y_i = Y[:, i:i+1]  # [d,1]
#
#             best_c = None
#             best_err = None
#
#             for c in range(n_clusters):
#                 D_c = dicts[c]
#                 if D_c is None:
#                     continue
#
#                 x_i = omp_single(D_c, y_i, sparsity=min(sparsity, D_c.shape[1]))
#                 recon = D_c @ x_i
#                 err = torch.norm(y_i - recon) ** 2
#
#                 if (best_err is None) or (err < best_err):
#                     best_err = err
#                     best_c = c
#
#             if best_c is not None:
#                 new_labels[i] = best_c
#
#         labels = new_labels
#
#     return labels

def refine_clusters_ksvd(Y, labels, n_clusters,
                         sparsity=5, n_ksvd_iter=3, n_global_iter=1):
    """
    Y: [d, N] feature matrix (columns are data points)
    labels: [N] initial cluster labels (from K-means)
    n_clusters: number of clusters
    sparsity: sparsity level in OMP
    n_ksvd_iter: iterations per K-SVD call
    n_global_iter: number of global refinement iterations

    Returns:
        labels_refined: [N] refined labels
    """
    device = Y.device
    d, N = Y.shape
    labels = labels.clone().to(device)

    for _ in range(n_global_iter):
        dicts = [None for _ in range(n_clusters)]

        # 1) Learn dictionary per cluster via K-SVD (same as before)
        for c in range(n_clusters):
            idx_c = (labels == c).nonzero(as_tuple=True)[0]
            if idx_c.numel() == 0:
                continue

            Y_c = Y[:, idx_c]  # [d, Nc]
            n_atoms_c = int(min(Y_c.shape[1], Y_c.shape[0]))
            if n_atoms_c == 0:
                continue

            D_c, _ = ksvd(
                Y_c,
                n_atoms=n_atoms_c,
                sparsity=min(sparsity, n_atoms_c),
                n_iter=n_ksvd_iter
            )
            dicts[c] = D_c

        # 2) Build global dictionary D_all and mapping from clusters to atom indices
        D_cols = []
        cluster_atom_indices = {}
        start = 0
        for c in range(n_clusters):
            D_c = dicts[c]
            if D_c is None:
                continue
            n_atoms_c = D_c.shape[1]
            D_cols.append(D_c)
            idxs = torch.arange(start, start + n_atoms_c, device=device)
            cluster_atom_indices[c] = idxs
            start += n_atoms_c

        if len(D_cols) == 0:
            # No valid dictionaries, return labels unchanged
            return labels

        D_all = torch.cat(D_cols, dim=1)  # [d, K_total]

        # 3) Reassign each sample using OMP on the whole dictionary
        new_labels = labels.clone()
        for i in range(N):
            y_i = Y[:, i:i+1]  # [d,1]

            # OMP on full dictionary
            x_i = omp_single(D_all, y_i, sparsity=min(sparsity, D_all.shape[1]))  # [K_total, 1]

            best_c = None
            best_err = None

            # Compute error for each cluster using only its atoms (corresponding positions in x_i)
            for c in range(n_clusters):
                if c not in cluster_atom_indices:
                    continue
                idxs = cluster_atom_indices[c]  # global atom indices for cluster c
                if idxs.numel() == 0:
                    continue

                D_c_all = D_all[:, idxs]    # [d, Nc_atoms]
                x_c = x_i[idxs, :]          # [Nc_atoms, 1]

                recon_c = D_c_all @ x_c
                err_c = torch.norm(y_i - recon_c) ** 2

                if (best_err is None) or (err_c < best_err):
                    best_err = err_c
                    best_c = c

            if best_c is not None:
                new_labels[i] = best_c

        print(torch.equal(labels, new_labels))
        labels = new_labels

    return labels



# ---------- Final selection: K-means + K-SVD + ℓ1 representative ----------
def kmeans_ksvd_prune(feature_matrix, k_keep,
                      kmeans_iters=20,
                      sparsity=12,
                      n_ksvd_iter=3,
                      n_global_iter=1,
                      device=None):
    """
    feature_matrix: [d, n_filters], columns = filter features (from AE)
    k_keep: desired number of filters to keep
    Returns:
        keep_indices: LongTensor of indices of filters to keep
    """
    d, n = feature_matrix.shape
    if device is None:
        device = feature_matrix.device

    k_keep = max(1, min(k_keep, n))

    # Use rows as samples for K-means: [n_filters, d]
    X = feature_matrix.t().to(device)  # [n, d]

    # 1) K-means initialization
    init_labels, _ = kmeans_torch(X, n_clusters=k_keep, n_iters=kmeans_iters, device=device)

    # 2) Refine clusters with K-SVD
    Y = feature_matrix.to(device)  # [d, n]
    labels_refined = refine_clusters_ksvd(
        Y, init_labels, n_clusters=k_keep,
        sparsity=sparsity,
        n_ksvd_iter=n_ksvd_iter,
        n_global_iter=n_global_iter
    )

    # 3) Select one representative per cluster using ℓ1 norm
    keep_indices = []
    X_row = feature_matrix.t().to(device)  # [n, d]

    for c in range(k_keep):
        cluster_idx = (labels_refined == c).nonzero(as_tuple=True)[0]
        if cluster_idx.numel() == 0:
            continue

        feats = X_row[cluster_idx]                  # [Nc, d]
        l1_norms = feats.abs().sum(dim=1)           # [Nc]
        best_local = cluster_idx[l1_norms.argmax()] # index in [0, n)
        keep_indices.append(best_local)

    if len(keep_indices) == 0:
        # fallback: keep top-k by global ℓ1 norm
        l1_global = X_row.abs().sum(dim=1)
        _, top_idx = torch.topk(l1_global, k_keep)
        keep_indices = top_idx.tolist()

    keep_indices = torch.tensor(keep_indices, dtype=torch.long, device=device)
    return keep_indices

def our_pruned_model_ae_cbmi(model, prune_per, latent_dim=27,
                        ae_epochs=1000, ae_lr=1e-3):
    prev_out_channels = None
    non_zero_indices = None
    k = 1 - prune_per
    temp = 0
    all_losses = []

    device = next(model.parameters()).device

    for name, layer in model.features.named_children():
        if isinstance(layer, nn.Conv2d):
            weights = layer.weight.data.clone()      # [out_c, in_c, k_h, k_w]
            F = weights
            if non_zero_indices is not None:
                F = weights[:, non_zero_indices, :, :]  # restrict input channels

            num_filters = F.size(0)
            F_flat = F.view(num_filters, -1)           # [num_filters, filter_dim]

            with torch.no_grad():
                F_flat_norm = F_flat.clone()

            # ---- Train AE on filters and get latent features ----
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

            # Feature matrix: [latent_dim, num_filters]
            feature_matrix = z.T.to(device)

            keep_ratio = k[temp].item() if torch.is_tensor(k[temp]) else float(k[temp])
            keep_count = int(num_filters * keep_ratio)
            keep_count = max(1, min(keep_count, num_filters))

            # --- K-means + K-SVD + ℓ1-based cluster pruning ---
            keep_indices = kmeans_ksvd_prune(
                feature_matrix,
                k_keep=keep_count,
                kmeans_iters=20,
                sparsity=5,
                n_ksvd_iter=3,
                n_global_iter=1,
                device=device
            )

            temp += 1
            non_zero_indices = keep_indices.cpu()  # indices of filters to keep

            # Apply pruning to conv layer
            pruned_filters = F[non_zero_indices]
            new_out_channels, new_in_channels, k_h, k_w = pruned_filters.shape

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


##########################################################
###########################
####################
#PRF
#Resnet Pruning
import numpy as np

# def prune_resnet_filters(model, prune_value, latent_dim=27, ae_epochs=500, ae_lr=1e-3):
#
#     device = next(model.parameters()).device
#     conv_layer = 0
#     first_ele = None      # Stores selected output channels
#     in_channels = None    # Needed to update conv2 input channels
#     out_channels = None   # Update module.out_channels
#
#     for layer_name, layer_module in model.named_modules():
#
#         # -----------------------------
#         # PRUNE CONVOLUTION LAYERS
#         # -----------------------------
#         if isinstance(layer_module, nn.Conv2d) and layer_name != 'conv1':
#
#             # Convert weight to tensor (NOT numpy)
#             W = layer_module.weight.data.clone().to(device)   # shape: [out_c, in_c, k, k]
#
#             # ------------------------------------------------------
#             # CASE 1: conv1 inside a BasicBlock → prune OUT channels
#             # ------------------------------------------------------
#             if 'conv1' in layer_name:       # prune output filters
#
#                 prune_count = prune_value[conv_layer]
#                 keep_ratio = 1 - prune_count
#
#                 # Flatten filter bank → (num_filters, filter_dim)
#                 num_filters = W.shape[0]
#                 W_flat = W.view(num_filters, -1)
#
#                 # Train filter autoencoder --------
#                 with torch.no_grad():
#                     W_flat_norm = W_flat.clone()
#
#                 ae = train_filter_autoencoder(
#                     W_flat_norm, latent_dim=latent_dim,
#                     num_epochs=ae_epochs, lr=ae_lr, device=device
#                 )
#
#                 ae.eval()
#                 with torch.no_grad():
#                     z = ae.encode(W_flat_norm)   # shape: [num_filters, latent_dim]
#                     feature_matrix = z.T
#
#                 # Select filters using optimize_frobenius
#                 keep_k = int(num_filters * keep_ratio)
#                 X = optimize_frobenius(feature_matrix, keep_k)
#
#                 first_ele = (X.diag() != 0).nonzero(as_tuple=True)[0].cpu()
#                 out_channels = first_ele
#
#                 # Apply pruning
#                 new_weight = W[out_channels, :, :, :].clone()
#                 layer_module.weight = nn.Parameter(new_weight)
#
#                 # Update metadata
#                 layer_module.out_channels = new_weight.shape[0]
#                 in_channels = list(range(layer_module.in_channels))  # unchanged
#
#             # ------------------------------------------------------
#             # CASE 2: conv2 → prune IN channels (use first_ele)
#             # ------------------------------------------------------
#             elif 'conv2' in layer_name:
#
#                 if first_ele is None:
#                     raise RuntimeError("conv2 encountered before conv1 pruning produced first_ele!")
#
#                 in_channels = first_ele.cpu()
#
#                 # Keep all output channels for conv2
#                 out_channels = torch.arange(W.shape[0])
#
#                 # Prune INPUT channels only
#                 new_weight = W[:, in_channels, :, :].clone()
#                 layer_module.weight = nn.Parameter(new_weight)
#
#                 layer_module.in_channels = new_weight.shape[1]
#                 layer_module.out_channels = new_weight.shape[0]
#
#                 conv_layer += 1
#
#         # ----------------------------------------
#         # PRUNE CORRESPONDING BATCHNORM LAYERS
#         # ----------------------------------------
#         if isinstance(layer_module, nn.BatchNorm2d) and layer_name != 'bn1' and 'bn1' in layer_name:
#
#             if first_ele is None:
#                 continue
#
#             idx = first_ele.cpu()
#
#             layer_module.weight = nn.Parameter(layer_module.weight.data[idx].clone())
#             layer_module.bias = nn.Parameter(layer_module.bias.data[idx].clone())
#
#             layer_module.running_mean = layer_module.running_mean[idx].clone()
#             layer_module.running_var = layer_module.running_var[idx].clone()
#
#             layer_module.num_features = len(idx)
#
#         # ------------------------
#         # STOP BEFORE FC LAYER
#         # ------------------------
#         if isinstance(layer_module, nn.Linear):
#             break


#############################
########################
############
# def prune_resnet_filters(model, prune_value, latent_dim=27, ae_epochs=500, ae_lr=1e-3):
#
#     device = next(model.parameters()).device
#     conv_layer = 0
#     keep_indices = None
#
#     for layer_name, layer_module in model.named_modules():
#
#         # PRUNE CONV1 (OUT channels)
#         if isinstance(layer_module, nn.Conv2d) and "conv1" in layer_name and layer_name != "conv1":
#
#             prune_ratio = prune_value[conv_layer]
#             W = layer_module.weight.data.clone().to(device)
#             out_c = W.shape[0]
#
#             # Number to keep
#             keep_k = int(out_c * (1 - prune_ratio))
#
#             # Flatten and train AE
#             W_flat = W.view(out_c, -1)
#             ae = train_filter_autoencoder(W_flat, latent_dim, ae_epochs, ae_lr, device)
#             ae.eval()
#             with torch.no_grad():
#                 z = ae.encode(W_flat)
#                 feature_matrix = z.T
#
#             X = optimize_frobenius(feature_matrix, keep_k)
#             keep_indices = (X.diag() != 0).nonzero(as_tuple=True)[0].cpu()
#
#             # Apply pruning to conv1
#             new_weight = W[keep_indices]
#             layer_module.weight = nn.Parameter(new_weight)
#             layer_module.out_channels = keep_k
#
#             conv_layer += 1
#
#         # PRUNE CONV2 (IN channels ONLY)
#         elif isinstance(layer_module, nn.Conv2d) and "conv2" in layer_name:
#
#             if keep_indices is None:
#                 raise RuntimeError("conv2 encountered before conv1 pruning")
#
#             W = layer_module.weight.data.clone().to(device)
#
#             # Prune only input channels
#             new_weight = W[:, keep_indices, :, :].clone()
#
#             layer_module.weight = nn.Parameter(new_weight)
#             layer_module.in_channels = new_weight.shape[1]
#             # DO NOT prune conv2.out_channels
#
#         # PRUNE BN AFTER CONV1
#         elif isinstance(layer_module, nn.BatchNorm2d) and "bn1" in layer_name and keep_indices is not None:
#
#             idx = keep_indices
#             layer_module.weight = nn.Parameter(layer_module.weight.data[idx].clone())
#             layer_module.bias = nn.Parameter(layer_module.bias.data[idx].clone())
#             layer_module.running_mean = layer_module.running_mean[idx].clone()
#             layer_module.running_var = layer_module.running_var[idx].clone()
#             layer_module.num_features = len(idx)
#
#         # STOP AT FC
#         if isinstance(layer_module, nn.Linear):
#             break

import torch
import torch.nn as nn

# ---------------------------------------------------------
#   PRUNE RESNET-56 USING AE + FROBENIUS + CORING-LOGIC
# ---------------------------------------------------------
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# # Note: Ensure the required helper functions (train_filter_autoencoder,
# # optimize_frobenius) are defined and accessible.
#
# # NOTE: The LambdaLayer class must be defined/imported in your script's scope
# # since it is used here for re-initialization.
# from resnet_12 import LambdaLayer
# def prune_resnet_filters(model, prune_rates,
#                          latent_dim=27, ae_epochs=500, ae_lr=1e-3):
#     """
#     Prunes a pre-trained ResNet model using the AE-Frobenius method,
#     including the critical step of rebuilding the LambdaLayer shortcut
#     to fix channel dimension mismatches.
#     """
#
#     device = next(model.parameters()).device
#
#     # Assuming ResNet-56 structure for indexing (3 stages of 9 blocks each)
#     cfg = [9, 9, 9]
#
#     conv_id = 0  # Index for prune_rates array
#     prev_selected = None  # Propagated indices (output of previous layer/block)
#
#     # =========================================================
#     # 0) PRUNE INITIAL CONV1 AND BN1
#     # =========================================================
#     conv1_init = model.conv1
#     W1_init = conv1_init.weight.data.clone().to(device)
#     out_c1_init = W1_init.shape[0]
#
#     prune_ratio_init = prune_rates[conv_id]
#     keep_k_init = max(1, int(out_c1_init * (1 - prune_ratio_init)))
#
#     W1_init_flat = W1_init.view(out_c1_init, -1)
#     ae_init = train_filter_autoencoder(W1_init_flat.to(device), latent_dim, ae_epochs, ae_lr, device)
#
#     with torch.no_grad():
#         Z_init = ae_init.encode(W1_init_flat.to(device))
#         feature_matrix_init = Z_init.T
#
#     # FIX: Detach feature matrix before Frobenius optimization
#     feature_matrix_input = feature_matrix_init.clone().detach()
#
#     X_init = optimize_frobenius(feature_matrix_input, keep_k_init)
#
#     top_k_vals, keep_idx_init = torch.topk(X_init.diag(), keep_k_init)
#     keep_idx_init = keep_idx_init.cpu()
#
#     # Apply Pruning to Conv1_init and BN1_init
#     new_W1_init = W1_init[keep_idx_init]
#     conv1_init.weight = nn.Parameter(new_W1_init)
#     conv1_init.out_channels = new_W1_init.shape[0]
#
#     bn1 = model.bn1
#     bn1.weight.data = bn1.weight.data[keep_idx_init].clone()
#     bn1.bias.data = bn1.bias.data[keep_idx_init].clone()
#     bn1.running_mean = bn1.running_mean[keep_idx_init].clone()
#     bn1.running_var = bn1.running_var[keep_idx_init].clone()
#     bn1.num_features = len(keep_idx_init)
#
#     prev_selected = keep_idx_init.clone()
#     conv_id += 1
#
#     # =========================================================
#     # 1) PRUNE RESIDUAL BLOCKS (layer1, layer2, layer3)
#     # =========================================================
#     for stage_index, num_blocks in enumerate(cfg):
#         layer_name = f"layer{stage_index + 1}"
#         sequential_layer = getattr(model, layer_name)
#
#         for block_index in range(num_blocks):
#             block = sequential_layer[block_index]
#
#             # Save the input size to the block (Shortcut path size)
#             block_input_size = len(prev_selected)
#
#             # --- 1. PRUNE CONV1 ---
#             conv1 = block.conv1
#             W1 = conv1.weight.data.clone().to(device)
#             W1 = W1[:, prev_selected.to(device), :, :]  # Restrict IN channels
#
#             out_c1 = W1.shape[0]
#             prune_ratio = prune_rates[conv_id]
#             keep_k1 = max(1, int(out_c1 * (1 - prune_ratio)))
#
#             W1_flat = W1.view(out_c1, -1)
#             ae = train_filter_autoencoder(W1_flat.to(device), latent_dim, ae_epochs, ae_lr, device)
#
#             with torch.no_grad():
#                 Z = ae.encode(W1_flat.to(device))
#                 feature_matrix = Z.T
#
#             # FIX: Detach feature matrix before Frobenius optimization
#             feature_matrix_input = feature_matrix.clone().detach()
#
#             X = optimize_frobenius(feature_matrix_input, keep_k1)
#
#             top_k_vals, keep_idx_conv1 = torch.topk(X.diag(), keep_k1)
#             keep_idx_conv1 = keep_idx_conv1.cpu()
#
#             # Apply Pruning to Conv1 and BN1
#             new_W1 = W1[keep_idx_conv1]
#             conv1.weight = nn.Parameter(new_W1)
#             conv1.out_channels = new_W1.shape[0]
#             conv1.in_channels = new_W1.shape[1]
#
#             bn1 = block.bn1
#             bn1.weight.data = bn1.weight.data[keep_idx_conv1].clone()
#             bn1.bias.data = bn1.bias.data[keep_idx_conv1].clone()
#             bn1.running_mean = bn1.running_mean[keep_idx_conv1].clone()
#             bn1.running_var = bn1.running_var[keep_idx_conv1].clone()
#             bn1.num_features = len(keep_idx_conv1)
#
#             prev_selected_conv1_out = keep_idx_conv1.clone()
#             block.inplanes = conv1.in_channels  # Update block metadata
#             conv_id += 1
#
#             # --- 2. PRUNE CONV2 ---
#             conv2 = block.conv2
#             W2 = conv2.weight.data.clone().to(device)
#             W2 = W2[:, prev_selected_conv1_out.to(device), :, :]
#
#             out_c2 = W2.shape[0]
#             prune_ratio2 = prune_rates[conv_id]
#             keep_k2 = max(1, int(out_c2 * (1 - prune_ratio2)))
#
#             # --- CRITICAL CONSTRAINT ENFORCEMENT ---
#             is_identity_block = isinstance(block.shortcut, nn.Sequential)
#
#             if is_identity_block:
#                 # Identity Block: Output size MUST match the input size (block_input_size = 26)
#                 keep_k2 = block_input_size
#
#             # Perform AE-Frobenius Pruning
#             W2_flat = W2.view(out_c2, -1)
#             ae2 = train_filter_autoencoder(W2_flat.to(device), latent_dim, ae_epochs, ae_lr, device)
#
#             with torch.no_grad():
#                 Z2 = ae2.encode(W2_flat.to(device))
#                 feature_matrix2 = Z2.T
#
#             # FIX: Detach feature matrix before Frobenius optimization
#             feature_matrix_input2 = feature_matrix2.clone().detach()
#
#             X2 = optimize_frobenius(feature_matrix_input2, keep_k2)
#
#             top_k_vals2, keep_idx_conv2 = torch.topk(X2.diag(), keep_k2)
#             keep_idx_conv2 = keep_idx_conv2.cpu()
#
#             # Apply Pruning to Conv2 and BN2
#             new_W2 = W2[keep_idx_conv2]
#             conv2.weight = nn.Parameter(new_W2)
#             conv2.out_channels = new_W2.shape[0]
#             conv2.in_channels = new_W2.shape[1]
#
#             bn2 = block.bn2
#             bn2.weight.data = bn2.weight.data[keep_idx_conv2].clone()
#             bn2.bias.data = bn2.bias.data[keep_idx_conv2].clone()
#             bn2.running_mean = bn2.running_mean[keep_idx_conv2].clone()
#             bn2.running_var = bn2.running_var[keep_idx_conv2].clone()
#             bn2.num_features = len(keep_idx_conv2)
#
#             # Update block metadata and propagate index
#             block.planes = conv2.out_channels  # The new output size (e.g., 21)
#             prev_selected = keep_idx_conv2.clone()
#             conv_id += 1
#
#             # ==========================================================
#             # 3) CRITICAL FIX: REBUILD LAMBDALAYER SHORTCUT
#             # ==========================================================
#             # This fixes the issue where the LambdaLayer closure holds stale values
#             # for 'inplanes' and 'planes' from the block's __init__
#
#             inplanes = block.conv1.in_channels
#             planes = block.conv2.out_channels
#             stride = block.stride
#
#             # Re-create the LambdaLayer only if the shortcut was originally created
#             # (i.e., it's not a simple nn.Sequential() which happens when inplanes == planes and stride=1)
#             if stride != 1 or inplanes != planes:
#
#                 # Check for stride=1 case (where inplanes should have been forced to equal planes)
#                 if stride == 1:
#                     # This handles the case where stride=1 but channels still changed
#                     # (which shouldn't happen for identity blocks if the logic above worked,
#                     # but is included for safety and completeness).
#                     block.shortcut = LambdaLayer(
#                         lambda x: F.pad(x[:, :, :, :],
#                                         (0, 0, 0, 0, (planes - inplanes) // 2,
#                                          planes - inplanes - (planes - inplanes) // 2), "constant", 0))
#
#                 # Check for stride!=1 case (downsampling/projection)
#                 elif stride != 1:
#                     block.shortcut = LambdaLayer(
#                         lambda x: F.pad(x[:, :, ::2, ::2],
#                                         (0, 0, 0, 0, (planes - inplanes) // 2,
#                                          planes - inplanes - (planes - inplanes) // 2), "constant", 0))
#
#     # =========================================================
#     # 2) PRUNE FINAL FC LAYER
#     # =========================================================
#     fc_layer = model.fc if model.num_layer == 56 else model.linear
#
#     W_fc = fc_layer.weight.data.clone().to(device)
#     W_fc_pruned = W_fc[:, prev_selected.to(device)]
#
#     # Update FC layer
#     fc_layer.weight = nn.Parameter(W_fc_pruned)
#     fc_layer.in_features = W_fc_pruned.size(1)
#
#     print("✔ AE + Frobenius ResNet pruning successful (Conv1 + Blocks + FC).")
#     print("✔ LambdaLayer shortcut rebuilt to fix channel mismatch.")
#     return model


############################
########################
####################
#Code for both conv1 and conv2 pruning
import torch
import torch.nn as nn
from resnet_12 import LambdaLayer   # Your model uses this
# train_filter_autoencoder and optimize_frobenius must already exist


# ----------------------------------------------------------
# Build a Conv1×1 + BN shortcut PROPERLY on GPU
# ----------------------------------------------------------
def build_shortcut(inC, outC, stride, device):
    sc = nn.Sequential(
        nn.Conv2d(inC, outC, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(outC)
    )
    return sc.to(device)


# ----------------------------------------------------------
# SAFETY CHECKER:
# Validates channel match after pruning
# ----------------------------------------------------------
def check_block_integrity(block, block_id):

    c1_in  = block.conv1.in_channels
    c1_out = block.conv1.out_channels

    c2_in  = block.conv2.in_channels
    c2_out = block.conv2.out_channels

    # Conv1 OUT == Conv2 IN
    if c1_out != c2_in:
        raise RuntimeError(
            f"[Integrity Error @ Block {block_id}] conv1_out={c1_out}, "
            f"but conv2_in={c2_in}"
        )

    # Shortcut output must match conv2 output
    if isinstance(block.shortcut, nn.Sequential) and len(block.shortcut) > 0:
        sc_out = block.shortcut[0].out_channels
    else:
        sc_out = c2_out  # identity

    if sc_out != c2_out:
        raise RuntimeError(
            f"[Integrity Error @ Block {block_id}] shortcut_out={sc_out}, "
            f"but conv2_out={c2_out}"
        )


# ----------------------------------------------------------
# FULL PRUNING FUNCTION (SAFE VERSION)
# ----------------------------------------------------------
def prune_resnet_filters(model, prune_rates,
                         latent_dim=27, ae_epochs=400, ae_lr=1e-3):

    device = next(model.parameters()).device
    conv_id = 0
    prev_keep = None

    # ------------------------------------------------------
    # 0) STEM CONV
    # ------------------------------------------------------
    stem = model.conv1
    W = stem.weight.data.clone().to(device)
    C_out = W.shape[0]

    prune_ratio = prune_rates[conv_id]
    keep_k = max(1, int(C_out * (1 - prune_ratio)))

    W_flat = W.view(C_out, -1)
    ae = train_filter_autoencoder(W_flat, latent_dim, ae_epochs, ae_lr, device)
    ae.eval()
    with torch.no_grad():
        Z = ae.encode(W_flat)
        F = Z.T

    X = optimize_frobenius(F, keep_k)
    keep_idx = (X.diag() != 0).nonzero(as_tuple=True)[0].cpu()

    # Apply pruning
    new_W = W[keep_idx]
    stem.weight = nn.Parameter(new_W)
    stem.out_channels = new_W.shape[0]

    # BN1 update
    bn1 = model.bn1
    bn1.weight = nn.Parameter(bn1.weight.data[keep_idx])
    bn1.bias   = nn.Parameter(bn1.bias.data[keep_idx])
    bn1.running_mean = bn1.running_mean[keep_idx]
    bn1.running_var  = bn1.running_var[keep_idx]
    bn1.num_features = len(keep_idx)

    prev_keep = keep_idx.clone()
    conv_id += 1

    # ------------------------------------------------------
    # 1) PRUNE ALL BLOCKS (3 stages × 9 blocks)
    # ------------------------------------------------------
    cfg = [9, 9, 9]
    block_count = 0

    for stage, num_blocks in enumerate(cfg):
        layer = getattr(model, f"layer{stage+1}")

        for b in range(num_blocks):
            block = layer[b]
            block_count += 1

            stride = block.stride

            # ==================================================
            # PRUNE CONV1 (OUT + IN)
            # ==================================================
            conv1 = block.conv1
            W1 = conv1.weight.data.clone().to(device)
            C_out1 = W1.shape[0]

            prune_ratio = prune_rates[conv_id]
            keep_k = max(1, int(C_out1 * (1 - prune_ratio)))

            W1_flat = W1.view(C_out1, -1)
            ae1 = train_filter_autoencoder(W1_flat, latent_dim, ae_epochs, ae_lr, device)
            ae1.eval()
            with torch.no_grad():
                Z1 = ae1.encode(W1_flat)
                F1 = Z1.T

            X1 = optimize_frobenius(F1, keep_k)
            keep_idx1 = (X1.diag() != 0).nonzero(as_tuple=True)[0].cpu()

            # OUT prune
            new_W1 = W1[keep_idx1]
            # IN prune
            new_W1 = new_W1[:, prev_keep, :, :]

            conv1.weight = nn.Parameter(new_W1)
            conv1.out_channels = new_W1.shape[0]
            conv1.in_channels  = new_W1.shape[1]

            # BN1 prune
            bn1 = block.bn1
            bn1.weight = nn.Parameter(bn1.weight.data[keep_idx1])
            bn1.bias   = nn.Parameter(bn1.bias.data[keep_idx1])
            bn1.running_mean = bn1.running_mean[keep_idx1]
            bn1.running_var  = bn1.running_var[keep_idx1]
            bn1.num_features = len(keep_idx1)

            prev_keep = keep_idx1.clone()
            conv_id += 1

            # ==================================================
            # PRUNE CONV2 (OUT + IN)
            # ==================================================
            conv2 = block.conv2
            W2 = conv2.weight.data.clone().to(device)
            C_out2 = W2.shape[0]

            prune_ratio2 = prune_rates[conv_id]
            keep_k2 = max(1, int(C_out2 * (1 - prune_ratio2)))

            W2_flat = W2.view(C_out2, -1)
            ae2 = train_filter_autoencoder(W2_flat, latent_dim, ae_epochs, ae_lr, device)
            ae2.eval()

            with torch.no_grad():
                Z2 = ae2.encode(W2_flat)
                F2 = Z2.T

            X2 = optimize_frobenius(F2, keep_k2)
            keep_idx2 = (X2.diag() != 0).nonzero(as_tuple=True)[0].cpu()

            # OUT prune
            new_W2 = W2[keep_idx2]
            # IN prune (from conv1)
            new_W2 = new_W2[:, prev_keep, :, :]

            conv2.weight = nn.Parameter(new_W2)
            conv2.out_channels = new_W2.shape[0]
            conv2.in_channels  = new_W2.shape[1]

            # BN2 prune
            bn2 = block.bn2
            bn2.weight = nn.Parameter(bn2.weight.data[keep_idx2])
            bn2.bias   = nn.Parameter(bn2.bias.data[keep_idx2])
            bn2.running_mean = bn2.running_mean[keep_idx2]
            bn2.running_var  = bn2.running_var[keep_idx2]
            bn2.num_features = len(keep_idx2)

            # New propagated index
            prev_keep = keep_idx2.clone()
            conv_id += 1

            # ==================================================
            # FIX SHORTCUT PROPERLY
            # ==================================================
            inC  = conv1.in_channels
            outC = conv2.out_channels

            need_proj = (inC != outC) or (stride != 1)

            if need_proj:
                block.shortcut = build_shortcut(inC, outC, stride, device)
            else:
                block.shortcut = nn.Sequential().to(device)

            # SAFETY CHECK
            check_block_integrity(block, block_count)

    # ------------------------------------------------------
    # 2) FIX FC LAYER
    # ------------------------------------------------------
    final_channels = prev_keep.shape[0]

    fc_old = model.fc
    new_fc = nn.Linear(final_channels, fc_old.out_features).to(device)

    new_fc.weight = nn.Parameter(fc_old.weight.data[:, prev_keep].clone().to(device))
    new_fc.bias   = nn.Parameter(fc_old.bias.data.clone().to(device))

    model.fc = new_fc

    print("✔ SAFE PRUNING COMPLETED (Shortcuts + FC + Integrity Checks OK)\n")



###############
###########################
def prune_resnet_filters_without_ae(model, prune_rates,
                         latent_dim=27, ae_epochs=400, ae_lr=1e-3):

    device = next(model.parameters()).device
    conv_id = 0
    prev_keep = None

    # ------------------------------------------------------
    # 0) STEM CONV
    # ------------------------------------------------------
    stem = model.conv1
    W = stem.weight.data.clone().to(device)
    C_out = W.shape[0]

    prune_ratio = prune_rates[conv_id]
    keep_k = max(1, int(C_out * (1 - prune_ratio)))

    W_flat = W.view(C_out, -1)
    F = W_flat.T
    X = optimize_frobenius(F, keep_k)
    keep_idx = (X.diag() != 0).nonzero(as_tuple=True)[0].cpu()

    # Apply pruning
    new_W = W[keep_idx]
    stem.weight = nn.Parameter(new_W)
    stem.out_channels = new_W.shape[0]

    # BN1 update
    bn1 = model.bn1
    bn1.weight = nn.Parameter(bn1.weight.data[keep_idx])
    bn1.bias   = nn.Parameter(bn1.bias.data[keep_idx])
    bn1.running_mean = bn1.running_mean[keep_idx]
    bn1.running_var  = bn1.running_var[keep_idx]
    bn1.num_features = len(keep_idx)

    prev_keep = keep_idx.clone()
    conv_id += 1

    # ------------------------------------------------------
    # 1) PRUNE ALL BLOCKS (3 stages × 9 blocks)
    # ------------------------------------------------------
    cfg = [9, 9, 9]
    block_count = 0

    for stage, num_blocks in enumerate(cfg):
        layer = getattr(model, f"layer{stage+1}")

        for b in range(num_blocks):
            block = layer[b]
            block_count += 1

            stride = block.stride

            # ==================================================
            # PRUNE CONV1 (OUT + IN)
            # ==================================================
            conv1 = block.conv1
            W1 = conv1.weight.data.clone().to(device)
            C_out1 = W1.shape[0]

            prune_ratio = prune_rates[conv_id]
            keep_k = max(1, int(C_out1 * (1 - prune_ratio)))

            W1_flat = W1.view(C_out1, -1)
            F1 = W1_flat.T
            X1 = optimize_frobenius(F1, keep_k)
            keep_idx1 = (X1.diag() != 0).nonzero(as_tuple=True)[0].cpu()

            # OUT prune
            new_W1 = W1[keep_idx1]
            # IN prune
            new_W1 = new_W1[:, prev_keep, :, :]

            conv1.weight = nn.Parameter(new_W1)
            conv1.out_channels = new_W1.shape[0]
            conv1.in_channels  = new_W1.shape[1]

            # BN1 prune
            bn1 = block.bn1
            bn1.weight = nn.Parameter(bn1.weight.data[keep_idx1])
            bn1.bias   = nn.Parameter(bn1.bias.data[keep_idx1])
            bn1.running_mean = bn1.running_mean[keep_idx1]
            bn1.running_var  = bn1.running_var[keep_idx1]
            bn1.num_features = len(keep_idx1)

            prev_keep = keep_idx1.clone()
            conv_id += 1

            # ==================================================
            # PRUNE CONV2 (OUT + IN)
            # ==================================================
            conv2 = block.conv2
            W2 = conv2.weight.data.clone().to(device)
            C_out2 = W2.shape[0]

            prune_ratio2 = prune_rates[conv_id]
            keep_k2 = max(1, int(C_out2 * (1 - prune_ratio2)))

            W2_flat = W2.view(C_out2, -1)
            F2 = W2_flat.T
            X2 = optimize_frobenius(F2, keep_k2)
            keep_idx2 = (X2.diag() != 0).nonzero(as_tuple=True)[0].cpu()

            # OUT prune
            new_W2 = W2[keep_idx2]
            # IN prune (from conv1)
            new_W2 = new_W2[:, prev_keep, :, :]

            conv2.weight = nn.Parameter(new_W2)
            conv2.out_channels = new_W2.shape[0]
            conv2.in_channels  = new_W2.shape[1]

            # BN2 prune
            bn2 = block.bn2
            bn2.weight = nn.Parameter(bn2.weight.data[keep_idx2])
            bn2.bias   = nn.Parameter(bn2.bias.data[keep_idx2])
            bn2.running_mean = bn2.running_mean[keep_idx2]
            bn2.running_var  = bn2.running_var[keep_idx2]
            bn2.num_features = len(keep_idx2)

            # New propagated index
            prev_keep = keep_idx2.clone()
            conv_id += 1

            # ==================================================
            # FIX SHORTCUT PROPERLY
            # ==================================================
            inC  = conv1.in_channels
            outC = conv2.out_channels

            need_proj = (inC != outC) or (stride != 1)

            if need_proj:
                block.shortcut = build_shortcut(inC, outC, stride, device)
            else:
                block.shortcut = nn.Sequential().to(device)

            # SAFETY CHECK
            check_block_integrity(block, block_count)

    # ------------------------------------------------------
    # 2) FIX FC LAYER
    # ------------------------------------------------------
    final_channels = prev_keep.shape[0]

    fc_old = model.fc
    new_fc = nn.Linear(final_channels, fc_old.out_features).to(device)

    new_fc.weight = nn.Parameter(fc_old.weight.data[:, prev_keep].clone().to(device))
    new_fc.bias   = nn.Parameter(fc_old.bias.data.clone().to(device))

    model.fc = new_fc

    print("✔ SAFE PRUNING COMPLETED (Shortcuts + FC + Integrity Checks OK)\n")

#############################
def build_downsample(inC, outC, stride, device):
    return nn.Sequential(
        nn.Conv2d(inC, outC, kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(outC)
    ).to(device)


# ---------------------------------------------------------
# SAFETY CHECK FOR BOTTLENECK
# ---------------------------------------------------------
def check_bottleneck(block, block_id):
    assert block.conv1.out_channels == block.conv2.in_channels, \
        f"[Block {block_id}] conv1→conv2 mismatch"

    assert block.conv2.out_channels == block.conv3.in_channels, \
        f"[Block {block_id}] conv2→conv3 mismatch"

    if getattr(block, "is_downsample", False):
        ds_out = block.downsample[0].out_channels
        c3_out = block.conv3.out_channels
        assert ds_out == c3_out, \
            f"[Block {block_id}] downsample_out != conv3_out"


# ---------------------------------------------------------
# MAIN PRUNING FUNCTION
# ---------------------------------------------------------
def prune_resnet50_filters_12(
    model,
    prune_rates,
    latent_dim=27,
    ae_epochs=500,
    ae_lr=1e-3
):
    device = next(model.parameters()).device
    conv_id = 0

    # =====================================================
    # 0) STEM CONV
    # =====================================================
    stem = model.conv1
    W = stem.weight.data.clone().to(device)
    Cout = W.shape[0]

    keep_k = max(1, int(Cout * (1 - prune_rates[conv_id])))
    Wf = W.view(Cout, -1)

    ae = train_filter_autoencoder(Wf, latent_dim, ae_epochs, ae_lr, device)
    ae.eval()
    with torch.no_grad():
        F = ae.encode(Wf).T

    X = optimize_frobenius(F, keep_k)
    keep_idx = (X.diag() > 0).nonzero(as_tuple=True)[0].cpu()

    stem.weight = nn.Parameter(W[keep_idx])
    stem.out_channels = len(keep_idx)

    bn = model.bn1
    bn.weight = nn.Parameter(bn.weight.data[keep_idx])
    bn.bias   = nn.Parameter(bn.bias.data[keep_idx])
    bn.running_mean = bn.running_mean[keep_idx]
    bn.running_var  = bn.running_var[keep_idx]
    bn.num_features = len(keep_idx)

    prev_keep = keep_idx.clone()
    conv_id += 1

    # =====================================================
    # RESNET-50 CONFIG
    # =====================================================
    cfg = [3, 4, 6, 3]  # blocks per stage
    block_id = 0

    for stage, num_blocks in enumerate(cfg):
        layer = getattr(model, f"layer{stage+1}")
        stage_input_keep = prev_keep.clone()

        for i in range(num_blocks):
            block = layer[i]
            block_id += 1

            # ---------------- conv1 ----------------
            conv1 = block.conv1
            W1 = conv1.weight.data.clone().to(device)
            C1 = W1.shape[0]

            keep_k1 = max(1, int(C1 * (1 - prune_rates[conv_id])))
            W1f = W1.view(C1, -1)

            ae1 = train_filter_autoencoder(W1f, latent_dim, ae_epochs, ae_lr, device)
            ae1.eval()
            with torch.no_grad():
                F1 = ae1.encode(W1f).T

            X1 = optimize_frobenius(F1, keep_k1)
            keep1 = (X1.diag() > 0).nonzero(as_tuple=True)[0].cpu()

            W1_new = W1[keep1][:, prev_keep]
            conv1.weight = nn.Parameter(W1_new)
            conv1.out_channels = len(keep1)
            conv1.in_channels  = len(prev_keep)

            bn1 = block.bn1
            bn1.weight = nn.Parameter(bn1.weight.data[keep1])
            bn1.bias   = nn.Parameter(bn1.bias.data[keep1])
            bn1.running_mean = bn1.running_mean[keep1]
            bn1.running_var  = bn1.running_var[keep1]
            bn1.num_features = len(keep1)

            prev_keep = keep1.clone()
            conv_id += 1

            # ---------------- conv2 ----------------
            conv2 = block.conv2
            W2 = conv2.weight.data.clone().to(device)
            C2 = W2.shape[0]

            keep_k2 = max(1, int(C2 * (1 - prune_rates[conv_id])))
            W2f = W2.view(C2, -1)

            ae2 = train_filter_autoencoder(W2f, latent_dim, ae_epochs, ae_lr, device)
            ae2.eval()
            with torch.no_grad():
                F2 = ae2.encode(W2f).T

            X2 = optimize_frobenius(F2, keep_k2)
            keep2 = (X2.diag() > 0).nonzero(as_tuple=True)[0].cpu()

            W2_new = W2[keep2][:, prev_keep]
            conv2.weight = nn.Parameter(W2_new)
            conv2.out_channels = len(keep2)
            conv2.in_channels  = len(prev_keep)

            bn2 = block.bn2
            bn2.weight = nn.Parameter(bn2.weight.data[keep2])
            bn2.bias   = nn.Parameter(bn2.bias.data[keep2])
            bn2.running_mean = bn2.running_mean[keep2]
            bn2.running_var  = bn2.running_var[keep2]
            bn2.num_features = len(keep2)

            prev_keep = keep2.clone()
            conv_id += 1

            # ---------------- conv3 (NO pruning) ----------------
            conv3 = block.conv3
            W3 = conv3.weight.data.clone().to(device)

            W3_new = W3[:, prev_keep]
            conv3.weight = nn.Parameter(W3_new)
            conv3.in_channels = len(prev_keep)

            prev_keep = torch.arange(conv3.out_channels)

            # ---------------- downsample ----------------
            if i == 0 and getattr(block, "is_downsample", False):
                ds = build_downsample(
                    inC=len(stage_input_keep),
                    outC=conv3.out_channels,
                    stride=block.stride,
                    device=device
                )
                block.downsample = ds

            check_bottleneck(block, block_id)

    # =====================================================
    # FC LAYER
    # =====================================================
    fc_old = model.fc
    new_fc = nn.Linear(len(prev_keep), fc_old.out_features).to(device)
    new_fc.weight = nn.Parameter(fc_old.weight.data[:, prev_keep])
    new_fc.bias   = nn.Parameter(fc_old.bias.data)
    model.fc = new_fc

    print("✅ ResNet-50 pruning completed safely.")
    
def prune_resnet50_filters(model, prune_rates, latent_dim=27, ae_epochs=500, ae_lr=1e-3):
    device = next(model.parameters()).device
    conv_id = 0

    # 1. Prune Stem (conv1 -> bn1)
    # -------------------------------------------------------
    stem = model.conv1
    W = stem.weight.data.clone()
    Cout = W.shape[0]
    keep_k = max(1, int(Cout * (1 - prune_rates[conv_id])))
    
    # AE training MUST NOT be in no_grad
    Wf = W.view(Cout, -1).to(device)
    ae = train_filter_autoencoder(Wf, latent_dim, ae_epochs, ae_lr, device)
    
    with torch.no_grad():
        Z = ae.encode(Wf)
    
    # Selection optimization (requires grad)
    X = optimize_frobenius(Z.T, keep_k)
    keep_idx = (X.diag() != 0).nonzero(as_tuple=True)[0].cpu()
    
    # IN-PLACE SLICE: No new layer initialization
    stem.weight = nn.Parameter(W[keep_idx].clone())
    stem.out_channels = len(keep_idx)
    
    bn1 = model.bn1
    bn1.weight = nn.Parameter(bn1.weight.data[keep_idx].clone())
    bn1.bias = nn.Parameter(bn1.bias.data[keep_idx].clone())
    bn1.running_mean = bn1.running_mean[keep_idx].clone()
    bn1.running_var = bn1.running_var[keep_idx].clone()
    bn1.num_features = len(keep_idx)

    prev_keep = keep_idx
    conv_id += 1

    # 2. Prune Stages
    # -------------------------------------------------------
    for stage_id in range(1, 5):
        layer = getattr(model, f"layer{stage_id}")
        stage_input_keep = prev_keep.clone() # Tracking entrance to the stage

        for i, block in enumerate(layer):
            # --- conv1 ---
            conv1 = block.conv1
            W1 = conv1.weight.data.clone()
            W1 = W1[:, prev_keep, :, :] # Slice inputs to match prev layer
            Cout1 = W1.shape[0]
            keep_k1 = max(1, int(Cout1 * (1 - prune_rates[conv_id])))
            
            Wf1 = W1.view(Cout1, -1).to(device)
            ae1 = train_filter_autoencoder(Wf1, latent_dim, ae_epochs, ae_lr, device)
            with torch.no_grad(): Z1 = ae1.encode(Wf1)
            
            X1 = optimize_frobenius(Z1.T, keep_k1)
            keep1 = (X1.diag() != 0).nonzero(as_tuple=True)[0].cpu()
            
            conv1.weight = nn.Parameter(W1[keep1].clone())
            conv1.in_channels = len(prev_keep)
            conv1.out_channels = len(keep1)
            
            # BN1 In-place
            block.bn1.weight = nn.Parameter(block.bn1.weight.data[keep1].clone())
            block.bn1.bias = nn.Parameter(block.bn1.bias.data[keep1].clone())
            block.bn1.running_mean = block.bn1.running_mean[keep1].clone()
            block.bn1.running_var = block.bn1.running_var[keep1].clone()
            block.bn1.num_features = len(keep1)
            conv_id += 1

            # --- conv2 ---
            conv2 = block.conv2
            W2 = conv2.weight.data.clone()
            W2 = W2[:, keep1, :, :]
            Cout2 = W2.shape[0]
            keep_k2 = max(1, int(Cout2 * (1 - prune_rates[conv_id])))
            
            Wf2 = W2.view(Cout2, -1).to(device)
            ae2 = train_filter_autoencoder(Wf2, latent_dim, ae_epochs, ae_lr, device)
            with torch.no_grad(): Z2 = ae2.encode(Wf2)
            
            X2 = optimize_frobenius(Z2.T, keep_k2)
            keep2 = (X2.diag() != 0).nonzero(as_tuple=True)[0].cpu()
            
            conv2.weight = nn.Parameter(W2[keep2].clone())
            conv2.in_channels = len(keep1)
            conv2.out_channels = len(keep2)
            
            # BN2 In-place
            block.bn2.weight = nn.Parameter(block.bn2.weight.data[keep2].clone())
            block.bn2.bias = nn.Parameter(block.bn2.bias.data[keep2].clone())
            block.bn2.running_mean = block.bn2.running_mean[keep2].clone()
            block.bn2.running_var = block.bn2.running_var[keep2].clone()
            block.bn2.num_features = len(keep2)
            conv_id += 1

            # --- conv3 (Slicing Inputs, Outputs stay constant) ---
            conv3 = block.conv3
            conv3.weight = nn.Parameter(conv3.weight.data[:, keep2, :, :].clone())
            conv3.in_channels = len(keep2)
            
            # --- In-place Downsample Slicing ---
            if i == 0 and hasattr(block, 'downsample') and block.downsample is not None:
                # Slice input channels of existing downsample convolution
                ds_conv = block.downsample[0]
                ds_conv.weight = nn.Parameter(ds_conv.weight.data[:, stage_input_keep, :, :].clone())
                ds_conv.in_channels = len(stage_input_keep)
                
                # Update BN num_features if downsample output was changed (rare in ResNet-50)
                ds_bn = block.downsample[1]
                ds_bn.num_features = ds_conv.out_channels

            # The output width of a block is determined by conv3
            prev_keep = torch.arange(conv3.out_channels)

    # 3. Final FC Layer
    model.fc.weight = nn.Parameter(model.fc.weight.data[:, prev_keep].clone())
    model.fc.in_features = len(prev_keep)
