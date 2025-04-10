import torch
import torch.nn.functional as F
from src.utils.utils import cartesian_to_lonlat

def haversine_distance(pred_coords, true_coords, use_rad=False):
    if (pred_coords.dim() == 2):
        pred_lon = pred_coords[:, 0]  # [batch]
        pred_lat = pred_coords[:, 1]  # [batch]
        true_lon = true_coords[:, 0]  # [batch]
        true_lat = true_coords[:, 1]  # [batch]
    else:
        pred_lon = pred_coords[:, :, 0]  # [batch, seqLen]
        pred_lat = pred_coords[:, :, 1]  # [batch, seqLen]
        true_lon = true_coords[:, :, 0]  # [batch, seqLen]
        true_lat = true_coords[:, :, 1]  # [batch, seqLen]
    
    dlon = true_lon - pred_lon  
    dlat = true_lat - pred_lat  
    
    a = torch.clamp(torch.sin(dlat/2)**2 + torch.cos(pred_lat) * torch.cos(true_lat) * torch.sin(dlon/2)**2, 1 - 0.9999999999, 0.9999999999)
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1-a))
    
    c = torch.nan_to_num(c, nan=0.0, posinf=0.0, neginf=0.0)
    
    if use_rad:
        return c * 6371.0
    else:
        return c

# Spatio loss
def compute_spatio_loss(p_pred, p_true, padding_masks, eps=1e-8):

    (p_pred, delta_coord) = p_pred
    # Calculate Haversine distance and apply padding mask
    geo_distance = haversine_distance(p_pred, p_true)
    loss_sphere = (geo_distance * padding_masks).sum() / (padding_masks.sum() + eps)

    # del_distance = haversine_distance(delta_coord, torch.zeros_like(delta_coord), use_rad=True)
    # delta_coord_loss = (del_distance * padding_masks).sum() / (padding_masks.sum() + eps)
    
    return loss_sphere

# Angle loss
def compute_angle_cos_loss(cyc_pred, cyc_true, padding_masks):
    # Calculate cosine similarity and apply padding mask
    cos_sim = F.cosine_similarity(cyc_pred, cyc_true, dim=-1)  # [b, s]
    masked_cos_sim = cos_sim * padding_masks.squeeze(-1)  # [b, s]
    
    # Calculate mean cosine similarity only over valid positions
    cosine_loss = 1 - (masked_cos_sim.sum() / padding_masks.sum())
    return cosine_loss

def calculate_angle_mse_loss(cyc_pred, cyc_true, padding_masks):
    true_angles = torch.atan2(cyc_true[:,:,0],cyc_true[:,:,1]).squeeze(-1)
    predicted_angles = torch.atan2(cyc_pred[:,:,0],cyc_pred[:,:,1]).squeeze(-1)
    angle_diff = predicted_angles - true_angles
    angle_diff = torch.abs((angle_diff + torch.pi) % (2 * torch.pi) - torch.pi)
    squared_errors = angle_diff ** 2
    # Apply padding mask and calculate mean only over valid positions
    return (squared_errors * padding_masks.squeeze(-1)).sum() / padding_masks.sum()

# Continuous loss
def compute_continuous_mse_loss(x_pred, x_true, padding_masks):
    # Calculate squared errors and apply padding mask for all 4 features
    squared_errors = (x_pred - x_true) ** 2  # [b, s, 4]
    masked_squared_errors = squared_errors * padding_masks  # [b, s, 4]
    
    # If no valid positions, return 0 to avoid division by zero
    if padding_masks.sum() == 0:
        return torch.tensor(0.0, device=x_pred.device)
    
    # Calculate mean squared error across all features and valid positions
    mse_loss = masked_squared_errors.sum() / padding_masks.sum()
    return mse_loss

# Time interval loss
def compute_time_interval_loss(x_pred, x_true, max_delta, padding_masks, eps=1e-8): 
    # Combine valid time interval check with padding mask
    valid_mask = (x_true <= max_delta) & padding_masks
    elementwise_loss = F.huber_loss(x_pred, x_true, delta=10, reduction='none')
    
    # Apply combined mask to losses
    masked_loss = elementwise_loss * valid_mask
    return masked_loss.sum() / (valid_mask.sum() + eps)

# Discrete loss
def compute_discrete_loss(y_true, y_pred, z, W_d_one_hot, padding_masks, lambda_reg=0.1):
    """
    Compute the discrete loss with smoothed cross-entropy and embedding regularization.
    Only missing positions (as indicated by the mask) contribute to the loss computation.

    Parameters:
    - y_true: Ground truth labels (one-hot encoded or smoothed), shape [b, s, 1].
    - y_pred: Predicted class probabilities, shape [b, s, |C|].
    - z: Decoder embeddings, shape [b, s, d].
    - W_d_one_hot: Original encoding space embeddings (W_d^T * one-hot(x)), shape [b, s, d].
    - padding_masks: Mask indicating valid positions, shape [b, s, 1].
            True indicates valid positions, False indicates missing positions.
    - lambda_reg: Regularization strength for embedding alignment (default: 0.1).

    Returns:
    - total_loss: Combined discrete loss.
    - cross_entropy_loss: Smoothed cross-entropy loss component.
    - reg_loss: Embedding regularization loss component.
    """
    # Flatten the tensors for batch-wise computation
    y_true_flat = y_true.squeeze(-1)  # [b, s]
    y_pred_flat = y_pred.view(-1, y_pred.size(-1))  # [b*s, |C|]
    padding_masks_flat = padding_masks.view(-1)  # [b*s]
    y_true_flat = y_true_flat.view(-1)  # [b*s]

    # Compute cross-entropy loss only for missing positions using padding mask
    missing_mask = padding_masks_flat  # Invert mask for missing positions
    y_pred_missing = y_pred_flat[missing_mask]
    y_true_missing = y_true_flat[missing_mask]
    
    cross_entropy_loss = F.cross_entropy(
        y_pred_missing,
        y_true_missing,
        reduction='mean'
    ) if y_pred_missing.numel() > 0 else torch.tensor(0.0)

    # Compute embedding regularization loss with padding mask
    embedding_diff = z - W_d_one_hot  # [b, s, d]
    missing_mask = padding_masks.squeeze(-1)  # [b, s]
    
    # Calculate L2 norm and apply missing mask
    reg_loss = torch.norm(embedding_diff, p=2, dim=-1) * missing_mask  # [b, s]
    reg_loss = reg_loss.sum() / (missing_mask.sum() + 1e-8)  # Avoid division by zero

    # Combine losses
    total_loss = cross_entropy_loss + lambda_reg * reg_loss

    return total_loss
