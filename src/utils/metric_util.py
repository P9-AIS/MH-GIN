import torch
import numpy as np

def calculate_one_mae(true_values, predicted_values, masks, masks_turn=False):
    """
    MAE for continuous values with mask support.
    """
    if masks_turn:
        masks = masks.permute(0, 2, 1)
    masks = masks.bool()
    
    diff = torch.abs(true_values - predicted_values)
    masked_diff = diff[~masks]
    
    if masked_diff.numel() == 0:
        return torch.tensor(0.0, device=true_values.device)
    
    return masked_diff.mean()

def calculate_one_smape(true_values, predicted_values, masks, masks_turn=False, epsilon=1e-8):
    """
    Symmetric MAPE for continuous values with mask support.
    """
    if masks_turn:
        masks = masks.permute(0, 2, 1)
    masks = masks.bool()
    
    masked_true = true_values[~masks]
    masked_pred = predicted_values[~masks]
    
    if masked_true.numel() == 0:
        return torch.tensor(0.0, device=true_values.device)
    
    numerator = torch.abs(masked_true - masked_pred)
    denominator = (torch.abs(masked_true) + torch.abs(masked_pred)) / 2 + epsilon
    return (numerator / denominator).mean()



def calculate_one_angle_mae(true_angles, predicted_angles, masks, masks_turn=False):
    """
    MAE for angles (radians), wrapped in [-pi, pi].
    """
    if masks_turn:
        masks = masks.permute(0, 2, 1)
    masks = masks.bool()
    
    masked_true = true_angles[~masks]
    masked_pred = predicted_angles[~masks]
    
    if masked_true.numel() == 0:
        return torch.tensor(0.0, device=true_angles.device)
    
    angle_diff = (masked_pred - masked_true + torch.pi) % (2 * torch.pi) - torch.pi
    return torch.abs(angle_diff).mean()

def calculate_one_angle_smape(true_angles, predicted_angles, masks, masks_turn=False, epsilon=1e-8):
    """
    SMAPE for angles (radians), wrapped in [-pi, pi].
    """
    if masks_turn:
        masks = masks.permute(0, 2, 1)
    masks = masks.bool()
    
    masked_true = true_angles[~masks]
    masked_pred = predicted_angles[~masks]
    
    if masked_true.numel() == 0:
        return torch.tensor(0.0, device=true_angles.device)
    
    angle_diff = (masked_pred - masked_true + torch.pi) % (2 * torch.pi) - torch.pi
    denominator = (torch.abs(masked_true) + torch.abs(masked_pred)) / 2 + epsilon
    return (torch.abs(angle_diff) / denominator).mean()



def calculate_acc(predictions, true_labels, mask):
    """
    Compute macro accuracy for discrete classification.
    
    predictions: [B, S, num_classes], torch tensor (logits or probabilities)
    true_labels: [B, S, 1], torch tensor (long)
    mask: [B, S, 1], torch tensor (bool or 0/1)
    """
    mask = mask.bool().squeeze(-1)
    batch_size = predictions.shape[0]
    
    acc_scores = []
    for i in range(batch_size):
        valid_pred = predictions[i][mask[i]]  # [valid_steps, num_classes]
        valid_true = true_labels[i][mask[i]].squeeze(-1)  # [valid_steps]
        if valid_true.numel() == 0:
            continue
        predicted_labels = valid_pred.argmax(dim=-1)
        acc_scores.append((predicted_labels == valid_true).float().mean())
    
    if len(acc_scores) == 0:
        return torch.tensor(0.0, device=predictions.device)
    
    return torch.stack(acc_scores).mean()

def calculate_acc_torch(predictions, true_labels, mask):
    """
    predictions: [b, s, |C|] (logits or probabilities)
    true_labels: [b, s, 1] (integer labels)
    mask: [b, s, 1] boolean mask (True=invalid)
    """
    valid_mask = (~mask).squeeze(-1)  # [b, s]
    batch_size = predictions.shape[0]
    acc_scores = []

    for i in range(batch_size):
        pred_i = predictions[i][valid_mask[i]]           # [s_valid, |C|]
        true_i = true_labels[i][valid_mask[i]].squeeze(-1)  # [s_valid]

        if pred_i.shape[0] == 0:
            continue

        predicted_labels = torch.argmax(pred_i, dim=-1)
        acc = (predicted_labels == true_i).float().mean()
        acc_scores.append(acc)

    if len(acc_scores) == 0:
        return torch.tensor(0.0, device=predictions.device)
    
    return torch.stack(acc_scores).mean()



def haversine_distance(pred, true):
    """
    pred, true: [N, 2] tensors, lon/lat in radians
    Returns approximate haversine distance in km.
    """
    R = 6371.0
    dlat = true[:, 1] - pred[:, 1]
    dlon = true[:, 0] - pred[:, 0]
    a = torch.sin(dlat/2)**2 + torch.cos(pred[:, 1]) * torch.cos(true[:, 1]) * torch.sin(dlon/2)**2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    return R * c

def batch_cpu_dtw(true_trajectories, predicted_trajectories, masks):
    """
    Simple CPU batch DTW using torch tensors
    """
    batch_size = true_trajectories.shape[0]
    dtw_distances = []
    
    true_trajectories = true_trajectories.cpu()
    predicted_trajectories = predicted_trajectories.cpu()
    masks = masks.cpu().bool().squeeze(-1)
    
    for i in range(batch_size):
        mask_i = masks[i]
        true_i = true_trajectories[i][mask_i]
        pred_i = predicted_trajectories[i][mask_i]
        if true_i.shape[0] == 0:
            dtw_distances.append(torch.tensor(0.0))
            continue
        
        n, m = true_i.shape[0], pred_i.shape[0]
        dtw = torch.zeros(n, m)
        for j in range(n):
            for k in range(m):
                cost = haversine_distance(pred_i[k:k+1], true_i[j:j+1])
                if j == 0 and k == 0:
                    dtw[j, k] = cost
                elif j == 0:
                    dtw[j, k] = cost + dtw[j, k-1]
                elif k == 0:
                    dtw[j, k] = cost + dtw[j-1, k]
                else:
                    dtw[j, k] = cost + torch.min(dtw[j-1, k], dtw[j, k-1], dtw[j-1, k-1])
        dtw_distances.append(dtw[-1, -1])
    
    return torch.stack(dtw_distances).mean()