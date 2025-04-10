import torch
import numpy as np

def calculate_one_mae(true_values, predicted_values, masks, masks_turn=False):
    # Move tensors to CPU, detach, and convert to numpy arrays. Expected shape: [B, S, 1]
    true_np = true_values.detach().cpu().numpy()
    pred_np = predicted_values.detach().cpu().numpy()
    masks_np = masks.detach().cpu().numpy()
    
    # If masks_turn is True, transpose the masks from [B, S, 1] to [B, 1, S], then squeeze to [B, S]
    if masks_turn:
        masks_np = np.transpose(masks_np, (0, 2, 1))
        masks_np = np.squeeze(masks_np, axis=1)
    else:
        masks_np = np.squeeze(masks_np, axis=-1)  # Now shape [B, S]
        
    # Squeeze the last dimension of true and predicted values to get shape [B, S]
    true_np = np.squeeze(true_np, axis=-1)
    pred_np = np.squeeze(pred_np, axis=-1)
    
    # Use the mask to only select valid entries along the sequence dimension
    valid_indices = ~masks_np  # Boolean mask with shape [B, S]
    
    # Compute the absolute errors from the valid entries and get the mean
    absolute_errors = np.abs(true_np[valid_indices] - pred_np[valid_indices])
    
    # Handle potential NaN values
    if len(absolute_errors) == 0:
        return 0.0
    mae = np.nanmean(absolute_errors)  # Use nanmean to handle NaN values
    return mae if not np.isnan(mae) else 0.0

def calculate_one_smape(true_values, predicted_values, masks, masks_turn=False, epsilon=1e-8):
    """
    Calculate the Symmetric MAPE (SMAPE) for valid positions.
    
    Args:
        true_values (torch.Tensor): Ground truth values, shape [B, S, 1].
        predicted_values (torch.Tensor): Predicted values, shape [B, S, 1].
        masks (torch.Tensor): Mask indicating invalid positions, shape [B, S, 1] or [B, 1, S].
        masks_turn (bool): Whether to transpose the mask dimensions. Default is False.
        epsilon (float): Small value to avoid division by zero.
    
    Returns:
        float: Mean Symmetric MAPE (SMAPE).
    """
    # Move tensors to CPU and convert to numpy arrays
    true_np = true_values.detach().cpu().numpy()
    pred_np = predicted_values.detach().cpu().numpy()
    masks_np = masks.detach().cpu().numpy()
    
    # If required, transpose masks then squeeze to reduce dimensions to [B, S]
    if masks_turn:
        masks_np = np.transpose(masks_np, (0, 2, 1))
        masks_np = np.squeeze(masks_np, axis=1)
    else:
        masks_np = np.squeeze(masks_np, axis=-1)
    
    # Squeeze the last dimension of true and predicted values to get [B, S]
    true_np = np.squeeze(true_np, axis=-1)
    pred_np = np.squeeze(pred_np, axis=-1)
    
    # Apply the mask to get only valid positions along the sequence dimension
    valid_indices = ~masks_np
    
    # If all indices are invalid (all False), return 0
    if not valid_indices.any():
        return 0.0
        
    y_true = true_np[valid_indices]
    y_pred = pred_np[valid_indices]
    
    # Convert back to torch tensors for SMAPE computation
    y_true = torch.tensor(y_true, dtype=torch.float32)
    y_pred = torch.tensor(y_pred, dtype=torch.float32)
    
    # Compute Symmetric MAPE (SMAPE)
    numerator = torch.abs(y_true - y_pred)
    denominator = (torch.abs(y_true) + torch.abs(y_pred)) / 2 + epsilon
    relative_error = numerator / denominator
    smape = torch.mean(relative_error)
    
    return smape.item()

def calculate_one_angle_mae(true_angles, predicted_angles, masks, masks_turn=False):
    if masks_turn:  # [B, maxlen, 1]
        masks = masks.permute(0, 2, 1)
        masks_np = np.squeeze(masks.detach().cpu().numpy(), axis=1)
    else:
        masks_np = np.squeeze(masks.detach().cpu().numpy(), axis=-1)
        
    true_np = np.squeeze(true_angles.detach().cpu().numpy(), axis=-1)
    pred_np = np.squeeze(predicted_angles.detach().cpu().numpy(), axis=-1)
    
    # Use the mask (now a numpy array) to select only valid entries; the sequence dimension will shrink.
    valid_indices = ~masks_np
    true_valid = true_np[valid_indices]
    pred_valid = pred_np[valid_indices]
    
    # Compute the adjusted angle differences in the range [-π, π)
    angle_diff = (pred_valid - true_valid + np.pi) % (2 * np.pi) - np.pi
    
    absolute_errors = np.abs(angle_diff)
    mae = np.mean(absolute_errors)
    return mae

def calculate_one_angle_smape(true_angles, predicted_angles, masks, masks_turn=False, epsilon=1e-8):
    """
    Calculate the Symmetric MAPE (SMAPE) for angle differences.
    
    Args:
        true_angles (torch.Tensor): Ground truth angles, shape [B, S, 1].
        predicted_angles (torch.Tensor): Predicted angles, shape [B, S, 1].
        masks (torch.Tensor): Mask indicating invalid positions, shape [B, S, 1] or [B, 1, S].
        masks_turn (bool): Whether to transpose the mask dimensions. Default is False.
        epsilon (float): Small value to avoid division by zero.
    
    Returns:
        float: Mean Symmetric MAPE (SMAPE).
    """
    if masks_turn:  # [B, maxlen, 1]
        masks = masks.permute(0, 2, 1)
        masks_np = np.squeeze(masks.detach().cpu().numpy(), axis=1)
    else:
        masks_np = np.squeeze(masks.detach().cpu().numpy(), axis=-1)
        
    true_np = np.squeeze(true_angles.detach().cpu().numpy(), axis=-1)
    pred_np = np.squeeze(predicted_angles.detach().cpu().numpy(), axis=-1)
    
    # Use the mask (now a numpy array) to select only valid entries; the sequence dimension will shrink.
    valid_indices = ~masks_np
    true_valid = true_np[valid_indices]
    pred_valid = pred_np[valid_indices]
    
    # Compute the adjusted angle differences in the range [-180, 180)
    # angle_diff = (pred_valid - true_valid + 180) % 360 - 180
    angle_diff = (pred_valid - true_valid + np.pi) % (2 * np.pi) - np.pi
    absolute_errors = np.abs(angle_diff)
    
    # Compute Symmetric MAPE (SMAPE) using numpy
    denominator = (np.abs(true_valid) + np.abs(pred_valid)) / 2 + epsilon
    relative_error = absolute_errors / denominator
    smape = np.mean(relative_error)
    
    return float(smape)

def calculate_acc(predictions, true_labels, mask):
    """
        predictions (numpy.ndarray):  [b, s, |C|]
        true_labels (numpy.ndarray):  [b, s, 1]
        mask (numpy.ndarray): Mask  [b, s, 1]
        Returns:
            float: Macro accuracy computed after reducing the s dimension using mask.
    """
    valid_mask = (~mask).squeeze(-1)  # shape: [b, s]
    acc_scores = []
    b = predictions.shape[0]
    
    for i in range(b):
        # For each batch element, filter out the invalid time steps using valid_mask
        pred_i = predictions[i][valid_mask[i]]  # shape: [s_valid, |C|]
        true_i = true_labels[i][valid_mask[i]].squeeze(-1)  # shape: [s_valid]
        
        if pred_i.shape[0] == 0:
            continue
        
        # Compute predicted labels by taking the argmax over the class probabilities
        predicted_labels = np.argmax(pred_i, axis=1)  # shape: [s_valid]
        acc = np.mean(predicted_labels == true_i)
        acc_scores.append(acc)
    
    macro_acc = np.mean(acc_scores) if len(acc_scores) > 0 else 0.0
    return macro_acc

def calculate_acc_real(predictions, true_labels, mask):
    """
    Calculate macro accuracy for predictions and true labels.

    Args:
        predictions (numpy.ndarray): Predicted class indices, shape [b, s, 1].
        true_labels (numpy.ndarray): True class indices, shape [b, s, 1].
        mask (numpy.ndarray): Mask indicating valid positions, shape [b, s, 1].

    Returns:
        float: Macro accuracy computed after reducing the s dimension using mask.
    """
    # Convert mask to a boolean mask and remove the last dimension
    valid_mask = (~mask).squeeze(-1)  # shape: [b, s]
    
    acc_scores = []
    b = predictions.shape[0]  # Batch size
    
    for i in range(b):
        # For each batch element, filter out the invalid time steps using valid_mask
        pred_i = predictions[i][valid_mask[i]].squeeze(-1)  # shape: [s_valid]
        true_i = true_labels[i][valid_mask[i]].squeeze(-1)  # shape: [s_valid]
        
        if pred_i.shape[0] == 0:
            continue  # Skip if no valid time steps
        
        # Compute accuracy by comparing predicted labels with true labels
        acc = np.mean(pred_i == true_i)
        acc_scores.append(acc)
    
    # Compute macro accuracy across all batches
    macro_acc = np.mean(acc_scores) if len(acc_scores) > 0 else 0.0
    return macro_acc

def haversine_cpu_distance(lon1, lat1, lon2, lat2):
    """
    Calculate the spherical distance between two points using the Haversine formula,
    using NumPy operations so that the computation is done on the CPU.

    Args:
        lon1, lat1 (float or np.ndarray): Longitude and latitude of the first point.
        lon2, lat2 (float or np.ndarray): Longitude and latitude of the second point.

    Returns:
        np.ndarray or float: The spherical distance (in kilometers) computed on the CPU.
    """
    R = 6371.0  # Earth's radius in kilometers
     # Convert latitude and longitude from degrees to radians
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)  # Difference in latitudes
    delta_lambda = np.radians(lon2 - lon1)  # Difference in longitudes
    
    # Haversine formula
    a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return c  # Return the actual geographical distance in meters
 
def dtw_cpu_distance(trajectory1, trajectory2):
    n, m = len(trajectory1), len(trajectory2)  # Lengths of the two trajectories
    dtw_matrix = np.zeros((n, m))  # Initialize the DTW matrix
    
    # Fill the DTW matrix using dynamic programming
    for i in range(n):
        for j in range(m):
            # Compute the cost (distance) between the current pair of points
            cost = haversine_cpu_distance(trajectory1[i][0], trajectory1[i][1], trajectory2[j][0], trajectory2[j][1])
            
            # Boundary conditions for the first row and column
            if i == 0 and j == 0:
                dtw_matrix[i, j] = cost  # Base case: top-left corner
            elif i == 0:
                dtw_matrix[i, j] = cost + dtw_matrix[i, j - 1]  # First row
            elif j == 0:
                dtw_matrix[i, j] = cost + dtw_matrix[i - 1, j]  # First column
            else:
                # General case: take the minimum of three possible paths
                dtw_matrix[i, j] = cost + min(
                    dtw_matrix[i - 1, j],      # From the left
                    dtw_matrix[i, j - 1],      # From above
                    dtw_matrix[i - 1, j - 1]   # Diagonal
                )
    
    # The final DTW distance is the value in the bottom-right corner of the matrix
    return dtw_matrix[-1, -1]

def batch_cpu_dtw(true_trajectories, predicted_trajectories, masks):
    true_trajectories = true_trajectories.detach().cpu().numpy()
    predicted_trajectories = predicted_trajectories.detach().cpu().numpy()
    masks = masks.detach().cpu().numpy()
    if not isinstance(true_trajectories, np.ndarray):
        true_trajectories = np.array(true_trajectories, dtype=np.float32)
    if not isinstance(predicted_trajectories, np.ndarray):
        predicted_trajectories = np.array(predicted_trajectories, dtype=np.float32)
    if not isinstance(masks, np.ndarray):
        masks = np.array(masks)

    b, _, _ = true_trajectories.shape
    dtw_distances = []
    for i in range(b):
        mask = np.squeeze(masks[i], axis=-1)
        true_traj = true_trajectories[i][mask]   # shape [s_valid, 2]
        pred_traj = predicted_trajectories[i][mask]  # shape [s_valid, 2] 
        # true_traj = true_trajectories[i][~(np.isclose(true_trajectories[i][:, 0], 181.0) & 
        #                                  np.isclose(true_trajectories[i][:, 1], 91.0))]  # shape [s_valid, 2]
        # pred_traj = predicted_trajectories[i][~(np.isclose(true_trajectories[i][:, 0], 181.0) & 
        #                                       np.isclose(true_trajectories[i][:, 1], 91.0))]  # shape [s_valid, 2]
        dtw_dist = dtw_cpu_distance(true_traj, pred_traj)
        dtw_distances.append(dtw_dist)

    mean_dtw = np.mean(dtw_distances)
    return mean_dtw