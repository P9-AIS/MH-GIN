import math
import numpy as np
from torch import Tensor, nn
from torch.nn import functional as F
from typing import Optional
import torch
import random
import os

_torch_activations_dict = {
    'elu': 'ELU',
    'leaky_relu': 'LeakyReLU',
    'prelu': 'PReLU',
    'relu': 'ReLU',
    'rrelu': 'RReLU',
    'selu': 'SELU',
    'celu': 'CELU',
    'gelu': 'GELU',
    'glu': 'GLU',
    'mish': 'Mish',
    'sigmoid': 'Sigmoid',
    'softplus': 'Softplus',
    'tanh': 'Tanh',
    'silu': 'SiLU',
    'swish': 'SiLU',
    'linear': 'Identity'
}

def _identity(x):
    return x

def get_functional_activation(activation: Optional[str] = None):
    if activation is None:
        return _identity
    activation = activation.lower()
    if activation == 'linear':
        return _identity
    if activation in ['tanh', 'sigmoid']:
        return getattr(torch, activation)
    if activation in _torch_activations_dict:
        return getattr(F, activation)
    raise ValueError(f"Activation '{activation}' not valid.")

def self_normalizing_activation(x: Tensor, r: float = 1.0):
    return r * F.normalize(x, p=2, dim=-1)

def seed_everything(seed: Optional[int] = None) -> int:
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min
    try:
        if seed is None:
            seed = os.environ.get("PL_GLOBAL_SEED")
        seed = int(seed)
    except (TypeError, ValueError):
        seed = _select_seed_randomly(min_seed_value, max_seed_value)

    if not (min_seed_value <= seed <= max_seed_value):
        seed = _select_seed_randomly(min_seed_value, max_seed_value)

    print(f"Global seed set to {seed}")
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed

def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def _select_seed_randomly(min_seed_value: int = 0, max_seed_value: int = 255) -> int:
    return random.randint(min_seed_value, max_seed_value)

##################################################################################

def format_mbr(mbr):
    return (min(mbr[0], mbr[2]),\
            min(mbr[1], mbr[3]),\
            max(mbr[0], mbr[2]),\
            max(mbr[1], mbr[3]))

def cartesian_to_lonlat(p: torch.Tensor):
    """
    Convert 3D Cartesian coordinates to geographic coordinates (longitude, latitude) in degrees.

    Given a tensor p of shape [b, s, 3], where the last dimension represents (p_x, p_y, p_z),
    the conversion is defined as:

        λ̂ = arctan2(p_y, p_x)
        φ̂ = arcsin(p_z / ||p||)

    The results (longitude and latitude) are computed in radians and then converted to degrees.

    Args:
        p (torch.Tensor): Input tensor with shape [b, s, 3].

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (lon, lat) each of shape [b, s] in degrees.
    """
    # Extract components.
    px = p[..., 0]
    py = p[..., 1]
    pz = p[..., 2]

    # Compute the norm of vector p.
    norm_p = torch.norm(p, p=2, dim=-1)  # Shape: [b, s]

    # Compute longitude in radians using arctan2.
    lambda_hat = torch.atan2(py, px) # [-pi, pi]
    lambda_hat = torch.clamp(lambda_hat, min=-math.pi, max=math.pi)

    # Calculate ratio with safe division and clamp to [-1, 1] for arcsin.
    ratio = pz / (norm_p + 1e-8)
    # ratio = torch.clamp(ratio, -1.0, 1.0)

    # Compute latitude in radians.
    phi_hat = torch.asin(ratio)
    phi_hat = torch.clamp(phi_hat, min=-math.pi/2 , max=math.pi/2)

    # Convert radians to degrees.
    lon = lambda_hat * (180.0 / math.pi)
    lat = phi_hat * (180.0 / math.pi)

    return lon, lat

def restore_timestamp(t_base, encoded, T:torch.tensor):
    """
    Restore the precise timestamp from multi-frequency sinusoidal encoding.

    This function recovers the exact timestamp using multi-scale phase synchronization.
    The encoding tensor should have shape [b, s, 8] with channels arranged as follows:
      [ sin(2πτ/T₁), cos(2πτ/T₁),
        sin(2πτ/T₂), cos(2πτ/T₂),
        sin(2πτ/T₃), cos(2πτ/T₃),
        sin(2πτ/T₄), cos(2πτ/T₄) ]

    The base timestamp (t_base) is given as seconds, and for example, for the date
    "2024-01-01 00:00:00", t_base must be the Unix timestamp corresponding to that moment.

    The function computes the phase for the finest frequency (T₄) as:
       φ₄ = arctan2(sin_component, cos_component)
    and then restores the timestamp as:
       τ = t_base + (T₄/(2π))*(φ₄ + 2π·n_offset)
    where n_offset is an integer used to unwrap the phase if needed.

    Args:
        t_base (float or torch.Tensor): The base timestamp in seconds, e.g., the timestamp
            for "2024-01-01 00:00:00".
        encoded (torch.Tensor): The sinusoidal encoding tensor with shape [b, s, 8].
        T (list or tuple): A list of periods in seconds, for example:
            [24*3600, 168*3600, 720*3600, 8760*3600] representing day, week, month, and year.
        n_offset (int, optional): The integer number of extra periods for phase unwrapping at T₄.
            Default is 0.

    Returns:
        torch.Tensor: The restored precise timestamp τ with shape [b, s].
    """
    # Extract the sine and cosine components corresponding to T₄.
    # Here, we assume that the last pair, at indices 6 and 7 (0-indexed), corresponds to T₄.
    size = T.size(0)
    sin_phi4 = encoded[..., 2*size-2].squeeze(-1)  # Shape becomes [b, s]
    cos_phi4 = encoded[..., 2*size-1].squeeze(-1)  # Shape becomes [b, s]

    # Compute the phase φ₄ using arctan2, which returns values in (-π, π].
    phi4 = torch.atan2(sin_phi4, cos_phi4)

    # Adjust the phase to be in the range [0, 2π).
    # phi4 = (phi4 + 2 * math.pi) % (2 * math.pi)

    # Retrieve T₄ from the provided period list.
    T4 = T[size-1]

    # Ensure t_base is a torch.Tensor.
    if not isinstance(t_base, torch.Tensor):
        t_base = torch.tensor(t_base, dtype=torch.float)

    # Compute the restored timestamp τ using the formula:
    # τ = t_base + (T₄ / (2π)) * (φ₄ + 2π * n_offset)
    # tau = t_base + (float(T4) / (2 * math.pi)) * (phi4 + 2 * math.pi * float(n_offset))
    # tau = t_base + (float(T4) / (2 * math.pi)) * (phi4 % (2 * math.pi) + 2 * math.pi * float(n_offset))
    tau = int(t_base/(8760*3600)) * (8760*3600) + (T4 * (phi4 + math.pi)) / (2 * math.pi) 
    # tau = tau.int()
    return tau


def get_real_angle(hat):
    """
    Decode angular values from sine and cosine components, returning the real angle in degrees within [0, 360).

    Args:
        hat_c (torch.Tensor): Cosine component, shape [b, s, 2].
        hat_s (torch.Tensor): Sine component, shape [b, s, 2].
        tau (float): Period of the angular values (not used in degree conversion).

    Returns:
        torch.Tensor: Decoded angular values in degrees, shape [b, s, 2].
    """
    # Compute raw angular values in radians using arctan2 (range: -π to π)
    theta = torch.atan2(hat[:,:,0:1], hat[:,:,1:2])
    # Convert radians to degrees to obtain angles in the range [0, 360)
    angle_degrees = (theta + torch.pi) * 180.0 / torch.pi
    return angle_degrees

def geo_to_cartesian(lon, lat, radius):
    """
    Converts geographical coordinates (longitude, latitude) of shape [b, s, 1] to 3D Cartesian coordinates.

    Parameters:
    lon (torch.Tensor): Longitude in degrees, shape [b, s, 1]
    lat (torch.Tensor): Latitude in degrees, shape [b, s, 1]
    radius (float): Earth's radius, default is 6371 km

    Returns:
    torch.Tensor: 3D Cartesian coordinates of shape [b, s, 3]
    """
    # Convert angles from degrees to radians
    lambda_rad = torch.deg2rad(lon)
    phi_rad = torch.deg2rad(lat)

    # Calculate Cartesian coordinates
    x = radius * torch.cos(lambda_rad) * torch.cos(phi_rad)
    y = radius * torch.sin(lambda_rad) * torch.cos(phi_rad)
    z = radius * torch.sin(phi_rad)

    # Concatenate into a tensor of shape [b, s, 3]
    return  torch.cat((x, y, z), dim=-1)

def get_time_sin_cos(t, T):
    """
    Generate time encoding features using sine and cosine functions.

    For each period in T, the function computes:
        sin(2π * t / T) and cos(2π * t / T),
    and returns a tensor with shape [b, s, len(T), 2], where the last dimension
    represents (sin, cos) for a given period.

    Args:
        t (torch.Tensor): A tensor of shape [b, s, 1] representing timestamps in seconds.
        T (torch.Tensor): A 1D tensor containing period values, e.g., T = torch.tensor([24, 168, 720, 8760]).float()

    Returns:
        torch.Tensor: A tensor of shape [b, s, len(T), 2] containing the time encoding features.
    """
    # Remove the last dimension: from [b, s, 1] to [b, s]
    t = t.squeeze(-1)
    b, s = t.size()

    # T = T.to(t.device)

    # Compute sine and cosine components.
    # t.unsqueeze(-1) expands t from [b, s] to [b, s, 1], which broadcasts with T of shape [len(T)]
    sin_component = torch.sin(2 * math.pi * t.unsqueeze(-1) / T - math.pi)  # shape: [b, s, len(T)]
    cos_component = torch.cos(2 * math.pi * t.unsqueeze(-1) / T - math.pi)  # shape: [b, s, len(T)]

    # Stack sin and cos along a new last dimension.
    # Final output shape: [b, s, len(T), 2], i.e., [(sin, cos), ...] for each period.
    e_tau = torch.stack([sin_component, cos_component], dim=-1)

    return e_tau.view(b, s, T.size(0)*2)

def get_time_interval(t, max_delta):
    """
    Calculate the time interval between consecutive timestamps.

    Args:
        t (torch.Tensor): A tensor of shape [b, s, 1] representing timestamps in seconds.
    
    """
    t = t.squeeze(-1)
    diffs = t[:, 1:] - t[:, :-1]
    diffs = torch.cat([torch.zeros_like(t[:, :1]), diffs], dim=1).unsqueeze(-1)
    diffs = torch.clamp(diffs, min=0, max=max_delta)
    return diffs

def angle_sin_cos(x, tau):
    scaled = (2 * math.pi * x) / tau
    sin_part = torch.sin(scaled)
    cos_part = torch.cos(scaled)
    return torch.cat([sin_part, cos_part], dim=-1)


def log_metrics(logger, final_metrics, coordinate_is_mae_smape=False, mean_test=False):
        attributes_dict = {
            'Spatial': ['coord_distance'],
            'Temporal': ['time_mae', 'time_smape'],
            'Angles': ['cog_mae', 'cog_smape', 'heading_mae', 'heading_smape'],
            'Continuous': ['width_mae', 'width_smape', 'length_mae', 'length_smape', 'draught_mae', 'draught_smape', 'sog_mae', 'sog_smape'],
            'Discrete': ['vessel_type_acc', 'cargo_type_acc', 'navi_status_acc']
        }
        if coordinate_is_mae_smape:
            attributes_dict['Spatial'] = ['coord_mae', 'coord_smape']
        if mean_test:
            attributes_dict['Continuous']= ['draught_mae', 'draught_smape', 'sog_mae', 'sog_smape']
            attributes_dict['Discrete']= ['cargo_type_acc', 'navi_status_acc']
        logger.info("="*60)
        for category, metrics in attributes_dict.items():
            logger.info(f"{'-'*15} {category} Metrics {'-'*15}")
            for metric in metrics:
                logger.info(f"{metric}: {final_metrics[metric]:.8f}")
        logger.info("="*60)

def log_loss_components(logger, *loss_components):
    """Log loss components with formatted output"""
    names = ["spatial", "temporal", "angular", "continuous", "discrete", "total"]
    components = dict(zip(names, loss_components))
    
    logger.info(
        f"Loss Components - "
        f"Spatial: {components['spatial']:.4f}, "
        f"Temporal: {components['temporal']:.4f}, "
        f"Angular: {components['angular']:.4f}"
    )
    logger.info(
        f"Continuous: {components['continuous']:.4f}, "
        f"Discrete: {components['discrete']:.4f}, "
        f"Total: {components['total']:.4f}"
    )



if __name__ == '__main__':
    # Example usage
    b = 2  # Batch size
    s = 3  # Sequence length

    # Longitudes and latitudes for Beijing, Shanghai, Shenzhen, New York, London, Paris
    longitude = torch.tensor([[[116.4074], [121.4737], [114.0683]],  # Beijing, Shanghai, Shenzhen
                              [[-74.0060], [-0.1278], [2.3522]]])  # New York, London, Paris
    latitude = torch.tensor([[[39.9042], [31.2304], [22.5431]],  # Beijing, Shanghai, Shenzhen
                             [[40.7128], [51.5074], [48.8566]]])  # New York, London, Paris

    # Convert to Cartesian coordinates
    cartesian_coords = geo_to_cartesian(longitude, latitude, radius=6371)

    print(f"3D Cartesian Coordinates:\n{cartesian_coords}")