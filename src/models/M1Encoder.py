import torch
import torch.nn as nn
import math
from src.utils.utils import get_time_interval

class CoordinateEncoder(nn.Module):
    """
    Spatial Coordinate Encoder using hybrid spherical-harmonic encoding.
    Inputs: lon (longitude) and lat (latitude) with shape [b, s, 1].
    Outputs: Encoded features with shape [b, s, 2, f], where f = d_model.
    """
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        
       # Learnable affine transformation parameters for each dimension
        self.linear_lon = nn.Linear(9, d_model)  # For longitude
        self.linear_lat = nn.Linear(9, d_model)  # For latitude
        self.activation = nn.Tanh()  # Hyperbolic tangent activation function
        
        # Initialize weights and biases using Xavier initialization
        nn.init.xavier_uniform_(self.linear_lon.weight)
        nn.init.zeros_(self.linear_lon.bias)
        nn.init.xavier_uniform_(self.linear_lat.weight)
        nn.init.zeros_(self.linear_lat.bias)

    def forward(self, lon, lat):
        """
        Args:
            lon (torch.Tensor): Longitude tensor of shape [b, s, 1] (in degrees).
            lat (torch.Tensor): Latitude tensor of shape [b, s, 1] (in degrees).
        Returns:
            torch.Tensor: Encoded features of shape [b, s, 2, f], where f = d_model.
        """
        lon_missing_mask = torch.isclose(lon, torch.tensor(181.0, dtype=torch.float32), atol=1e-5)  # [b, s, 1]
        lat_missing_mask = torch.isclose(lat, torch.tensor(91.0, dtype=torch.float32), atol=1e-5)  # [b, s, 1]
        # Convert degrees to radians
        lon_rad = torch.deg2rad(lon)  # Shape [b, s, 1]
        lat_rad = torch.deg2rad(lat)  # Shape [b, s, 1]

        # Compute the hybrid spherical-harmonic feature vector
        sin_lon = torch.sin(lon_rad)
        cos_lon = torch.cos(lon_rad)
        sin_lat = torch.sin(lat_rad)
        cos_lat = torch.cos(lat_rad)

        # Feature vector components
        f1 = sin_lon * cos_lat  # sin(λ)cos(φ)
        f2 = cos_lon * cos_lat  # cos(λ)cos(φ)
        f3 = sin_lat            # sin(φ)
        f4 = torch.sin(2 * lon_rad) * cos_lat  # sin(2λ)cos(φ)
        f5 = torch.cos(2 * lon_rad) * cos_lat  # cos(2λ)cos(φ)

        # Combine into a single feature vector [b, s, 5]
        features = torch.cat([sin_lon, cos_lon, sin_lat, cos_lat, f1, f2, f3, f4, f5], dim=-1)  # Shape [b, s, 5]

        # Apply learnable affine transformations and activation for each dimension
        encoded_lon = self.activation(self.linear_lon(features)) # Shape [b, s, d_model]
        encoded_lat = self.activation(self.linear_lat(features))  # Shape [b, s, d_model]
        encoded_lon = encoded_lon.masked_fill(lon_missing_mask.expand_as(encoded_lon), 0.0)
        encoded_lat = encoded_lat.masked_fill(lat_missing_mask.expand_as(encoded_lat), 0.0)

        # Stack the encoded features along the feature dimension
        output = torch.stack([encoded_lat, encoded_lon], dim=2)  # Shape [b, s, 2, d_model]

        return output

class TimeEncoder(nn.Module):
    """Temporal Feature Encoder (Multi-Frequency Sine Encoding)"""

    def __init__(self, d_model, max_delta, frequencies=None):
        super().__init__()
        self.max_delta = max_delta
        if frequencies is None:
            # frequencies = [60, 3600, 24*3600, 168*3600, 720*3600, 8760*3600]
            frequencies = [60, 3600, 24*3600]
        self.frequencies = frequencies
        
        self.interval_encoder = nn.Sequential(
            nn.Linear(1, d_model), 
            nn.ReLU()
        )
        
        self.mlp = nn.Sequential(
            nn.Linear(3 * len(frequencies) + d_model, d_model),
            nn.Tanh(),
        )

    def forward(self, t):
        t_interval = get_time_interval(t, self.max_delta)  # [b, s, 1]
        missing_mask = torch.isclose(t, torch.tensor(-1.0, dtype=torch.float32), atol=1e-5)
        t_interval = t_interval.masked_fill(missing_mask, 0.0)
        
        time_features = []
        for freq in self.frequencies:
            sin = torch.sin(2 * math.pi * t / freq - math.pi)
            cos = torch.cos(2 * math.pi * t / freq - math.pi)
            time_features.extend([sin, cos, torch.log(t_interval + 1e-6)])  
            
        interval_encoded = self.interval_encoder(torch.log(t_interval + 1e-6))  # [b, s, d_model//2]
        
        encoded_features = torch.cat([
            torch.cat(time_features, dim=-1), 
            interval_encoded
        ], dim=-1)
        
        x = self.mlp(encoded_features).unsqueeze(2)
        x = x.masked_fill(missing_mask.unsqueeze(-1), 0.0)
        return x

class CyclicalEncoder(nn.Module):
    """
    Cyclical Encoder for attributes with physical periodicity (e.g., heading angle or course over ground).
    Implements the formula:
        e_p = phi(W_p * [sin(pi * x_p / 180), cos(pi * x_p / 180)]^T + b_p)
    where phi(x) = x + sin^2(x) introduces explicit periodic inductive bias.
    """
    def __init__(self, d_model):
        """
        Initialize the encoder.

        Args:
            d_model (int): Dimensionality of the output embedding.
        """
        super().__init__()
        # Linear layer to map trigonometric features to d_model dimensions
        self.linear = nn.Linear(2, d_model)
        # Activation function: phi(x) = x + sin^2(x)
        # self.activation = lambda x: x + torch.sin(x) ** 2

    def forward(self, x):
        mask = (x == -1)
        
        # Step 1: Convert input angles from degrees to radians
        x_rad = torch.deg2rad(x) - torch.pi  # Shape [b, s, 1]

        # Step 2: Compute sine and cosine features
        sin_features = torch.sin(x_rad)  # Shape [b, s, 1]
        cos_features = torch.cos(x_rad)  # Shape [b, s, 1]

        # Step 3: Combine sine and cosine features into a single vector [b, s, 2]
        trig_features = torch.cat([sin_features, cos_features], dim=-1)  # Shape [b, s, 2]

        # Step 4: Apply linear transformation and activation
        x_encoded = self.linear(trig_features)  # Shape [b, s, d_model]
        # x_encoded = self.activation(x_encoded)  # Shape [b, s, d_model]

        x_encoded = x_encoded.masked_fill(mask.expand_as(x_encoded), 0.0)
        
        # Step 5: Reshape to [b, s, 1, f]
        output = x_encoded.unsqueeze(2)  # Shape [b, s, 1, f]

        return output

class ContinuousEncoder(nn.Module):
    def __init__(self, d_model, mu, sigma):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, d_model),
        )
        self.alpha = nn.Parameter(torch.ones(1), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x, mu, sigma):
        missing_mask = torch.eq(x, -1)  # [b, s, 1]
        
        denominator = torch.clamp(sigma, min=1e-6)
        x_norm = (x - mu) / denominator
        
        x_projected = x_norm * self.alpha + self.beta
        batch_size, seq_len, _ = x_projected.shape
        
        x_encoded = self.mlp(x_projected.view(-1, 1))  # [b*s, d_model]
        x_encoded = x_encoded.view(batch_size, seq_len, 1, -1)  # [b, s, 1, d_model]
        
        x_encoded = x_encoded.masked_fill(missing_mask.unsqueeze(-1), 0.0)
        
        return x_encoded, self.alpha, self.beta

class ContinuousEncoderTwo(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.Tanh(),
        )

    def forward(self, x, min_val, max_val):
        missing_mask = torch.eq(x, -1)  # [b, s, 1]
        
        denominator = torch.clamp(max_val - min_val, min=1e-6)
        x_norm = (x - min_val) / denominator
        
        batch_size, seq_len, _ = x_norm.shape
        
        x_encoded = self.mlp(x_norm.view(-1, 1))  # [b*s, d_model]
        x_encoded = x_encoded.view(batch_size, seq_len, 1, -1)  # [b, s, 1, d_model]
        
        x_encoded = x_encoded.masked_fill(missing_mask.unsqueeze(-1), 0.0)
        
        return x_encoded

class DiscreteEncoder(nn.Module):
    """Discrete Feature Encoder with missing value handling"""

    def __init__(self, d_model, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(num_classes + 1, d_model, padding_idx=num_classes)
        
        with torch.no_grad():
            self.embedding.weight[num_classes].zero_()

    def forward(self, x):
        x = x.long().squeeze(-1)  # [b, s]
        out = self.embedding(x)  # [b, s, d_model]
        return out.unsqueeze(2)  # [b, s, 1, d_model]

class HeterogeneousAttributeEncoder(nn.Module):

    def __init__(self, d_model,
                 status,
                 navi_status_class_num,
                 destination_class_num,
                 cargo_type_class_num,
                 vessel_type_class_num,
                 max_delta,
                 continuous_two = False,
                 frequencies=None):
        super().__init__()
        self.data_status = status
        self.max_delta = max_delta
        self.coordinate_encoder = CoordinateEncoder(d_model)
        self.time_encoder = TimeEncoder(d_model, max_delta, frequencies)
        self.continuous_two = continuous_two
        
        self.cog_cyclical_encoder = CyclicalEncoder(d_model)
        self.heading_angle_cyclical_encoder = CyclicalEncoder(d_model)
        
        self.vessel_width_continuous_encoder = ContinuousEncoder(d_model, self.data_status['vessel_width_mean'], self.data_status['vessel_width_std'])
        self.vessel_length_continuous_encoder = ContinuousEncoder(d_model, self.data_status['vessel_length_mean'], self.data_status['vessel_length_std'])
        self.vessel_draught_continuous_encoder = ContinuousEncoder(d_model, self.data_status['vessel_draught_mean'], self.data_status['vessel_draught_std'])
        self.sog_continuous_encoder = ContinuousEncoder(d_model, self.data_status['sog_mean'], self.data_status['sog_std'])
        self.rot_continuous_encoder = ContinuousEncoder(d_model, self.data_status['rot_mean'], self.data_status['rot_std'])
        if continuous_two:
            self.vessel_width_continuous_encoder = ContinuousEncoderTwo(d_model)
            self.vessel_length_continuous_encoder = ContinuousEncoderTwo(d_model)
            self.vessel_draught_continuous_encoder = ContinuousEncoderTwo(d_model)
            self.sog_continuous_encoder = ContinuousEncoderTwo(d_model)
            self.rot_continuous_encoder = ContinuousEncoderTwo(d_model)
            
        self.vessel_type_discrete_encoder = DiscreteEncoder(d_model, num_classes=vessel_type_class_num)
        self.destination_discrete_encoder = DiscreteEncoder(d_model, num_classes=destination_class_num)
        self.cargo_type_discrete_encoder = DiscreteEncoder(d_model, num_classes=cargo_type_class_num)
        self.navi_status_discrete_encoder = DiscreteEncoder(d_model, num_classes=navi_status_class_num)
        
    def forward(self, tensor_input):  # [b, s, n]
        b, s, _ = tensor_input.shape
        spatial_output = self.coordinate_encoder(tensor_input[:, :, 12:13], tensor_input[:, :, 11:12])  # longitude, latitude
        time_output = self.time_encoder(tensor_input[:, :, 13:14])  # timestamp
        
        cog_output = self.cog_cyclical_encoder(tensor_input[:, :, 9:10])  # cog
        heading_angle_output = self.heading_angle_cyclical_encoder(tensor_input[:, :, 10:11])  # heading_angle
        
        vessel_type_output = self.vessel_type_discrete_encoder(tensor_input[:, :, 0:1])
        destination_output = self.destination_discrete_encoder(tensor_input[:, :, 4:5])
        cargo_type_output = self.cargo_type_discrete_encoder(tensor_input[:, :, 5:6]) 
        navi_status_output = self.navi_status_discrete_encoder(tensor_input[:, :, 6:7])
        
        if not self.continuous_two:
            width_mu = torch.tensor(self.data_status['vessel_width_mean'], device=tensor_input.device)
            width_sigma = torch.tensor(self.data_status['vessel_width_std'], device=tensor_input.device)
            length_mu = torch.tensor(self.data_status['vessel_length_mean'], device=tensor_input.device)
            length_sigma = torch.tensor(self.data_status['vessel_length_std'], device=tensor_input.device)
            draught_mu = torch.tensor(self.data_status['vessel_draught_mean'], device=tensor_input.device)
            draught_sigma = torch.tensor(self.data_status['vessel_draught_std'], device=tensor_input.device)
            sog_mu = torch.tensor(self.data_status['sog_mean'], device=tensor_input.device)
            sog_sigma = torch.tensor(self.data_status['sog_std'], device=tensor_input.device)
            rot_mu = torch.tensor(self.data_status['rot_mean'], device=tensor_input.device)
            rot_sigma = torch.tensor(self.data_status['rot_std'], device=tensor_input.device)
            
            width_mu_matrix = width_mu.unsqueeze(0).repeat(b*s, 1).view(b, s, -1)  # [b, s, 1]
            width_sigma_matrix = width_sigma.unsqueeze(0).repeat(b*s, 1).view(b, s, -1)  # [b, s, 1] 
            length_mu_matrix = length_mu.unsqueeze(0).repeat(b*s, 1).view(b, s, -1)  # [b, s, 1]
            length_sigma_matrix = length_sigma.unsqueeze(0).repeat(b*s, 1).view(b, s, -1)  # [b, s, 1]
            draught_mu_matrix = draught_mu.unsqueeze(0).repeat(b*s, 1).view(b, s, -1)  # [b, s, 1]
            draught_sigma_matrix = draught_sigma.unsqueeze(0).repeat(b*s, 1).view(b, s, -1)  # [b, s, 1]
            sog_mu_matrix = sog_mu.unsqueeze(0).repeat(b*s, 1).view(b, s, -1)  # [b, s, 1]
            sog_sigma_matrix = sog_sigma.unsqueeze(0).repeat(b*s, 1).view(b, s, -1)  # [b, s, 1]
            rot_mu_matrix = rot_mu.unsqueeze(0).repeat(b*s, 1).view(b, s, -1)  # [b, s, 1]
            rot_sigma_matrix = rot_sigma.unsqueeze(0).repeat(b*s, 1).view(b, s, -1)  # [b, s, 1]
            
                
            width_output, width_alpha, width_beta = self.vessel_width_continuous_encoder(tensor_input[:, :, 1:2],
                                                                                        mu=width_mu_matrix,
                                                                                        sigma=width_sigma_matrix)
            length_output, length_alpha, length_beta = self.vessel_length_continuous_encoder(tensor_input[:, :, 2:3],
                                                                                            mu=length_mu_matrix,
                                                                                            sigma=length_sigma_matrix)
            draught_output, draught_alpha, draught_beta = self.vessel_draught_continuous_encoder(tensor_input[:, :, 3:4],
                                                                                                mu=draught_mu_matrix,
                                                                                                sigma=draught_sigma_matrix)
            sog_output, sog_alpha, sog_beta = self.sog_continuous_encoder(tensor_input[:, :, 7:8],
                                                                        mu=sog_mu_matrix, sigma=sog_sigma_matrix)
            rot_output, rot_alpha, rot_beta = self.rot_continuous_encoder(tensor_input[:, :, 8:9],
                                                                        mu=rot_mu_matrix, sigma=rot_sigma_matrix)
            continuous_ab_dict = {
                'rot': (rot_alpha, rot_beta),
                'sog': (sog_alpha, sog_beta),
                'draught': (draught_alpha, draught_beta),
                'length': (length_alpha, length_beta),
                'width': (width_alpha, width_beta)
            }
        else:
            width_min = torch.tensor(self.data_status['vessel_width_min'], device=tensor_input.device)
            width_max = torch.tensor(self.data_status['vessel_width_max'], device=tensor_input.device)
            length_min = torch.tensor(self.data_status['vessel_length_min'], device=tensor_input.device)
            length_max = torch.tensor(self.data_status['vessel_length_max'], device=tensor_input.device)
            draught_min = torch.tensor(self.data_status['vessel_draught_min'], device=tensor_input.device)
            draught_max = torch.tensor(self.data_status['vessel_draught_max'], device=tensor_input.device)
            sog_min = torch.tensor(self.data_status['sog_min'], device=tensor_input.device)
            sog_max = torch.tensor(self.data_status['sog_max'], device=tensor_input.device)
            rot_min = torch.tensor(self.data_status['rot_min'], device=tensor_input.device)
            rot_max = torch.tensor(self.data_status['rot_max'], device=tensor_input.device)
            
            width_min_matrix = width_min.unsqueeze(0).repeat(b*s, 1).view(b, s, -1)  # [b, s, 1]
            width_max_matrix = width_max.unsqueeze(0).repeat(b*s, 1).view(b, s, -1)  # [b, s, 1] 
            length_min_matrix = length_min.unsqueeze(0).repeat(b*s, 1).view(b, s, -1)  # [b, s, 1]
            length_max_matrix = length_max.unsqueeze(0).repeat(b*s, 1).view(b, s, -1)  # [b, s, 1]
            draught_min_matrix = draught_min.unsqueeze(0).repeat(b*s, 1).view(b, s, -1)  # [b, s, 1]
            draught_max_matrix = draught_max.unsqueeze(0).repeat(b*s, 1).view(b, s, -1)  # [b, s, 1]
            sog_min_matrix = sog_min.unsqueeze(0).repeat(b*s, 1).view(b, s, -1)  # [b, s, 1]
            sog_max_matrix = sog_max.unsqueeze(0).repeat(b*s, 1).view(b, s, -1)  # [b, s, 1]
            rot_min_matrix = rot_min.unsqueeze(0).repeat(b*s, 1).view(b, s, -1)  # [b, s, 1]
            rot_max_matrix = rot_max.unsqueeze(0).repeat(b*s, 1).view(b, s, -1)  # [b, s, 1]
            
                
            width_output = self.vessel_width_continuous_encoder(tensor_input[:, :, 1:2],
                                                                min_val=width_min_matrix,
                                                                max_val=width_max_matrix)
            length_output = self.vessel_length_continuous_encoder(tensor_input[:, :, 2:3],
                                                                 min_val=length_min_matrix,
                                                                 max_val=length_max_matrix)
            draught_output = self.vessel_draught_continuous_encoder(tensor_input[:, :, 3:4],
                                                                   min_val=draught_min_matrix,
                                                                   max_val=draught_max_matrix)
            sog_output = self.sog_continuous_encoder(tensor_input[:, :, 7:8],
                                                    min_val=sog_min_matrix,
                                                    max_val=sog_max_matrix)
            rot_output = self.rot_continuous_encoder(tensor_input[:, :, 8:9],
                                                    min_val=rot_min_matrix,
                                                    max_val=rot_max_matrix)    
            continuous_ab_dict = {}

        output = torch.cat((vessel_type_output, width_output, length_output, draught_output,
                        destination_output, cargo_type_output, navi_status_output,
                        sog_output, rot_output, cog_output, heading_angle_output,
                        spatial_output, time_output),
                        dim=2)
           
        return output, continuous_ab_dict

