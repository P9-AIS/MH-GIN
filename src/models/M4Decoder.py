import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedFusion(nn.Module):
    def __init__(self, feature_dim):
        super(GatedFusion, self).__init__()
        self.W_g = nn.Linear(5 * feature_dim, 5)  #  W_g
        self.W_d = nn.ModuleList([nn.Linear(feature_dim, feature_dim) for _ in range(5)])  #  W_d^l

    def forward(self, x):
        b_s, n, num_scales, feature_dim = x.shape

        # Step 1: Reshape and Concatenate across time scales (Concat)
        x_reshaped = x.view(b_s * n, num_scales, feature_dim)  # [b*s*n, 5, f]
        concat_features = x_reshaped.view(b_s * n, -1)  # [b*s*n, 5*f]

        # Step 2: Compute gating weights (g)
        g = torch.sigmoid(self.W_g(concat_features))  # [b*s*n, 5]
        g = g.unsqueeze(-1)  # [b*s*n, 5, 1]

        # Step 3: Apply scale-specific transformations (W_d^l h^{*,l})
        transformed_features = []
        for l in range(num_scales):
            transformed = self.W_d[l](x_reshaped[:, l, :])  # [b*s*n, f]
            transformed_features.append(transformed.unsqueeze(1))  # [b*s*n, 1, f]
        transformed_features = torch.cat(transformed_features, dim=1)  # [b*s*n, 5, f]

        # Step 4: Element-wise multiplication and summation (sum)
        gated_output = g * transformed_features  # [b*s*n, 5, f]
        output = torch.sum(gated_output, dim=1)  # [b*s*n, f]

        # Step 5: Reshape back to [b*s, n, f]
        output = output.view(b_s, n, feature_dim)  # [b*s, n, f]

        return output

class GatedFusionTwo(nn.Module):
    def __init__(self, feature_dim):
        super(GatedFusionTwo, self).__init__()
        # 5*feature_dim，feature_dim
        self.W_d = nn.ModuleList([nn.Linear(5 * feature_dim, feature_dim, bias=True) for _ in range(14)])

    def forward(self, x):
        assert not torch.isnan(x).any(), "GatedFusionTwo-x contains NaN values"
        assert not torch.isinf(x).any(), "GatedFusionTwo-x contains Inf values"
        # : [b_s, n=14, num_scales=5, feature_dim]
        b_s, n, num_scales, f = x.shape
        
        outputs = []
        for l in range(n):
            # l: [b_s, num_scales, f]
            x_l = x[:, l, :, :]
            #  [b_s, 5*f]
            x_l_flat = x_l.reshape(b_s, -1)
            #  [b_s, f]
            transformed = self.W_d[l](x_l_flat)
            outputs.append(transformed.unsqueeze(1))  #  [b_s, 1, f]
        
        # [b_s, n, f]
        output = torch.cat(outputs, dim=1)
        assert not torch.isnan(output).any(), "GatedFusionTwo-output contains NaN values"
        assert not torch.isinf(output).any(), "GatedFusionTwo-output contains Inf values"
        return output

class SpatioDecoder(nn.Module):
    def __init__(self, feature_dim):
        super(SpatioDecoder, self).__init__()
        self.W_lambda = nn.Linear(feature_dim * 2, 1, bias=True)
        self.W_phi = nn.Linear(feature_dim * 2, 1, bias=True)
        self.gamma_lambda = nn.Parameter(torch.tensor(0.1))
        self.gamma_phi = nn.Parameter(torch.tensor(0.1))
        
    def calculate_sliding_window_base(self, values, missing_value, window_size=5):
        batch_size, seq_len = values.shape
        device = values.device
        
        mask = (values != missing_value).float()  # [batch_size, seq_len]
        masked_values = values * mask
        
        window_sums = torch.zeros(batch_size, seq_len, device=device)
        window_counts = torch.zeros(batch_size, seq_len, device=device)
        
        pad_size = window_size
        padded_values = F.pad(masked_values, (pad_size, pad_size), "constant", 0)
        padded_mask = F.pad(mask, (pad_size, pad_size), "constant", 0)
        
        for i in range(2 * window_size + 1):
            window_sums += padded_values[:, i:i+seq_len]
            window_counts += padded_mask[:, i:i+seq_len]
        
        window_counts = torch.clamp(window_counts, min=1.0)
        window_avg = window_sums / window_counts
        
        return torch.deg2rad(window_avg.reshape(-1))

    def forward(self, e_lambda, e_phi, raw_lambda, raw_phi):
        # Concatenate longitude and latitude embeddings
        concat_features = torch.cat([e_lambda.squeeze(1), e_phi.squeeze(1)], dim=-1)  # [b*s, 2f]
        delta_lambda = torch.tanh(self.W_lambda(concat_features)) * self.gamma_lambda
        delta_phi = torch.tanh(self.W_phi(concat_features)) * self.gamma_phi

        with torch.no_grad():
            lambda_base = self.calculate_sliding_window_base(raw_lambda, 181)
            phi_base = self.calculate_sliding_window_base(raw_phi, 91)
        
        lambda_pred = lambda_base.unsqueeze(1) + delta_lambda
        phi_pred = phi_base.unsqueeze(1) + delta_phi

        spatio_pred = torch.cat([lambda_pred, phi_pred], dim=1) 
        delta_coord = torch.cat([delta_lambda, delta_phi], dim=1)
        return spatio_pred, delta_coord

class NeuralPointProcess(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.intensity_net = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
            nn.Softplus()
        )
        self.base_rate = nn.Parameter(torch.tensor(0.1))

    def forward(self, e_tau):
        intensity = self.intensity_net(e_tau) + self.base_rate
        
        intensity = torch.clamp(intensity, min=1e-7)
        
        epsilon = 1e-7
        u = torch.rand_like(intensity) * (1 - 2*epsilon) + epsilon  
        
        predicted_delta = -torch.log(u) / intensity
        return predicted_delta.squeeze()

class TemporalDecoder(nn.Module):
    def __init__(self, feature_dim, max_delta=300.0):
        super().__init__()
        self.process = NeuralPointProcess(feature_dim)
        self.max_delta = max_delta
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_normal_(self.process.intensity_net[-2].weight, gain=0.1)
        nn.init.constant_(self.process.intensity_net[-2].bias, 0.5)

    def forward(self, e_tau):
        base_delta = self.process(e_tau.squeeze(1))
        return torch.clamp(base_delta, min=1e-7, max=self.max_delta)

class CyclicalDecoder(nn.Module):

    def __init__(self, feature_dim):
        """
        Initialize the Cyclical Decoder.

        Parameters:
        - feature_dim: Dimension of the input features (f).
        - period: Period (\tau) of the cyclical variable.
        """
        super(CyclicalDecoder, self).__init__()

        # MLP with custom activation to generate trigonometric components
        self.mlp_phi = nn.Sequential(
            nn.Linear(feature_dim, 2),
            nn.Tanh(),
        )

    def forward(self, e_xk):
       """
        Forward pass for cyclical decoding.
        Parameters:
        - e_xk: Input embedding, shape [b*s, 1, f].
        Returns:
        - hat_theta: Reconstructed angular value, shape [b*s, 2].
       """
       e_xk_flat = e_xk.squeeze(1) # [b*s, f]   
       h_theta = self.mlp_phi(e_xk_flat) # [b*s, 2]
       return h_theta

class ContinuousDecoder(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        # MLP to generate h_n ∈ R
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, 1, bias=False),
        )

    def forward(self, e_xk, mu, sigma, alpha=1.0, beta=0.0):
        h_n = self.mlp(e_xk.squeeze(1))  # [b*s, 1]
        
        sigma = torch.clamp(sigma, min=1e-6)  # Prevent negative/zero std deviation
        
        alpha_safe = torch.clamp(alpha, min=1e-6) + 1e-8
        normalized = (h_n - beta) / alpha_safe
        
        x_hat = mu + sigma * normalized
        
        return x_hat

class ContinuousDecoderTwo(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        # MLP to generate h_n ∈ R
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, e_xk, min_val, max_val):
        h_n = self.mlp(e_xk.squeeze(1))  # [b*s, 1]

        delta_range = max_val - min_val
        
        safe_denominator = torch.clamp(delta_range, min=1e-6)  # Prevent negative/zero std deviation
        
        x_hat = h_n * safe_denominator + min_val  # [b*s, n]
        
        x_hat = torch.clamp(x_hat, min=min_val, max=max_val)
        return x_hat

class DiscreteDecoder(nn.Module):
 
    def __init__(self, feature_dim, output_dim, num_classes, smoothing_factor=0.1):
        """
        Initialize the Discrete Decoder.

        Parameters:
        - feature_dim: Dimension of the input features (f).
        - num_classes: Number of classes (|C|).
        - class_prototypes: Predefined class prototypes (\mathbf{w}_i), shape [|C|, d].
        - smoothing_factor: Hierarchical label smoothing factor (\alpha).
        """
        super(DiscreteDecoder, self).__init__()

        # MLP to generate intermediate representation \mathbf{z}
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, output_dim),
        )

        # Learnable temperature parameter
        self.temperature = nn.Parameter(torch.ones(1), requires_grad=True)
        # nn.init.xavier_uniform_(self.temperature)

        # Class prototypes (\mathbf{w}_i)
        self.class_prototypes = nn.Linear(in_features=num_classes, out_features=output_dim, bias=False)  # [|C|, d]

        # Smoothing factor (\alpha)
        self.smoothing_factor = smoothing_factor

    def forward(self, e_xk):
        """
        Forward pass for discrete decoding.

        Parameters:
        - e_xk: Input embedding, shape [b*s, 1, f].

        Returns:
        - hat_y: Class probability distribution, shape [b*s, |C|].
        """
        # Step 1: Generate intermediate representation \mathbf{z}
        z = self.mlp(e_xk.squeeze(1))  # [b*s, d]

        # Step 2: Compute similarity scores with class prototypes
        logits = torch.matmul(z, self.class_prototypes.weight) / self.temperature  # [b*s, |C|]

        # Step 3: Apply softmax to compute class probabilities
        s = F.softmax(logits, dim=-1)  # [b*s, |C|]

        # # Step 4: Hierarchical label smoothing
        uniform_dist = torch.full_like(s, 1.0 / s.size(-1))  # Uniform distribution [|C|]
        hat_y = (1 - self.smoothing_factor) * s + self.smoothing_factor * uniform_dist  # Smoothed probabilities
        # hat_y = s
       
        assert not torch.isnan(hat_y).any(), "discrete_decoder-hat_y contains NaN values"
        assert not torch.isinf(hat_y).any(), "discrete_decoder-hat_y contains Inf values"

        # [b*s, |C|] #[b*s, d], [b*s, d]
        return hat_y, z

class HeterogeneousAttributeDecoder(nn.Module):
    def __init__(self, 
                   d_data_model,
                   data_status,
                   device,
                   data_embedding_dim,
                   use_continuous_decoder_two=False,
                   max_delta=300.0,
                 ):
        super(HeterogeneousAttributeDecoder, self).__init__()
       
        self.data_status = data_status
        self.device = device
        self.max_delta = max_delta

        self.g_fusion_decoder = GatedFusionTwo(d_data_model)
        self.spatio_decoder = SpatioDecoder(d_data_model)
        
        self.temporal_decoder = TemporalDecoder(d_data_model, max_delta)
        
        
        self.cog_cyclical_decoder = CyclicalDecoder(d_data_model)
        self.heading_cyclical_decoder = CyclicalDecoder(d_data_model)
        
        self.vessel_width_continuous_decoder = ContinuousDecoder(d_data_model)
        self.vessel_length_continuous_decoder = ContinuousDecoder(d_data_model)
        self.vessel_draught_continuous_decoder = ContinuousDecoder(d_data_model)
        self.sog_continuous_decoder = ContinuousDecoder(d_data_model)
        self.rot_continuous_decoder = ContinuousDecoder(d_data_model)

        self.use_continuous_decoder_two = use_continuous_decoder_two
        if use_continuous_decoder_two:
            self.vessel_width_continuous_decoder = ContinuousDecoderTwo(d_data_model)
            self.vessel_length_continuous_decoder = ContinuousDecoderTwo(d_data_model)
            self.vessel_draught_continuous_decoder = ContinuousDecoderTwo(d_data_model)
            self.sog_continuous_decoder = ContinuousDecoderTwo(d_data_model)
            self.rot_continuous_decoder = ContinuousDecoderTwo(d_data_model)
       
        self.vessel_type_discrete_decoder = DiscreteDecoder(d_data_model, data_embedding_dim, len(self.data_status['vessel_type_unique'])+1)
        self.destination_discrete_decoder = DiscreteDecoder(d_data_model, data_embedding_dim, len(self.data_status['destination_unique'])+1)
        self.cargo_type_discrete_decoder = DiscreteDecoder(d_data_model, data_embedding_dim, len(self.data_status['cargo_type_unique'])+1)
        self.navi_status_discrete_decoder = DiscreteDecoder(d_data_model, data_embedding_dim, len(self.data_status['navi_status_unique'])+1)


    def forward(self, input_list, raw_features, observed_data, continuous_ab_dict, b, s):
        
        # # input_list[ b_s, n, 5, feature_dim]  out [b*s, n, f]
        out = self.g_fusion_decoder(align_and_concatenate_tensors(input_list))
        assert not torch.isnan(out).any(), "data_decoder-out contains NaN values"
        assert not torch.isinf(out).any(), "data_decoder-out contains Inf values"
        
        # input: lon [b*s, 1, f]. lat [b*s, 1, f]. output [b*s, 3]
        p_spatio, delta_coord = self.spatio_decoder(out[:, -2:-1, :], out[:, -3:-2, :], observed_data[:, :, -2], observed_data[:, :, -3])
        assert not torch.isnan(p_spatio).any(), "data_decoder-p_spatio contains NaN values"
        assert not torch.isinf(p_spatio).any(), "data_decoder-p_spatio contains Inf values"

        # input: [b*s, 1, f] output [b*s, 8]
        time_tau = self.temporal_decoder(out[:, -1:, :])
        assert not torch.isnan(time_tau).any(), "data_decoder-time_tau contains NaN values"
        assert not torch.isinf(time_tau).any(), "data_decoder-time_tau contains Inf values"
       
        # input: [b*s, 1, f] output [b*s, 1] [b*s, 1]
        cog_hat = self.cog_cyclical_decoder(out[:, 9:10, :])
        heading_hat = self.heading_cyclical_decoder(out[:, 10:11, :])
        assert not torch.isnan(cog_hat).any(), "data_decoder-cog_hat contains NaN values"
        assert not torch.isinf(heading_hat).any(), "data_decoder-heading_hat contains Inf values"
       
        if not self.use_continuous_decoder_two:
            # input: [b*s, n, f], min_x: [b*s, n] max_x: [b*s, n]
            # 2, 3, 4, 8, 9
            width_mu = torch.tensor(self.data_status['vessel_width_mean'], device=self.device)
            width_sigma = torch.tensor(self.data_status['vessel_width_std'], device=self.device)
            length_mu = torch.tensor(self.data_status['vessel_length_mean'], device=self.device)
            length_sigma = torch.tensor(self.data_status['vessel_length_std'], device=self.device)
            draught_mu = torch.tensor(self.data_status['vessel_draught_mean'], device=self.device)
            draught_sigma = torch.tensor(self.data_status['vessel_draught_std'], device=self.device)
            sog_mu = torch.tensor(self.data_status['sog_mean'], device=self.device)
            sog_sigma = torch.tensor(self.data_status['sog_std'], device=self.device)
            rot_mu = torch.tensor(self.data_status['rot_mean'], device=self.device)
            rot_sigma = torch.tensor(self.data_status['rot_std'], device=self.device)
           
            width_mu_matrix = width_mu.unsqueeze(0).repeat(b*s, 1)  # [b*s, 1]
            width_sigma_matrix = width_sigma.unsqueeze(0).repeat(b*s, 1)  # [b*s, 1] 
            length_mu_matrix = length_mu.unsqueeze(0).repeat(b*s, 1)  # [b*s, 1]
            length_sigma_matrix = length_sigma.unsqueeze(0).repeat(b*s, 1)  # [b*s, 1]
            draught_mu_matrix = draught_mu.unsqueeze(0).repeat(b*s, 1)  # [b*s, 1]
            draught_sigma_matrix = draught_sigma.unsqueeze(0).repeat(b*s, 1)  # [b*s, 1] 
            sog_mu_matrix = sog_mu.unsqueeze(0).repeat(b*s, 1)  # [b*s, 1]
            sog_sigma_matrix = sog_sigma.unsqueeze(0).repeat(b*s, 1)  # [b*s, 1]
            rot_mu_matrix = rot_mu.unsqueeze(0).repeat(b*s, 1)  # [b*s, 1]
            rot_sigma_matrix = rot_sigma.unsqueeze(0).repeat(b*s, 1)  # [b*s, 1]

            sigma_matrix = torch.cat((width_sigma_matrix, length_sigma_matrix, draught_sigma_matrix, sog_sigma_matrix, rot_sigma_matrix), dim=1)
            sigma_matrix = sigma_matrix.view(b, s, -1)

            width_hat = self.vessel_width_continuous_decoder(out[:, 1:2, :], mu=width_mu_matrix, sigma=width_sigma_matrix, 
                                             alpha=continuous_ab_dict['width'][0], beta=continuous_ab_dict['width'][1])
            length_hat = self.vessel_length_continuous_decoder(out[:, 2:3, :], mu=length_mu_matrix, sigma=length_sigma_matrix, 
                                             alpha=continuous_ab_dict['length'][0], beta=continuous_ab_dict['length'][1])
            draught_hat = self.vessel_draught_continuous_decoder(out[:, 3:4, :], mu=draught_mu_matrix, sigma=draught_sigma_matrix, 
                                             alpha=continuous_ab_dict['draught'][0], beta=continuous_ab_dict['draught'][1])
            sog_hat = self.sog_continuous_decoder(out[:, 7:8, :], mu=sog_mu_matrix, sigma=sog_sigma_matrix, 
                                             alpha=continuous_ab_dict['sog'][0], beta=continuous_ab_dict['sog'][1])
            rot_hat = self.rot_continuous_decoder(out[:, 8:9, :], mu=rot_mu_matrix, sigma=rot_sigma_matrix, 
                                                alpha=continuous_ab_dict['rot'][0], beta=continuous_ab_dict['rot'][1])   
        else:
            # input: [b*s, n, f], min_x: [b*s, n] max_x: [b*s, n]
            # 2, 3, 4, 8, 9
            width_min = torch.tensor(self.data_status['vessel_width_min'], device=self.device)
            width_max = torch.tensor(self.data_status['vessel_width_max'], device=self.device)
            length_min = torch.tensor(self.data_status['vessel_length_min'], device=self.device)
            length_max = torch.tensor(self.data_status['vessel_length_max'], device=self.device)
            draught_min = torch.tensor(self.data_status['vessel_draught_min'], device=self.device)
            draught_max = torch.tensor(self.data_status['vessel_draught_max'], device=self.device)
            sog_min = torch.tensor(self.data_status['sog_min'], device=self.device)
            sog_max = torch.tensor(self.data_status['sog_max'], device=self.device)
            rot_min = torch.tensor(self.data_status['rot_min'], device=self.device)
            rot_max = torch.tensor(self.data_status['rot_max'], device=self.device)
            
            width_min_matrix = width_min.unsqueeze(0).repeat(b*s, 1)  # [b*s, 1]
            width_max_matrix = width_max.unsqueeze(0).repeat(b*s, 1)  # [b*s, 1] 
            length_min_matrix = length_min.unsqueeze(0).repeat(b*s, 1)  # [b*s, 1]
            length_max_matrix = length_max.unsqueeze(0).repeat(b*s, 1)  # [b*s, 1]
            draught_min_matrix = draught_min.unsqueeze(0).repeat(b*s, 1)  # [b*s, 1]
            draught_max_matrix = draught_max.unsqueeze(0).repeat(b*s, 1)  # [b*s, 1] 
            sog_min_matrix = sog_min.unsqueeze(0).repeat(b*s, 1)  # [b*s, 1]
            sog_max_matrix = sog_max.unsqueeze(0).repeat(b*s, 1)  # [b*s, 1] 
            rot_min_matrix = rot_min.unsqueeze(0).repeat(b*s, 1)  # [b*s, 1]
            rot_max_matrix = rot_max.unsqueeze(0).repeat(b*s, 1)  # [b*s, 1]

            width_hat = self.vessel_width_continuous_decoder(out[:, 1:2, :], min_val=width_min_matrix, max_val=width_max_matrix)
            length_hat = self.vessel_length_continuous_decoder(out[:, 2:3, :], min_val=length_min_matrix, max_val=length_max_matrix)
            draught_hat = self.vessel_draught_continuous_decoder(out[:, 3:4, :], min_val=draught_min_matrix, max_val=draught_max_matrix)
            sog_hat = self.sog_continuous_decoder(out[:, 7:8, :], min_val=sog_min_matrix, max_val=sog_max_matrix)
            rot_hat = self.rot_continuous_decoder(out[:, 8:9, :], min_val=rot_min_matrix, max_val=rot_max_matrix)
            
            sigma_matrix = None

        continuous_hat = torch.cat((width_hat, length_hat, draught_hat, sog_hat, rot_hat), dim=1)
        continuous_dict = {
            "continuous_hat": continuous_hat.view(b, s, -1),
            "sigma_matrix": sigma_matrix,
        }

        assert not torch.isnan(width_hat).any(), "data_decoder-width_hat contains NaN values"
        assert not torch.isinf(length_hat).any(), "data_decoder-length_hat contains Inf values"
        assert not torch.isnan(draught_hat).any(), "data_decoder-draught_hat contains NaN values"
        assert not torch.isinf(sog_hat).any(), "data_decoder-sog_hat contains Inf values"
        assert not torch.isnan(rot_hat).any(), "data_decoder-rot_hat contains NaN values"

        # [b*s, 1, f]. [b*s, |C|] [b*s, f]
        vessel_type_y, vessel_type_z = self.vessel_type_discrete_decoder(out[:, 0:1, :])
        destination_y, destination_z = self.destination_discrete_decoder(out[:, 4:5, :])
        cargo_type_y, cargo_type_z = self.cargo_type_discrete_decoder(out[:, 5:6, :])
        navi_status_y, navi_status_z = self.navi_status_discrete_decoder(out[:, 6:7, :])

        # Index mapping: 0:vessel_type, 4:destination, 5:cargo_type, 6:navi_status
        discrete_list = [
            {"x": raw_features[:, :, idx:idx+1, :].squeeze(2), "z": [], "y": []}
            for idx in [0, 4, 5, 6]
        ]
        # [b, s, f], [b, s, |C|], [b, s, f]
        discrete_list[0]["z"] = vessel_type_z.view(b, s, -1)
        discrete_list[0]["y"] = vessel_type_y.view(b, s, -1)
        discrete_list[1]["z"] = destination_z.view(b, s, -1)
        discrete_list[1]["y"] = destination_y.view(b, s, -1)
        discrete_list[2]["z"] = cargo_type_z.view(b, s, -1)
        discrete_list[2]["y"] = cargo_type_y.view(b, s, -1)
        discrete_list[3]["z"] = navi_status_z.view(b, s, -1)
        discrete_list[3]["y"] = navi_status_y.view(b, s, -1)

        return (p_spatio.view(b, s, -1), delta_coord.view(b, s, -1)), \
            time_tau.view(b, s, -1), \
            cog_hat.view(b, s, -1), \
            heading_hat.view(b, s, -1), \
            continuous_dict, \
            discrete_list
        
def align_and_concatenate_tensors(tensor_list, target_l=5):
    aligned_tensor_list = []

    # Step 1:
    for cus_tensor in tensor_list:
        b_s, n_i, l_i, f = cus_tensor.shape
        if l_i < target_l:
            padding_size = target_l - l_i
            padding = torch.zeros(b_s, n_i, padding_size, f, dtype=cus_tensor.dtype, device=cus_tensor.device)
            aligned_tensor = torch.cat([cus_tensor, padding], dim=2)
        else:
            aligned_tensor = cus_tensor[:, :, :target_l, :]
        aligned_tensor_list.append(aligned_tensor)

    # Step 2:
    concatenated_tensor = torch.cat(aligned_tensor_list, dim=1)  #

    return concatenated_tensor

if __name__ == "__main__":
    b_s, n, num_scales, feature_dim = 10, 4, 5, 64
    input_tensor = torch.randn(b_s, n, num_scales, feature_dim)  # [b*s, n, 5, f]
    model = GatedFusion(feature_dim)
    output_tensor = model(input_tensor)
    print("Input Tensor Shape:", input_tensor.shape)
    print("Output Tensor Shape:", output_tensor.shape)
