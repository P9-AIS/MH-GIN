import torch
import torch.nn as nn

from src.utils.loss_util import compute_spatio_loss, compute_angle_cos_loss, compute_discrete_loss, compute_time_interval_loss, compute_continuous_mse_loss, calculate_angle_mse_loss, haversine_distance 
from src.models.M4Decoder import HeterogeneousAttributeDecoder
from src.models.M3GraphPropagation import CrossScaleDependencyMining
from src.models.M2MutipleScaleMining import ESNModel, AttentionModel, AttentionScaleModel
from src.models.M1Encoder import HeterogeneousAttributeEncoder
from src.utils.metric_util import calculate_one_angle_mae, calculate_one_angle_smape, calculate_one_mae, calculate_one_smape, calculate_acc, batch_cpu_dtw
from src.utils.utils import geo_to_cartesian, get_time_interval, cartesian_to_lonlat, get_real_angle


class MtsHGnn(nn.Module):
    def __init__(self,
                 logger,
                 device,
                 dataset_name,
                 d_data_model,
                 data_status,
                 d_mining_model,
                 hidden_size,
                 output_size,
                 horizon,
                 num_layers=4,
                 graph_mask_values=[4, 8],
                 leaking_rate=0.9,
                 spectral_radius=0.9,
                 w_ih_scale=1,
                 density=0.9,
                 alpha_decay=False,
                 reservoir_activation='tanh',
                 bidirectional=True,
                 R_e=1.,
                 conti_huber_delta=1.0,
                 dis_in_lambda_reg=0.0,
                 max_delta=350.0,
                 M2_mode= "scale_attn",
                 use_graph_propagation=True,
                 window_size=3,
                 ):
        super(MtsHGnn, self).__init__()
        self.logger = logger
        self.data_status = data_status
        self.device = device
        self.dataset_name = dataset_name
        self.t_base = torch.tensor([0]).int().to(device)  # Example Unix timestamp for 2024-01-01 00:00:00 (UTC)
        self.R_e = R_e
        self.conti_huber_delta = conti_huber_delta
        self.dis_in_lambda_reg = dis_in_lambda_reg
        self.max_delta = max_delta
        self.M2_mode = M2_mode
        self.use_graph_propagation = use_graph_propagation
        self.graph_mask_values = graph_mask_values
        self.lambda_spatial, self.lambda_temporal, self.lambda_angular, self.lambda_continuous, self.lambda_discrete = 1, 1, 1, 1, 1

        self.data_encoder = HeterogeneousAttributeEncoder(d_model=d_data_model,
                                                          max_delta=max_delta,
                                                          status=self.data_status,
                                                          navi_status_class_num=len(data_status["navi_status_unique"]),
                                                          cargo_type_class_num=len(data_status["cargo_type_unique"]),
                                                          destination_class_num=len(data_status["destination_unique"]),
                                                          vessel_type_class_num=len(data_status["vessel_type_unique"]),
                                                          )
        
        if self.M2_mode == "esn":
            self.temporal_model = ESNModel(d_data_model, hidden_size, output_size, horizon, num_layers,
                                      leaking_rate, spectral_radius, w_ih_scale, density,
                                      alpha_decay, reservoir_activation, bidirectional)
        elif self.M2_mode == "avg":
            self.temporal_model = AttentionModel(d_data_model, window_size=window_size, num_attributes=14)
        elif self.M2_mode == "scale_attn":
            self.temporal_model = AttentionScaleModel(d_data_model, window_size=window_size, num_attributes=14)

        self.dependency_mining_encoder = CrossScaleDependencyMining(d_mining_model, graph_mask_values=graph_mask_values)

        self.data_decoder = HeterogeneousAttributeDecoder(d_data_model=d_data_model, data_status=data_status, device=device, data_embedding_dim=d_data_model, max_delta=max_delta)

    def forward(self, batch, evaluate=True):
        # ----------------------
        # Stage 1: Data Preparation
        # ----------------------
        observed_data = batch["observed_data"].contiguous().to(self.device)
        observed_labels = batch["observed_labels"].to(self.device)
        masks = batch["masks"].to(self.device)
        padding_masks = batch["padding_masks"].to(self.device)
        self._validate_tensor(observed_data, "observed_data")
        # ----------------------
        # Stage 2: Graph-based Imputation
        # ----------------------
        (p_pred, delta_coord), time_tau, cog, heading_angle, \
         continuous_dict, discrete_list = self.impute_missing_values(observed_data)
        # ----------------------
        # Stage 3: Loss Computation
        # ----------------------
        loss_total, loss_list = self._calculate_loss(
            (p_pred, delta_coord), time_tau, cog, heading_angle,
            continuous_dict, discrete_list,
            observed_labels, padding_masks
        )
        # ----------------------
        # Stage 4: Evaluation Metrics
        # ----------------------
        eval_dict = self._compute_evaluation_metrics(
            p_pred, time_tau, cog, heading_angle,
            continuous_dict, discrete_list,
            observed_labels, masks
        ) if evaluate else {"mae": 0.0, "smape": 0.0, "dtw": 0.0, "acc": 0.0}

        return (loss_total, loss_list), eval_dict

    def impute_missing_values(self, observed_data):
        # ----------------------
        # Stage 1: Feature Decomposition
        # ----------------------
        # Encode raw features x (
        # embedding tensor [b, s, n, f]
        x, continuous_ab_dict = self.data_encoder(observed_data)  
        # Validate encoder output
        self._validate_tensor(x, "data_encoder output")

        # ----------------------
        # Stage 2: Temporal Scale Generation
        # ----------------------
        # Split features by temporal scales (3,3,1,4,3 groups), 
        # Scale mode embedding list, is a list of 5 tensors, the shape of each tensor is [b, s, n_i, f]
        time_scale_splits = _build_scale_mode_embedding_tensor(x.contiguous())
        
        # Process each temporal scale with the selected temporal model 
        processed_scales = []
        for scale_idx, scale_tensor in enumerate(time_scale_splits, 1):
            temporal_output = self.temporal_model(scale_tensor, scale_idx)
            b, s, n_i, l_i, f = temporal_output.shape
            
            reshaped_output = temporal_output.reshape(b*s, n_i, l_i, f)
            self._validate_tensor(reshaped_output, f"Temporal scale {scale_idx} output")
            processed_scales.append(reshaped_output)

        # ----------------------
        # Stage 3: Cross-scale Dependency Modeling
        # ----------------------
        # scale mode features list, which is a list of 5 tensors, the shape of each tensor is [b*s, n_i, l_i, f]
        if self.use_graph_propagation:
            cross_scale_features = self.dependency_mining_encoder(processed_scales)
        else:
            # scale_mode_features = self.dependency_mining_encoder._build_attribute_mode_tensor_from_scale_features_list(processed_scales)
            # attribute_mode_features = self.dependency_mining_encoder._build_attribute_mode_tensor_from_scale_mode_tensor(scale_mode_features)
            # attribute_features_list = self.dependency_mining_encoder._build_attribute_features_list_from_attribute_mode_tensor(attribute_mode_features)
            # cross_scale_features = self.dependency_mining_encoder._build_scale_features_list_from_attribute_features_list(attribute_features_list)
            cross_scale_features = processed_scales
        # scale mode features list, which is a list of 5 tensors, the shape of each tensor is [b*s, n_i, l_i, f]

        # ----------------------
        # Stage 4: Multi-modal Decoding
        # ----------------------
        return self.data_decoder(
            input_list=cross_scale_features,
            raw_features=x,
            observed_data=observed_data,
            continuous_ab_dict=continuous_ab_dict,
            b=b,  # From last processed scale
            s=s   # From last processed scale
        )

    def _calculate_loss(self, p_pred, time_tau, cog, heading_angle,
                        continuous_dict, discrete_list,
                        observed_labels, padding_masks):
        

        loss_list = [0, 0, 0, 0, 0]
        # Spatial loss (geographical coordinates)
        if not (11 in self.graph_mask_values or 12 in self.graph_mask_values):
            loss_sphere = self._calculate_spatial_loss(
                p_pred, observed_labels, padding_masks
            )
            loss_list[0] = loss_sphere * self.lambda_spatial
        
        # Temporal loss (periodic time features)
        if not 13 in self.graph_mask_values:
            loss_time = self._calculate_temporal_loss(
                time_tau, observed_labels, padding_masks
            )
            loss_list[1] = loss_time * self.lambda_temporal
        
        # Angular motion loss
        loss_cycl = self._calculate_angular_loss(
            cog, heading_angle, observed_labels, padding_masks
        )
        loss_list[2] = loss_cycl * self.lambda_angular

        # Continuous features loss
        loss_cont = self._calculate_continuous_loss(
            continuous_dict, observed_labels, padding_masks
        )
        # print(f"loss_cont: {loss_cont}")
        loss_list[3] = loss_cont * self.lambda_continuous

        # Discrete features loss
        loss_disc = self._calculate_discrete_loss(
            discrete_list, observed_labels, padding_masks
        )
        loss_list[4] = loss_disc * self.lambda_discrete

        loss_total = sum(loss_list)
        # log_loss_components(self.logger, *loss_list, loss_total)
        return loss_total, loss_list

    def _calculate_spatial_loss(self, p_pred, labels, padding_masks):
        """Compute geospatial loss (longitude/latitude)"""
        lon = torch.deg2rad(labels[:, :, -2:-1])
        lat = torch.deg2rad(labels[:, :, -3:-2])
        p_true = torch.cat([lon, lat], dim=-1)
        return compute_spatio_loss(p_pred, p_true, padding_masks[:, :, -2:-1].squeeze(-1))

    def _calculate_temporal_loss(self, time_tau, labels, padding_masks):
        """Compute temporal encoding loss"""
        e_tau_true = get_time_interval(labels[:, :, -1:], max_delta=self.max_delta)
        loss_tau = compute_time_interval_loss(time_tau, e_tau_true, self.max_delta, padding_masks[:, :, -1:])
        return loss_tau

    def _calculate_angular_loss(self, cog, heading, labels, padding_masks):
        """Compute cyclical angular losses for COG and heading"""
        # cog_true = angle_sin_cos(labels[:, :, 9:10], self.tau)
        # heading_true = angle_sin_cos(labels[:, :, 10:11], self.tau)
        totl_loss = 0
        if not 9 in self.graph_mask_values:
            cog_true = labels[:, :, 9:10]
            cog_true = torch.deg2rad(cog_true) - torch.pi
            cog_sin_features = torch.sin(cog_true)  # Shape [b, s, 1]
            cog_cos_features = torch.cos(cog_true)  # Shape [b, s, 1]
            cog_sin_cos = torch.cat([cog_sin_features, cog_cos_features], dim=-1)  # Shape [b, s, 2]

            totl_loss += 0.3 * calculate_angle_mse_loss(cog, cog_sin_cos, padding_masks[:, :, 9:10])
            totl_loss += 0.7 * compute_angle_cos_loss(cog, cog_sin_cos, padding_masks[:, :, 9:10])
        if not 10 in self.graph_mask_values:
            heading_true = labels[:, :, 10:11]
            heading_true = torch.deg2rad(heading_true) - torch.pi
            heading_sin_features = torch.sin(heading_true)  # Shape [b, s, 1]
            heading_cos_features = torch.cos(heading_true)  # Shape [b, s, 1]
            heading_sin_cos = torch.cat([heading_sin_features, heading_cos_features], dim=-1)  # Shape [b, s, 2]
           
            totl_loss += 0.3 * calculate_angle_mse_loss(heading, heading_sin_cos, padding_masks[:, :, 10:11])
            totl_loss += 0.7 * compute_angle_cos_loss(heading, heading_sin_cos, padding_masks[:, :, 10:11])
        return totl_loss

    def _calculate_continuous_loss(self, continuous_dict, labels, padding_masks):
        """Compute MSE loss for continuous ship parameters"""
        x_true = torch.cat([
            labels[:, :, 1:2], labels[:, :, 2:3], labels[:, :, 3:4],
            labels[:, :, 7:8], labels[:, :, 8:9]
        ], dim=-1)
        
        conti_mask = torch.cat([
            padding_masks[:, :, 1:2], padding_masks[:, :, 2:3], padding_masks[:, :, 3:4],
            padding_masks[:, :, 7:8], padding_masks[:, :, 8:9]
        ], dim=-1)

       # Map feature indices to their positions in continuous_hat and x_true
        feature_mapping = [(1, 0), (2, 1), (3, 2), (7, 3)]
        
        conti_loss = 0
        valid_components = 0
        for feat_idx, cont_idx in feature_mapping:
            if not feat_idx in self.graph_mask_values:
                conti_loss += compute_continuous_mse_loss(
                    continuous_dict['continuous_hat'][:, :, cont_idx:cont_idx+1],
                    x_true[:, :, cont_idx:cont_idx+1],
                    conti_mask[:, :, cont_idx:cont_idx+1]
                )
                valid_components += 1
        
        conti_loss = conti_loss / valid_components if valid_components > 0 else 0
    
        return conti_loss

    def _calculate_discrete_loss(self, discrete_list, labels, padding_masks):
        """Compute losses for categorical features"""
        LOSS_COMPONENTS = [
            # (feature_index, list_index, mask_index)
            (0, 0, 0),    # Vessel type
            (5, 2, 5),    # Cargo type
            (6, 3, 6),    # Navigation status
            # (4, 1, 4)  # Uncomment for destination (danish_ais)
        ]
        
        total_loss = 0
        valid_components = 0
        for feat_idx, list_idx, mask_idx in LOSS_COMPONENTS:    
            if not feat_idx in self.graph_mask_values:
                total_loss += compute_discrete_loss(
                    y_true=labels[:, :, feat_idx:feat_idx+1].long(),
                    y_pred=discrete_list[list_idx]['y'],
                    z=discrete_list[list_idx]['z'],
                    W_d_one_hot=discrete_list[list_idx]['x'],
                    padding_masks=padding_masks[:, :, mask_idx:mask_idx+1],
                    lambda_reg=self.dis_in_lambda_reg
                )
                valid_components += 1
        return total_loss / valid_components if valid_components > 0 else 0

    def _validate_tensor(self, tensor, name):
        """Validate tensor for NaN/Inf values"""
        assert not torch.isnan(tensor).any(), f"{name} contains NaN values"
        assert not torch.isinf(tensor).any(), f"{name} contains Inf values"

    def _validate_feature_dimensions(self, tensors):
        """Ensure consistent batch and sequence dimensions"""
        base_shape = tensors[0].shape
        for i, tensor in enumerate(tensors[1:]):
            assert tensor.shape[:2] == base_shape[:2], \
                f"Feature {i+1} dimension mismatch: {tensor.shape} vs {base_shape}"

    def evaluate(self, batch):
         # ----------------------
        # Stage 1: Data Preparation
        # ----------------------
        observed_data = batch["observed_data"].contiguous().to(self.device)
        observed_labels = batch["observed_labels"].to(self.device)
        masks = batch["masks"].to(self.device)
        padding_masks = batch["padding_masks"].to(self.device)
        self._validate_tensor(observed_data, "observed_data")
        # ----------------------
        # Stage 2: Graph-based Imputation
        # ----------------------
        ((p_pred, delta_coord), time_tau, (cog, heading_angle), 
         continuous_dict, discrete_list) = self.impute_missing_values(observed_data)
        # ----------------------
        # Stage 3: Evaluation Metrics
        # ----------------------
        return self.get_eval_values(p_pred, time_tau,
                                    cog, heading_angle,
                                    continuous_dict,
                                    discrete_list,
                                    observed_labels, masks)

    def _compute_evaluation_metrics(self, *args):
        """Delegate evaluation metric computation"""
        return self.get_eval_values(*args)

    def get_eval_values(self, coord_pred, time_tau,
                        cog, heading_angle,
                        continuous_dict,
                        discrete_list,
                        observed_labels, masks):
        final_metrics = {}
        # Coordinate conversion
        coord_metrics = self._calculate_coordinate_metrics(coord_pred, observed_labels, masks)
        final_metrics.update(coord_metrics)

        # Time prediction metrics
        # time_pred = restore_timestamp(self.t_base, time_tau, self.T).unsqueeze(-1)
        time_metrics = self._calculate_time_metrics(time_tau, observed_labels, masks)
        final_metrics.update(time_metrics)

        # Angle prediction metrics
        angle_metrics = self._calculate_angle_metrics(cog, heading_angle, observed_labels, masks)
        final_metrics.update(angle_metrics)

        # Continuous variables metrics
        conti_metrics = self._calculate_continuous_metrics(continuous_dict, observed_labels, masks)
        final_metrics.update(conti_metrics)

        # Discrete classification metrics
        discrete_acc = self._calculate_discrete_accuracy(discrete_list, observed_labels, masks)
        final_metrics.update(discrete_acc)

        # log_metrics(self.logger, final_metrics, coordinate_is_mae_smape=True)        
        return final_metrics

    def _calculate_coordinate_metrics(self, coord_pred, labels, observed_masks, metric='haversine'):
        # lon, lat = cartesian_to_lonlat(coord_pred)
        lon, lat = coord_pred[..., 0], coord_pred[..., 1]
        pred_lon_lat = torch.stack([lon, lat], dim=-1)
        true_lon_lat = torch.cat([torch.deg2rad(labels[..., -2:-1]), torch.deg2rad(labels[..., -3:-2])], dim=-1)
        missing_mask = ~observed_masks[..., -2:-1]

        if metric == 'haversine':
            pred_lon_lat = pred_lon_lat[missing_mask.squeeze(-1)]
            true_lon_lat = true_lon_lat[missing_mask.squeeze(-1)]
            distance = haversine_distance(pred_lon_lat, true_lon_lat, False).mean()
            return {
            'coord_distance': distance
            }

        elif metric == 'dtw':
            pred_lon_lat = pred_lon_lat * missing_mask + true_lon_lat * (~missing_mask)
            distance = batch_cpu_dtw(true_lon_lat, pred_lon_lat, missing_mask)
            return {
            'coord_distance': distance
           }
        else:
            def compute_metrics(true, pred, mask):
                return {
                'mae': calculate_one_angle_mae(true, pred, mask),
                'smape': calculate_one_angle_smape(true, pred, mask)
            }

            results = {}
            lon_metrics = compute_metrics(torch.deg2rad(labels[..., -2:-1]), lon.unsqueeze(-1), ~missing_mask)
            lat_metrics = compute_metrics(torch.deg2rad(labels[..., -3:-2]), lat.unsqueeze(-1), ~missing_mask)
            results['coord_mae'] = (lon_metrics['mae'] + lat_metrics['mae']) / 2.0
            results['coord_smape'] = (lon_metrics['smape'] + lat_metrics['smape']) / 2.0

            return results
           

    def _calculate_time_metrics(self, pred, labels, masks):
        observed_time_mask = masks[..., -1:]
        true_time_interval = get_time_interval(labels[..., -1:], self.max_delta)
        
        valid_interval_mask = (true_time_interval <= self.max_delta)
        combined_mask = (~observed_time_mask) & valid_interval_mask
        
        return {
            'time_mae': calculate_one_mae(true_time_interval, pred, ~combined_mask),
            'time_smape': calculate_one_smape(true_time_interval, pred, ~combined_mask)
        }

    def _calculate_angle_metrics(self, cog_pred, heading_pred, labels, masks):
        def compute_metrics(true, pred, mask):
            return {
                'mae': calculate_one_angle_mae(true, pred, mask),
                'smape': calculate_one_angle_smape(true, pred, mask)
            }

        results = {}
        cog_metrics = compute_metrics(torch.deg2rad(labels[..., 9:10]), torch.deg2rad(get_real_angle(cog_pred)), masks[..., 9:10])
        results['cog_mae'] = cog_metrics['mae']
        results['cog_smape'] = cog_metrics['smape']
        heading_metrics = compute_metrics(torch.deg2rad(labels[..., 10:11]), torch.deg2rad(get_real_angle(heading_pred)), masks[..., 10:11])
        results['heading_mae'] = heading_metrics['mae']
        results['heading_smape'] = heading_metrics['smape']

        return results

    def _calculate_continuous_metrics(self, conti_dict, labels, masks):
        conti_hat = conti_dict['continuous_hat']
        metric_config = {
            'width':   (0, 1, 1),
            'length':  (1, 2, 2),
            'draught': (2, 3, 3),
            'sog':     (3, 7, 7),
            # 'rot':   (4, 8, 8)
        }
        
        results = {}
        for name, (pred_idx, label_idx, mask_idx) in metric_config.items():
            pred = conti_hat[..., pred_idx:pred_idx+1]
            true = labels[..., label_idx:label_idx+1]
            mask = masks[..., mask_idx:mask_idx+1]
            
            results[f'{name}_mae'] = calculate_one_mae(true, pred, mask)
            results[f'{name}_smape'] = calculate_one_smape(true, pred, mask)
        
        return results

    def _calculate_discrete_accuracy(self, discrete_list, labels, masks):
        accuracy_config = [
            ('vessel_type', 0, 0, 0),
            # ('destination', 1, 4, 4),
            ('cargo_type', 2, 5, 5),
            ('navi_status', 3, 6, 6)
        ]
        
        total, acc = 0, 0
        results = {}
        for name, pred_idx, label_idx, mask_idx in accuracy_config:
            if pred_idx >= len(discrete_list): 
                continue
            
            pred = discrete_list[pred_idx]['y'].detach()
            true = labels[..., label_idx:label_idx+1].long().detach()
            mask = masks[..., mask_idx:mask_idx+1].detach()
            
            # keep everything in torch and run calculate_acc on tensors
            current_acc = calculate_acc(
                pred.cpu(), true.cpu(), mask.cpu()
            )

            # 🔑 Ensure result is tensor (not numpy)
            if not isinstance(current_acc, torch.Tensor):
                current_acc = torch.tensor(current_acc, dtype=torch.float32)

            results[f'{name}_acc'] = current_acc.to(self.device)  # safe for DataParallel
            acc += current_acc.item()
            total += 1
        
        return results

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

def _build_scale_mode_embedding_tensor(x):
    b, s, n, f = x.shape
    if n < 14:  # 3 + 3 + 1 + 4 + 3 = 14
        raise ValueError(f"Input tensor's third dimension (n={n}) is too small for the specified splitting rule.")
    splits = [
        x[:, :, :3, :],  #  3
        x[:, :, 3:6, :],  #  3  
        x[:, :, 6:7, :],  #  1
        x[:, :, 7:11, :],  #  4
        x[:, :, 11:14, :]  #  3
    ]
    return splits


if __name__ == "__main__":
    pass
