import torch
import torch.nn as nn

class CrossScaleDependencyMining(nn.Module):
    def __init__(self, feature_dim, num_scales=5, graph_mask_values=[4, 8]):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_scales = num_scales
        self.graph_mask_values =  graph_mask_values

        # Stage 1 learnable adjacency matrix
        self.ak_l = nn.Parameter(torch.Tensor(43, 43))
        self.ak_l_p = self._create_fixed_adjacency_stage1(self.ak_l.device, self.graph_mask_values)
        # Stage 2 learnable adjacency matrix
        self.ax_l = nn.Parameter(torch.Tensor(43, 43))
        self.ax_l_p = self._create_fixed_adjacency_stage2(self.ax_l.device)

        nn.init.xavier_uniform_(self.ak_l)
        nn.init.xavier_uniform_(self.ax_l)
       
        self.linear = nn.Linear(3 * feature_dim, feature_dim, bias=False)

        # Learnable function for stage 1 dynamic edge weights
        self.f_edge = nn.Linear(2 * feature_dim, 1, bias=False)

        self._init_adj_matrices()

    def _create_fixed_adjacency_stage1(self, device, graph_mask_values):
        """Create non-learnable block-diagonal adjacency matrix with specified mask values."""
        matrix_size = 43
        block_sizes = [14, 11, 8, 7, 3]
        
        # Initialize zero matrix
        adj = torch.zeros((matrix_size, matrix_size), dtype=torch.float32)
        
        # Build block diagonals and record block offsets
        start_offsets = []
        start = 0
        for size in block_sizes:
            end = start + size
            adj[start:end, start:end] = 1  # Set block to 1
            start_offsets.append(start)
            start = end  # Move to next block

        # Define mask mapping: each mask_value maps to list of (block_idx, local_idx)
        mask_mapping = {
            0: [(0, 0)],
            1: [(0, 1)],
            2: [(0, 2)],
            3: [(0, 3), (1, 0)],
            4: [(0, 4), (1, 1)],
            5: [(0, 5), (1, 2)],
            6: [(0, 6), (1, 3), (2, 0)],
            7: [(0, 7), (1, 4), (2, 1), (3, 0)],
            8: [(0, 8), (1, 5), (2, 2), (3, 1)],
            9: [(0, 9), (1, 6), (2, 3), (3, 2)],
            10: [(0, 10), (1, 7), (2, 4), (3, 3)],
            11: [(0, 11), (1, 8), (2, 5), (3, 4), (4, 0)],
            12: [(0, 12), (1, 9), (2, 6), (3, 5), (4, 1)],
            13: [(0, 13), (1, 10), (2, 7), (3, 6), (4, 2)],
        }
        
        # Apply mask for each mask_value in the input list
        for mv in graph_mask_values:
            if mv in mask_mapping:
                for block_idx, local_idx in mask_mapping[mv]:
                    start = start_offsets[block_idx]
                    global_idx = start + local_idx
                    adj[global_idx, :] = 0
                    adj[:, global_idx] = 0
        
        # Ensure matrix is on correct device and non-trainable
        adj = adj.to(device).requires_grad_(False)
        
        return adj    

    def _create_fixed_adjacency_stage2(self, device):
        """Create non-learnable block-diagonal adjacency matrix"""
        matrix_size = 43
        block_sizes = [1,1,1,2,2,2,3,4,4,4,4,5,5,5]
        
        # Initialize zero matrix
        adj = torch.zeros((matrix_size, matrix_size), dtype=torch.float32)
        
        # Build block diagonals
        start = 0
        for size in block_sizes:
            end = start + size
            adj[start:end, start:end] = 1  # Set block to 1
            start = end  # Move to next block
        
        # Ensure matrix is on correct device and non-trainable
        adj = adj.to(device).requires_grad_(False)
        
        return adj

    def _init_adj_matrices(self):
        # Initialize stage1 adjacency matrix (Ak)
        degree = torch.sum(self.ak_l_p, dim=1, keepdim=True)
        self.degree_inv_sqrt_k = torch.sqrt(1.0 / degree)
        self.degree_inv_sqrt_k[torch.isinf(self.degree_inv_sqrt_k)] = 0.0
        self.degree_inv_sqrt_k.requires_grad = False

        # Initialize stage2 adjacency matrix (Ax)
        degree = torch.sum(self.ax_l_p, dim=1, keepdim=True)
        self.degree_inv_sqrt_x = torch.sqrt(1.0 / degree)
        self.degree_inv_sqrt_x[torch.isinf(self.degree_inv_sqrt_x)] = 0.0
        self.degree_inv_sqrt_x.requires_grad = False

    # get attribute mode tensor [b*s, sum([1,1,1,2,2,2,3,4,4,4,4,5,5,5]), f]
    def _build_attribute_mode_tensor_from_scale_features_list(self, inputs):
        reshaped_tensors = []
        for tensor in inputs:
            # tensor [b*s, n_i, i, f]
            reshaped_tensor = tensor.reshape(tensor.size(0), -1, tensor.size(-1))
            # tensor [b*s, n_i*i, f]
            reshaped_tensors.append(reshaped_tensor)
        concatenated_tensor = torch.cat(reshaped_tensors, dim=1)

        return concatenated_tensor

    # get scale mode tensor [b*s, sum([14, 11, 8, 7, 3]), f]
    def _build_scale_mode_tensor_from_scale_features_list(self, inputs):
        # inputs: list of 5 tensors [b*s, n_i, l_i, f]
        hk_parts = []

        # Collect features for each temporal layer
        for l in range(5):  # l=0~4
            part = []
            for i, tensor in enumerate(inputs):
                _, n, l_total, _ = tensor.shape
                if l < l_total:
                    feat = tensor[:, :, l, :]  # [b*s, n, f]
                    part.append(feat)
            if part:
                hk_parts.append(torch.cat(part, dim=1))
        return torch.cat(hk_parts, dim=1)  # [b*s, 43, f]

    def _build_scale_features_list_from_attribute_features_list(self, outputs):
        out_update_list = []
        ptr = 0
        # Step 1:
        first_three = torch.stack(outputs[:3], dim=1)  # [b*s, 3, 1, f]
        out_update_list.append(first_three)
        ptr += 3

        # Step 2:
        size_2_tensors = []
        while ptr < len(outputs) and outputs[ptr].size(1) == 2:
            size_2_tensors.append(outputs[ptr])
            ptr += 1
        if size_2_tensors:
            size_2_stacked = torch.cat(size_2_tensors, dim=1).view(-1, 3, 2, outputs[0].size(-1))  # [b*s, 3, 2, f]
            out_update_list.append(size_2_stacked)

        # Step 3:
        size_3_tensors = []
        while ptr < len(outputs) and outputs[ptr].size(1) == 3:
            size_3_tensors.append(outputs[ptr])
            ptr += 1
        if size_3_tensors:
            size_3_stacked = torch.cat(size_3_tensors, dim=1).view(-1, 1, 3, outputs[0].size(-1))  # [b*s, 1, 3, f]
            out_update_list.append(size_3_stacked)

        # Step 4:
        size_4_tensors = []
        while ptr < len(outputs) and outputs[ptr].size(1) == 4:
            size_4_tensors.append(outputs[ptr])
            ptr += 1
        if size_4_tensors:
            size_4_stacked = torch.cat(size_4_tensors, dim=1).view(-1, 4, 4, outputs[0].size(-1))  # [b*s, 4, 4, f]
            out_update_list.append(size_4_stacked)

        # Step 5:
        size_5_tensors = []
        while ptr < len(outputs) and outputs[ptr].size(1) == 5:
            size_5_tensors.append(outputs[ptr])
            ptr += 1
        if size_5_tensors:
            size_5_stacked = torch.cat(size_5_tensors, dim=1).view(-1, 3, 5, outputs[0].size(-1))  # [b*s, 3, 5, f]
            out_update_list.append(size_5_stacked)

        return out_update_list

    # get attribute mode tensor from scale mode tensor [b*s, sum([1,1,1,2,2,2,3,4,4,4,4,5,5,5]), f]
    def _build_attribute_mode_tensor_from_scale_mode_tensor(self, hx):
        # Split hx into temporal blocks, 
        temp = torch.split(hx, [14, 11, 8, 7, 3], dim=1)

        # Initialize Hxc with zeros
        batch_size, _, feat_dim = hx.size()
        hxc = torch.zeros(batch_size, 43, feat_dim, device=hx.device)

        # Step 2(1): 0-2 from first block -> positions 0-2
        hxc[:, 0:3] = temp[0][:, 0:3]

        # Step 2(2): 3 cycles of 3+0, 4+1, 5+2 -> positions 3-8
        for i in range(3):
            src1 = temp[0][:, 3 + i:4 + i]  # First block slice
            src2 = temp[1][:, i:i + 1]  # Second block slice
            hxc[:, 3 + 2 * i:5 + 2 * i] = torch.cat([src1, src2], dim=1)

        # Step 2(3): 6+3+0 -> positions 9-11
        src0 = temp[0][:, 6:7]
        src1 = temp[1][:, 3:4]
        src2 = temp[2][:, 0:1]
        hxc[:, 9:12] = torch.cat([src0, src1, src2], dim=1)

        # Step 2(4): 4 cycles of 7+4+1+0 -> positions 12-27
        for i in range(4):
            sources = [
                temp[0][:, 7 + i:8 + i],
                temp[1][:, 4 + i:5 + i],
                temp[2][:, 1 + i:2 + i],
                temp[3][:, 0 + i:1 + i]
            ]
            start = 12 + i * 4
            hxc[:, start:start + 4] = torch.cat(sources, dim=1)

        # Step 2(5): 3 cycles of 11+8+5+5+0 -> positions 28-42
        for i in range(3):
            sources = [
                temp[0][:, 11 + i:12 + i],
                temp[1][:, 8 + i:9 + i],
                temp[2][:, 5 + i:6 + i],
                temp[3][:, 4 + i:5 + i],
                temp[4][:, 0 + i:1 + i]
            ]
            start = 28 + i * 5
            hxc[:, start:start + 5] = torch.cat(sources, dim=1)

        return hxc

    def _build_attribute_features_list_from_attribute_mode_tensor(self, h_final_out):
        output_sizes = [1, 1, 1, 2, 2, 2, 3, 4, 4, 4, 4, 5, 5, 5]
        outputs = []
        ptr = 0
        for size in output_sizes:
            outputs.append(h_final_out[:, ptr:ptr + size, :])
            ptr += size
        if ptr != h_final_out.size(1):
            raise ValueError(
                f"Sum of output_sizes ({ptr}) does not match the second dimension of input tensor ({h_final_out.size(1)}).")
        return outputs

    # def _scale_graph_propagate(self, hk):
    #     # Compute normalized adjacency
    #     degree_inv_sqrt = self.degree_inv_sqrt_k.to(hk.device)
    #     norm_ak = degree_inv_sqrt * (self.ak_l * self.ak_l_p.to(hk.device)) * degree_inv_sqrt.t()

    #     return torch.matmul(norm_ak, hk)

    def _compute_dynamic_adjacency(self, hk_curr, hk_next):
        """Compute dynamic adjacency matrix based on current and higher-scale contextual features"""
        batch_size, n_nodes, feat_dim = hk_curr.shape
        
        hk_i = hk_curr.unsqueeze(2).expand(-1, -1, n_nodes, -1)  # [b*s, 43, 43, f]
        hk_j = hk_curr.unsqueeze(1).expand(-1, n_nodes, -1, -1)  # [b*s, 43, 43, f]
        
        edge_features = torch.cat([hk_i, hk_j], dim=-1)  # [b*s, 43, 43, 2*f]
        
        edge_weights = self.f_edge(edge_features).squeeze(-1)  # [b*s, 43, 43]
        
        edge_weights = torch.sigmoid(edge_weights)
        
        return edge_weights

    def _scale_graph_propagate(self, hk, hk_next=None):
        """Stage 1: Intra-scale Temporal Alignment with dynamic adjacency"""
        if hk_next is not None:
            dynamic_weights = self._compute_dynamic_adjacency(hk, hk_next)
        else:
            dynamic_weights = self._compute_dynamic_adjacency(hk, hk)
        
        ak_l_p_device = self.ak_l_p.to(hk.device)
        ak_l_device = self.ak_l.to(hk.device)
        
        A_k = dynamic_weights * ak_l_p_device + ak_l_device * ak_l_p_device
        
        degree_inv_sqrt = self.degree_inv_sqrt_k.to(hk.device) 
        norm_A_k = degree_inv_sqrt * A_k * degree_inv_sqrt.t()
        
        return torch.matmul(norm_A_k, hk)

    def _attribute_graph_propagate(self, hx):
        # Compute normalized adjacency
        degree_inv_sqrt = self.degree_inv_sqrt_x.to(hx.device)
        norm_ax = degree_inv_sqrt * (self.ax_l * self.ax_l_p.to(hx.device)) * degree_inv_sqrt.t()

        return torch.matmul(norm_ax, hx)

    def forward(self, inputs):
        # inputs: scale mode features list, which is a list of 5 tensors, the shape of each tensor is [b*s, n_i, l_i, f]
        h_0_attribute_mode = self._build_attribute_mode_tensor_from_scale_features_list(inputs)

        # Stage 1: Intra-scale Temporal Alignment
        h_0_scale_mode = self._build_scale_mode_tensor_from_scale_features_list(inputs)  # [b*s, 43, f]
        h_1_scale_mode = self._scale_graph_propagate(h_0_scale_mode)

        h_1_attribute_mode = self._build_attribute_mode_tensor_from_scale_mode_tensor(h_1_scale_mode)

        # Stage 2: Cross-scale Dependency Learning
        # Feature reorganization (simplified for demonstration)
        h_2_attribute_mode = self._attribute_graph_propagate(h_1_attribute_mode)
        # [b*s, 43, f]
        combined_attribute_mode = torch.cat([h_0_attribute_mode, h_1_attribute_mode, h_2_attribute_mode], dim=-1)  # [b*s, 43, 3*f]
        h_final_out_attribute_mode = self.linear(combined_attribute_mode)  # Linear layer converts [b*s, 43, 3*f] to [b*s, 43, f]
        
        # output is attribute features list, which is a list of 14 tensors, the shape of each tensor is [b*s, l_i, f]
        outputs = self._build_attribute_features_list_from_attribute_mode_tensor(h_final_out_attribute_mode)
        return self._build_scale_features_list_from_attribute_features_list(outputs)

if __name__ == "__main__":
    pass