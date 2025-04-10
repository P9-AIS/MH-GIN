import torch
from torch import nn
import argparse
import torch.nn.functional as F

from src.models.reservoir import Reservoir
from src.utils.parser_utils import str_to_bool

class ESNModel(nn.Module):
    """ Applies an Echo State Network to an input sequence. Multi-layer Echo
       State Network is based on paper
       Deep Echo State Network (DeepESN): A Brief Survey - Gallicchio, Micheli 2017

       Args:
           input_size: The number of expected features in the input x.
           hidden_size: The number of features in the hidden state h.
           output_size: The number of expected features in the output y.
           num_layers: Number of recurrent layers. Default: 1
           leaking_rate: Leaking rate of reservoir's neurons. Default: 1
           spectral_radius: Desired spectral radius of recurrent weight matrix.
               Default: 0.9
           w_ih_scale: Scale factor for first layer's input weights (w_ih_l0). It
               can be a number or a tensor of size '1 + input_size' and first element
               is the bias' scale factor. Default: 1
           density: Recurrent weight matrix's density. Default: 1
       """
    def __init__(self,
                 input_size,
                 hidden_size,
                 output_size,
                 horizon,
                 num_layers=1,
                 leaking_rate=0.9,
                 spectral_radius=0.9,
                 w_ih_scale=1,
                 density=0.9,
                 alpha_decay=False,
                 reservoir_activation='tanh',
                 bidirectional=False,
                 ):
        super(ESNModel, self).__init__()
        self.reservoir = Reservoir(input_size=input_size,
                                   hidden_size=hidden_size,
                                   input_scaling=w_ih_scale,
                                   num_layers=num_layers,
                                   leaking_rate=leaking_rate,
                                   spectral_radius=spectral_radius,
                                   density=density,
                                   activation=reservoir_activation,
                                   alpha_decay=alpha_decay,
                                   bidirectional=bidirectional)

    def forward(self, x, time_scale=1):
        # x: [batches, steps, nodes, features]
        # if time_scale > 1:
        #     ox = self.reservoir(x, time_scale-1, return_last_state=False) # b s n (l f)
        #     b, s, n, lf = ox.shape
        #     l = time_scale - 1
        #     f = lf // l
        #     ox = ox.view(b, s, n, l, f)
        #     x = x.unsqueeze(3)
        #     x = torch.cat([x, ox], dim=3)  # [b, s, n, l + 1, f]
        # else:
        #     x = x.unsqueeze(3) # [b, s, n, 1, f]

        ox = self.reservoir(x, time_scale, return_last_state=False) # b s n (l f)
        b, s, n, lf = ox.shape
        l = time_scale
        f = lf // l
        ox = ox.view(b, s, n, l, f) 
        return ox

    @staticmethod
    def add_model_specific_args(parser: argparse):
        parser.add_argument('--reservoir-size', type=int, default=32, choices=[16, 32, 64, 128, 256])
        parser.add_argument('--reservoir-layers', type=int, default=1, choices=[1, 2, 3, 4, 5])
        parser.add_argument('--spectral-radius', type=float, default=0.9,choices=[0.7, 0.8, 0.9])
        parser.add_argument('--leaking-rate', type=float, default=0.9, choices=[0.7, 0.8, 0.9])
        parser.add_argument('--density', type=float, default=0.7, choices=[0.7, 0.8, 0.9])
        parser.add_argument('--input-scaling', type=float, default=1., choices=[1., 1.5, 2.])
        parser.add_argument('--bidirectional', type=str_to_bool, default=True)
        parser.add_argument('--alpha-decay', type=str_to_bool,  default=False)
        parser.add_argument('--reservoir-activation', type=str, default='tanh')
        parser.add_argument('--horizon', type=int, default=1)
        return parser

class AttentionModel(nn.Module):
    def __init__(self, d_model, window_size=10, num_attributes=14):
        super(AttentionModel, self).__init__()
        self.d_model = d_model
        self.window_size = window_size
        
        self.attn_keys = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(num_attributes)
        ])
        self.attn_queries = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(num_attributes)
        ]) 
        self.attn_transforms = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(num_attributes)
        ])
    
    def forward(self, scale_tensor, scale_idx):
        # attribute num of each scale: 
        attribute_nums = [3, 3, 1, 4, 3]
        b, s, n_i, f = scale_tensor.shape
        l_i = scale_idx
        k = self.window_size
        
        h = []
        for attribute_idx in range(n_i):
            seq_input = scale_tensor[:, :, attribute_idx, :]
            
            padded = F.pad(seq_input, (0, 0, k, k), "constant", 0)
            
            windows = padded.unfold(1, 2*k+1, 1).permute(0, 1, 3, 2)
            
            zero_feat_mask = torch.all(windows == 0, dim=-1)  # [b, s, 2k+1]
            
            query = self.attn_queries[attribute_idx + sum(attribute_nums[:scale_idx-1])](seq_input).unsqueeze(2)
            keys = self.attn_keys[attribute_idx + sum(attribute_nums[:scale_idx-1])](windows)
            attention = torch.matmul(query, keys.transpose(-1, -2)).squeeze(2) / (f**0.5)
            
            attn_weights = torch.ones_like(attention).detach()
            attn_weights = attn_weights.masked_fill(zero_feat_mask, -1e9)
            attn_weights = F.softmax(attn_weights, dim=-1)  # [b, s, 2k+1]
            
            context = torch.einsum('bst,bstf->bsf', attn_weights, windows)
            
            h_one_attribute = self.attn_transforms[attribute_idx + sum(attribute_nums[:scale_idx-1])](context)
            h.append(h_one_attribute.reshape(b, s, -1).unsqueeze(2))
        
        h = torch.cat(h, dim=2)
        
        output = h.unsqueeze(3).expand(-1, -1, -1, l_i, -1)
        
        return output

class AttentionScaleModel(nn.Module):
    def __init__(self, d_model, window_size=10, num_attributes=14):
        super(AttentionScaleModel, self).__init__()
        self.d_model = d_model
        self.window_size = window_size
        
        self.attn_keys = nn.ModuleList([
            nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(5)]) for _ in range(num_attributes)
        ])
        self.attn_queries = nn.ModuleList([
            nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(5)]) for _ in range(num_attributes)
        ]) 
        self.attn_transforms = nn.ModuleList([
            nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(5)]) for _ in range(num_attributes)
        ])

    def forward(self, scale_tensor, scale_idx):
        # attribute num of each scale: 
        attribute_nums = [3, 3, 1, 4, 3]
        b, s, n_l, f = scale_tensor.shape
        l_i = scale_idx
        k = self.window_size * (6 - l_i)

        output = []
        for attribute_idx in range(n_l):
            h = []
            for l in range(l_i):
                # [b, s, f]
                seq_input = scale_tensor[:, :, attribute_idx, :]
                padded = F.pad(seq_input, (0, 0, k, k), "constant", 0)
                windows = padded.unfold(1, 2*k+1, 1).permute(0, 1, 3, 2)
                
                zero_feat_mask = torch.all(windows == 0, dim=-1)  # [b, s, 2k+1]
                
                query = self.attn_queries[attribute_idx + sum(attribute_nums[:scale_idx-1])][l](seq_input).unsqueeze(2)
                keys = self.attn_keys[attribute_idx + sum(attribute_nums[:scale_idx-1])][l](windows)
                attention = torch.matmul(query, keys.transpose(-1, -2)).squeeze(2) / (f**0.5)
                
                # attn_weights = torch.ones_like(attention).detach()
                attn_weights = attention
                attn_weights = attn_weights.masked_fill(zero_feat_mask, -1e9)
                attn_weights = F.softmax(attn_weights, dim=-1)  # [b, s, 2k+1]
                
                # [b, s, f]
                context = torch.einsum('bst,bstf->bsf', attn_weights, windows)
                
                # [b, s, f]
                h_one_attribute = self.attn_transforms[attribute_idx + sum(attribute_nums[:scale_idx-1])][l](context)
                h.append(h_one_attribute.reshape(b, s, -1).unsqueeze(2))
            # [b, s, l_i, f]
            h = torch.cat(h, dim=2)
            output.append(h.unsqueeze(2))

        output = torch.cat(output, dim=2) 
        return output

