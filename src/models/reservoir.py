"""

Code extensively inspired by https://github.com/stefanonardo/pytorch-esn

"""

import numpy as np
import torch
import torch.nn as nn
import torch.sparse
from einops import rearrange
from torch.nn import functional as F

from src.utils.utils import get_functional_activation
from src.utils.utils import self_normalizing_activation

class ReservoirLayer(nn.Module):
    """
    Core ESN layer implementation
    Key features:
    - Fixed random weights (non-trainable)
    - Spectral radius control (echo state property)
    - Leaky integrator mechanism (memory/update balance)
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 spectral_radius,
                 leaking_rate,
                 bias=True,
                 density=1.,
                 in_scaling=1.,
                 bias_scale=1.,
                 activation='tanh'):
        super(ReservoirLayer, self).__init__()
        self.w_ih_scale = in_scaling
        self.b_scale = bias_scale
        self.density = density
        self.hidden_size = hidden_size
        self.alpha = leaking_rate
        self.spectral_radius = spectral_radius

        assert activation in ['tanh', 'relu', 'self_norm', 'identity']
        if activation == 'self_norm':
            self.activation = self_normalizing_activation
        else:
            self.activation = get_functional_activation(activation)

        self.w_ih = nn.Parameter(torch.Tensor(hidden_size, input_size),
                                 requires_grad=False)
        self.w_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size),
                                 requires_grad=False)
        if bias is not None:
            self.b_ih = nn.Parameter(torch.Tensor(hidden_size),
                                     requires_grad=False)
        else:
            self.register_parameter('b_ih', None)
        self.reset_parameters()

    def reset_parameters(self):
        # Initialize input-hidden weights with proper scaling
        self.w_ih.data.uniform_(-1, 1)
        self.w_ih.data.mul_(self.w_ih_scale)

        # Initialize biases with proper scaling if present
        if self.b_ih is not None:
            self.b_ih.data.uniform_(-1, 1)
            self.b_ih.data.mul_(self.b_scale)

        # Initialize recurrent weights
        self.w_hh.data.uniform_(-1, 1)

        # Apply density mask if needed
        if self.density < 1:
            n_units = self.hidden_size * self.hidden_size
            mask = self.w_hh.data.new_ones(n_units)
            # Calculate number of weights to keep
            keep = int(n_units * self.density)
            # Create mask with 1s for kept weights
            masked_weights = torch.randperm(n_units)[:n_units-keep]
            mask[masked_weights] = 0.
            self.w_hh.data.mul_(mask.view(self.hidden_size, self.hidden_size))

        # Spectral radius adjustment with numerical stability
        abs_eigs = torch.linalg.eigvals(self.w_hh.data).abs()
        max_eig = torch.max(abs_eigs)
        # Only adjust if matrix is not all zeros
        if max_eig > 1e-8:  # Add small epsilon to avoid division by zero
            self.w_hh.data.mul_(self.spectral_radius / max_eig)

    def forward(self, x, h):
        # Leaky integrator equation implementation
        # h_new = (1-α)*h_prev + α*activation(W_ih*x + W_hh*h + b)
        out1 = F.linear(x, self.w_ih, self.b_ih)  # Input transformation
        out2 = F.linear(h, self.w_hh)  # Recurrent connection
        h_new = self.activation(out1 + out2)
        h_new = (1 - self.alpha) * h + self.alpha * h_new  # Leakage integration
        return h_new


class Reservoir(nn.Module):
    """
    Multi-layer Echo State Network container
    Supported features:
    - Bidirectional processing
    - Layer-wise alpha decay
    - Multi-scale temporal pattern extraction
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 input_scaling=1.,
                 num_layers=1,
                 leaking_rate=0.9,
                 spectral_radius=0.9,
                 density=0.9,
                 activation='tanh',
                 bias=True,
                 alpha_decay=False,
                 bidirectional=False):
        super(Reservoir, self).__init__()
        self.mode = activation
        self.input_size = input_size
        self.input_scaling = input_scaling
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.leaking_rate = leaking_rate
        self.spectral_radius = spectral_radius
        self.density = density
        self.bias = bias
        self.alpha_decay = alpha_decay
        self.bidirectional = bidirectional

        layers = []
        alpha = leaking_rate
        for i in range(num_layers):
            layers.append(
                ReservoirLayer(
                    input_size=input_size if i == 0 else hidden_size,
                    hidden_size=hidden_size,
                    in_scaling=input_scaling,
                    density=density,
                    activation=activation,
                    spectral_radius=spectral_radius,
                    leaking_rate=alpha
                )
            )
            if self.alpha_decay:
                alpha = np.clip(alpha - 0.1, 0.1, 1.)

        self.reservoir_layers = nn.ModuleList(layers)
        if self.bidirectional:
            self.reverse_reservoir_layers = nn.ModuleList(layers)

    def reset_parameters(self):
        for layer in self.reservoir_layers:
            layer.reset_parameters()
        if self.bidirectional:
            for layer in self.reverse_reservoir_layers:
                layer.reset_parameters()

    def forward_prealloc(self, x, return_last_state=False):
        # x : b s n f
        *batch_size, steps, nodes, _ = x.size()

        if x.ndim == 4:
            batch_size = x.size(0)
            x = rearrange(x, 'b s n f -> s (b n) f')

        out = torch.empty((steps, x.size(1),
                           len(self.reservoir_layers) * self.hidden_size),
                          dtype=x.dtype, device=x.device)
        out[0] = 0
        size = [slice(i * self.hidden_size, (i + 1) * self.hidden_size)
                for i in range(len(self.reservoir_layers))]
        # for each step, update the reservoir states for all layers
        for s in range(steps):
            # for all layers, observe input and compute updated states
            x_s = x[s]
            for i, layer in enumerate(self.reservoir_layers):
                x_s = layer(x_s, out[s, :, size[i]])
                out[s, :, size[i]] = x_s
        if isinstance(batch_size, int):
            out = rearrange(out, 's (b n) f -> b s n f', b=batch_size, n=nodes)
        if return_last_state:
            return out[:, -1]
        return out


    def pre_forward(self, x, layer_num, reservoir_layers, h0=None):
        # x : b s n f
        batch_size, steps, nodes, _ = x.size()
        if h0 is None:

            h0 = x.new_zeros(layer_num, batch_size * nodes,
                             self.hidden_size, requires_grad=False)
        x = rearrange(x, 'b s n f -> s (b n) f')
        # print(f'x-first-shape: {x.shape}')
        out = []
        h = h0
        # print(f'h0-first-shape: {h0.shape}')
        # for each step, update the reservoir states for all layers
        for s in range(steps):
            # print(f'step: {s}')
            h_s = []
            # for all layers, observe input and compute updated states
            x_s = x[s]
            for i in range(layer_num):
                x_s = reservoir_layers[i](x_s, h[i])
                h_s.append(x_s)  # b*n, f
            # update all states
            h = torch.stack(h_s)  # l, b*n, f
            # collect states
            out.append(h)
        out = torch.stack(out)  # [s, l, b, (n), f]
        out = rearrange(out, 's l (b n) f -> b s n (l f)', b=batch_size,
                        n=nodes)
        # [b, s, n, l*f]
        return out


    def forward(self, x, layer_num, h0=None, return_last_state=False):
        """
        Forward workflow:
        1. Input restructuring (batch/sequence handling)
        2. Time-step state updates
        3. Bidirectional processing (optional)
        4. State aggregation and output
        """
        # Forward processing
        out = self.pre_forward(x, layer_num, self.reservoir_layers, h0)
        
        # Bidirectional handling
        if self.bidirectional:
            out_reverse = self.pre_forward(torch.flip(x, [1]), layer_num, self.reverse_reservoir_layers, h0)
            out += out_reverse
            
        return out[:, -1] if return_last_state else out
