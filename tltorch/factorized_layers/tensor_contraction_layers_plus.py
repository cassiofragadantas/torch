"""
Tensor Contraction Layers Plus
"""

# Authors: Jean Kossaifi, Cassio F. Dantas
# License: BSD 3 clause

from tensorly import tenalg
import torch
import torch.nn as nn
from torch.nn import init

import math

import tensorly as tl
tl.set_backend('pytorch')


class TCLplus(nn.Module):
    """Tensor Contraction Layer Plus (TCL+) [1]_

    Parameters
    ----------
    input_size : int iterable
        shape of the input, excluding batch size
    rank : int list or int
        rank of the TCL+, will also be the output-shape (excluding batch-size)
        if int, the same rank will be used for all dimensions
    sum_terms : int, default is 1
        Summing terms in the TCL+, sum_terms=1 leads to standard TCL.
    verbose : int, default is 1
        level of verbosity

    References
    ----------
    .. [1] TO APPEAR.
    """
    def __init__(self, input_shape, rank, sum_terms=1, verbose=0, bias=False, **kwargs):
        super().__init__(**kwargs)
        self.verbose = verbose
        self.sum_terms = int(sum_terms)

        if isinstance(input_shape, int):
            self.input_shape = (input_shape, )
        else:
            self.input_shape = tuple(input_shape)

        self.order = len(input_shape)

        if isinstance(rank, int):
            self.rank = (rank, )*self.order
        else:
            self.rank = tuple(rank)

        # Start at 1 as the batch-size is not projected
        self.contraction_modes = list(range(1, self.order + 1))
        for k in range(self.sum_terms):
            for i, (s, r) in enumerate(zip(self.input_shape, self.rank)):
                self.register_parameter(f'factor_{k}{i}', nn.Parameter(torch.Tensor(r, s)))
        
        # self.factors = ParameterList(parameters=factors)
        if bias:
            self.bias = nn.Parameter(
                tl.tensor(self.output_shape), requires_grad=True)
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    @property
    def factors(self):
        res = []
        for k in range(self.sum_terms):
            res.append([getattr(self, f'factor_{k}{i}') for i in range(self.order)])
        return res

    def forward(self, x):
        """Performs a forward pass"""
        res = 0.0
        #res = torch.zeros((x.shape[0], ) + self.rank, dtype=torch.float32)        
        for k in range(self.sum_terms):
            res += tenalg.multi_mode_dot(
                    x, self.factors[k], modes=self.contraction_modes)
        # Implement this sum in a lower level as tenalg.sum_multi_mode_dot ?      

        if self.bias is not None:
            return res + self.bias
        else:
            return res

    def reset_parameters(self):
        """Sets the parameters' values randomly

        Todo
        ----
        This may be renamed to init_from_random for consistency with TensorModules
        """
        for k in range(self.sum_terms):
            for i in range(self.order):
                init.kaiming_uniform_(self.factors[k][i], a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.input_shape[0])
            init.uniform_(self.bias, -bound, bound)
