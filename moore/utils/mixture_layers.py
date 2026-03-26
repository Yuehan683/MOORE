# Copyright 2021 The PODNN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#=====================================================================================

"""
Implementation is adapted from https://github.com/caisr-hh/podnn.
This GS version has been extended so its monitoring outputs stay aligned with the
HH version used in the project.
"""

import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy

n_models_global = 5
agg_out_dim = 3


class InputLayer(nn.Module):
    """
       InputLayer stucture the data in a parallel form ready to be consumed by
       the upcoming parallel layes.
    """

    def __init__(self,n_models):
        """
        Arg:
            n_models: number of individual models within the ensemble
        """
        super(InputLayer, self).__init__()
        self.n_models = n_models
        global n_models_global
        n_models_global = self.n_models

    def forward(self,x):
        """
        Arg
            x: is the input to the network as in other standard deep neural networks.

        return:
            x_parallel: is the parallel form of the received input (x).
        """
        x_parallel = torch.unsqueeze(x,0)
        x_parallel_next = torch.unsqueeze(x, 0)
        for _ in range(1,self.n_models):
            x_parallel =  torch.cat((x_parallel,x_parallel_next),axis=0)

        return x_parallel


class ParallelLayer(nn.Module):
    """
        Parallellayer creates a parallel layer from the structure of unit_model it receives.
    """

    def __init__(self, unit_model):
        """
        Arg:
            unit_model: specifies what computational module each unit of the parallel layer contains.
                        unit_model is a number of layer definitions followed by each other.
        """
        super(ParallelLayer,self).__init__()
        self.n_models = n_models_global
        self.model_layers = []
        for _ in range(self.n_models):
            for j in range(len(unit_model)):
                try:
                    unit_model[j].reset_parameters()
                except Exception:
                    pass
            self.model_layers.append(deepcopy(unit_model))
        self.model_layers = nn.ModuleList(self.model_layers)

    def forward(self, x):
        """
        Arg:
            x: is the parallel input with shape [n_models,n_samples,dim] for fully connected layers.

        return:
            parallel_output: is the output formed by applying modules within each units on the input.

            shape: [n_models,n_samples,dim] for fully connected layers..
        """
        parallel_output = self.model_layers[0](x[0])
        parallel_output = torch.unsqueeze(parallel_output,0)
        for i in range(1,self.n_models):
            next_layer = self.model_layers[i](x[i])
            next_layer = torch.unsqueeze(next_layer, 0)
            parallel_output = torch.cat((parallel_output,next_layer),0)

        return parallel_output


def orth_error_stats(basis: torch.Tensor):
    """
    Compute the same core orthogonality diagnostics that the HH version logs.
    basis shape: [K, B, D]
    """
    if basis.numel() == 0:
        return {
            "orth/err_fro_mean": 0.0,
            "orth/err_fro_max": 0.0,
            "orth/err_fro_p95": 0.0,
        }

    x = torch.permute(basis, (1, 0, 2))  # [B, K, D]
    gram = torch.matmul(x, x.transpose(1, 2))
    eye = torch.eye(gram.shape[-1], device=gram.device, dtype=gram.dtype).unsqueeze(0)
    fro = torch.linalg.norm(gram - eye, dim=(1, 2), ord='fro')

    return {
        "orth/err_fro_mean": float(fro.mean().item()),
        "orth/err_fro_max": float(fro.max().item()),
        "orth/err_fro_p95": float(torch.quantile(fro, 0.95).item()),
    }


class OrthogonalLayer1D(nn.Module):
    """
    Orthogonalize expert outputs with Gram-Schmidt while exposing monitoring
    statistics aligned with the HH implementation.
    """

    def __init__(self, hh_canon_sign=True, hh_rank_tol=1e-6):
        super(OrthogonalLayer1D, self).__init__()
        self.hh_canon_sign = bool(hh_canon_sign)
        self.hh_rank_tol = float(hh_rank_tol)
        self.last_stats = {
            "orth/err_fro_mean": 0.0,
            "orth/err_fro_max": 0.0,
            "orth/err_fro_p95": 0.0,
            "hh/min_abs_diagR": 0.0,
            "hh/max_abs_diagR": 0.0,
            "hh/diagR_ratio": 0.0,
            "hh/rank_fail_rate": 0.0,
            "hh/canon_sign": float(self.hh_canon_sign),
        }

    def forward(self, x):
        """
        Arg:
            x: The parallel formatted input with shape [n_models, n_samples, dim]

        return:
            basis: Orthogonalized version of the input (x), same shape as input.
        """
        x1 = torch.transpose(x, 0, 1)  # [B, K, D]
        eps = torch.finfo(x1.dtype).eps if x1.is_floating_point() else 1e-12
        safe_tol = max(self.hh_rank_tol, eps)

        batch_size, n_models, dim = x1.shape
        basis_cols = []
        diag_vals = []

        for i in range(n_models):
            v = x1[:, i, :]  # [B, D]
            if basis_cols:
                basis_stack = torch.stack(basis_cols, dim=1)  # [B, i, D]
                proj_coeff = torch.matmul(v.unsqueeze(1), basis_stack.transpose(1, 2))  # [B,1,i]
                proj = torch.matmul(proj_coeff, basis_stack).squeeze(1)  # [B,D]
                w = v - proj
            else:
                w = v

            resid_norm = torch.linalg.norm(w, dim=1)  # [B]
            diag_vals.append(resid_norm)

            denom = resid_norm.unsqueeze(1).clamp_min(safe_tol)
            basis_cols.append(w / denom)

        basis = torch.stack(basis_cols, dim=1)  # [B,K,D]
        basis = torch.transpose(basis, 0, 1)    # [K,B,D]

        diag_tensor = torch.stack(diag_vals, dim=1)  # [B,K]
        abs_diag = diag_tensor.abs()
        min_abs = abs_diag.min(dim=1).values
        max_abs = abs_diag.max(dim=1).values
        ratio = max_abs / min_abs.clamp_min(safe_tol)
        rank_fail = (min_abs < self.hh_rank_tol).float()

        stats = orth_error_stats(basis)
        stats.update({
            "hh/min_abs_diagR": float(min_abs.mean().item()),
            "hh/max_abs_diagR": float(max_abs.mean().item()),
            "hh/diagR_ratio": float(ratio.mean().item()),
            "hh/rank_fail_rate": float(rank_fail.mean().item()),
            "hh/canon_sign": float(self.hh_canon_sign),
        })
        self.last_stats = stats
        return basis