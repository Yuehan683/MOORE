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

This version keeps the ORIGINAL GS forward path unchanged, while exposing
monitoring statistics for the GS diagnostics implementation.
"""

import torch
import torch.nn as nn
from copy import deepcopy

n_models_global = 5
agg_out_dim = 3


class InputLayer(nn.Module):
    """
       InputLayer structures the data in a parallel form ready to be consumed by
       the upcoming parallel layers.
    """

    def __init__(self, n_models):
        """
        Arg:
            n_models: number of individual models within the ensemble
        """
        super(InputLayer, self).__init__()
        self.n_models = n_models
        global n_models_global
        n_models_global = self.n_models

    def forward(self, x):
        """
        Arg
            x: input to the network as in standard deep neural networks.

        return:
            x_parallel: parallel form of x.
        """
        x_parallel = torch.unsqueeze(x, 0)
        x_parallel_next = torch.unsqueeze(x, 0)
        for _ in range(1, self.n_models):
            x_parallel = torch.cat((x_parallel, x_parallel_next), axis=0)

        return x_parallel


class ParallelLayer(nn.Module):
    """
        ParallelLayer creates a parallel layer from the structure of unit_model it receives.
    """

    def __init__(self, unit_model):
        """
        Arg:
            unit_model: specifies what computational module each unit of the parallel layer contains.
        """
        super(ParallelLayer, self).__init__()
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
            x: parallel input with shape [n_models, n_samples, dim] for fully connected layers.

        return:
            parallel_output: shape [n_models, n_samples, dim]
        """
        parallel_output = self.model_layers[0](x[0])
        parallel_output = torch.unsqueeze(parallel_output, 0)
        for i in range(1, self.n_models):
            next_layer = self.model_layers[i](x[i])
            next_layer = torch.unsqueeze(next_layer, 0)
            parallel_output = torch.cat((parallel_output, next_layer), 0)

        return parallel_output


def orth_error_stats(basis: torch.Tensor):
    """
    Compute core orthogonality diagnostics.
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
    Keep the original GS forward path unchanged, while exposing monitoring
    statistics for the GS diagnostics implementation.
    """

    def __init__(self, diag_canon_sign=True, diag_rank_tol=1e-6):
        super(OrthogonalLayer1D, self).__init__()
        self.diag_canon_sign = bool(diag_canon_sign)
        self.diag_rank_tol = float(diag_rank_tol)
        self.last_stats = {
            "orth/err_fro_mean": 0.0,
            "orth/err_fro_max": 0.0,
            "orth/err_fro_p95": 0.0,
            "diag/min_abs_resid": 0.0,
            "diag/max_abs_resid": 0.0,
            "diag/resid_ratio": 0.0,
            "diag/rank_fail_rate": 0.0,
            "diag/canon_sign": float(self.diag_canon_sign),
        }

    def forward(self, x):
        """
        Arg:
            x: The parallel formatted input with shape [n_models, n_samples, dim]

        return:
            basis: Orthogonalized version of the input (x), same shape as input.
        """
        x1 = torch.transpose(x, 0, 1)  # [B, K, D]

        # === ORIGINAL GS FORWARD (unchanged) ===
        first = x1[:, 0, :]  # [B, D]
        first_norm = torch.linalg.norm(first, dim=1)  # [B]
        basis = torch.unsqueeze(first / torch.unsqueeze(first_norm, 1), 1)  # [B, 1, D]

        # Only for diagnostics; do NOT feed back into forward.
        diag_vals = [first_norm]

        for i in range(1, x1.shape[1]):
            v = x1[:, i, :]
            v = torch.unsqueeze(v, 1)  # [B, 1, D]

            # Original GS projection/removal path
            w = v - torch.matmul(torch.matmul(v, torch.transpose(basis, 2, 1)), basis)

            w_norm = torch.linalg.norm(w, dim=2)  # [B, 1]
            diag_vals.append(w_norm.squeeze(1))

            # Original GS normalization path: NO clamp_min / NO safe_tol in forward
            wnorm = w / torch.unsqueeze(w_norm, 2)
            basis = torch.cat([basis, wnorm], axis=1)

        basis_out = torch.transpose(basis, 0, 1)  # [K, B, D]

        # === diagnostics only ===
        diag_tensor = torch.stack(diag_vals, dim=1)  # [B, K]
        abs_diag = diag_tensor.abs()
        min_abs = abs_diag.min(dim=1).values
        max_abs = abs_diag.max(dim=1).values

        # Use tolerance only for LOGGING to avoid divide-by-zero in metrics.
        eps = torch.finfo(abs_diag.dtype).eps if abs_diag.is_floating_point() else 1e-12
        ratio = max_abs / min_abs.clamp_min(max(self.diag_rank_tol, eps))
        rank_fail = (min_abs < self.diag_rank_tol).float()

        stats = orth_error_stats(basis_out)
        stats.update({
            "diag/min_abs_resid": float(min_abs.mean().item()),
            "diag/max_abs_resid": float(max_abs.mean().item()),
            "diag/resid_ratio": float(ratio.mean().item()),
            "diag/rank_fail_rate": float(rank_fail.mean().item()),
            "diag/canon_sign": float(self.diag_canon_sign),
        })
        self.last_stats = stats

        return basis_out
