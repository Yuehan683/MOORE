import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import moore.utils.mixture_layers as mixture_layers
from moore.utils.mixture_layers_svd import OrthogonalLayer1D_SVD


class MiniGridPPOMixtureSHNetwork_SVD(nn.Module):
    def __init__(self, input_shape,
                       output_shape,
                       n_features,
                       n_contexts=1,
                       n_experts=4,
                       orthogonal=True,
                       use_cuda=False,
                       task_encoder_bias=False,
                       **kwargs):

        super().__init__()

        self._n_input = input_shape
        self._n_output = output_shape[0]
        self._n_contexts = n_contexts
        self._orthogonal = orthogonal
        self._use_cuda = use_cuda

        n_input_channels = self._n_input[-1]

        self._task_encoder = nn.Linear(n_contexts, n_experts, bias=task_encoder_bias)

        # === 保持与原 networks_ppo.py 一致的 CNN ===
        cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.Flatten(),
        )

        with torch.no_grad():
            n_flatten = cnn(torch.zeros((1, 3, 7, 7)).float()).shape[1]

        # === 关键变化：OrthogonalLayer1D() -> OrthogonalLayer1D_SVD() ===
        if orthogonal:
            self.cnn = nn.Sequential(
                mixture_layers.InputLayer(n_models=n_experts),
                mixture_layers.ParallelLayer(cnn),
                OrthogonalLayer1D_SVD()
            )
        else:
            self.cnn = nn.Sequential(
                mixture_layers.InputLayer(n_models=n_experts),
                mixture_layers.ParallelLayer(cnn)
            )

        input_size = n_flatten + n_contexts

        self._output_head = nn.Sequential()
        if len(n_features) > 0:
            self._output_head.append(nn.Linear(input_size, n_features[0]))
            self._output_head.append(nn.Tanh())
            input_size = n_features[0]

        self._output_head.append(nn.Linear(input_size, self._n_output))

    def forward(self, state, c=None):
        if isinstance(c, int):
            c = torch.tensor([c])
        if isinstance(c, np.ndarray):
            c = torch.from_numpy(c)

        c = F.one_hot(c, num_classes=self._n_contexts)

        if self._use_cuda:
            c = c.cuda()

        w = self._task_encoder(c.float()).unsqueeze(1)

        features_cnn = self.cnn(state.float())            # [n_experts, B, dim]
        features_cnn = torch.permute(features_cnn, (1, 0, 2))  # [B, n_experts, dim]

        features_cnn = w @ features_cnn
        features_cnn = features_cnn.squeeze(1)

        features_cnn = torch.tanh(features_cnn)

        f = self._output_head(torch.cat((features_cnn, c.float()), dim=1))
        return f