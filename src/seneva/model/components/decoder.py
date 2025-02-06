# Copyright 2023 (c) David Juanwu Lu and Purdue Digital Twin Lab
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Variational decoder for the SeNeVA model."""
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.distributions as dist
import torch.nn as nn

from seneva.model.functional import (
    compute_circular_iou,
    kl_divergence_categorical,
    linear_gaussian_reconstruction_loss,
)
from seneva.model.layers import layer_norm, linear, variance_scaling

# Constants
TIKHONOV_REGULARIZATION = 0.5 / math.pi

# Type Aliases
_tensor_2_t = Tuple[torch.Tensor, torch.Tensor]


@dataclass
class GenerativeOutput:
    s_mean: torch.Tensor
    """torch.Tensor: The mean vector of the latent state distribution."""
    s_vars: torch.Tensor
    """torch.Tensor: The diagonal entries of the latent state covariance."""
    y_mean: torch.Tensor
    """torch.Tensor: The mean vector of the conditional emission."""
    y_vars: torch.Tensor
    """torch.Tensor: The diagonal entries of the conditional emission covar."""


class GenerativeNet(nn.Module):
    """Generative network for joint prior distribution.

    .. note::

        The generative network consists of several feed-forward neural
        networks for each of the random variables in the joint prior
        distribution. Here, y_t|s_t is modeled as a Gaussian distribution,
        s_t|x_t,z is modeled as a Bayesian mixture of Gaussians, and z is
        modeled as a Categorical random variable.
    """

    # ----------- public attributes ----------- #
    hidden_size: int
    """int: Hidden layer size of the generative network modules."""
    num_mixtures: int
    """int: Number of mixture components for the `p(s|x,z)` model."""
    horizon: int
    """int: Number of time steps in frames to predict."""

    # ----------- private attributes ----------- #
    _dynamic: nn.ModuleDict
    """nn.ModuleDict: A list of RNNs parameterizing latent Gaussians."""
    _emission_mean: nn.Linear
    """nn.Linear: A linear layer parameterizing emission means."""
    _emission_vars: nn.Sequential
    """nn.Sequential: An MLP parameterizing emission log-variances."""
    _init_states: nn.Parameter
    """nn.Parameter: Learnable initial latent states."""

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_mixtures: int,
        horizon: int,
        output_dim: int = 3,
    ) -> None:
        super().__init__()

        # save arguments
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_mixtures = num_mixtures
        self.horizon = horizon
        self.output_dim = output_dim

        # build neural network modules
        self._dynamic = nn.ModuleDict(
            {
                "input_head": nn.Sequential(
                    linear(input_dim, hidden_size),
                    layer_norm(normalized_shape=hidden_size),
                    nn.SiLU(),
                ),
                "lstm": nn.ModuleList(
                    [
                        nn.LSTM(
                            input_size=hidden_size + input_dim,
                            hidden_size=hidden_size,
                            batch_first=True,
                            bidirectional=False,
                            num_layers=2,
                        )
                        for _ in range(num_mixtures)
                    ]
                ),
                "mean_head": linear(
                    hidden_size, hidden_size, init_scale=1e-10
                ),
                "vars_head": nn.Sequential(
                    linear(hidden_size, hidden_size, init_scale=1e-10),
                    nn.Softplus(),
                ),
            }
        )
        self._emission_mean = linear(
            input_dim + hidden_size, output_dim, init_scale=1e-10
        )

        # Initialize the learnable parameters
        self._emission_vars = nn.Parameter(
            torch.zeros(output_dim), requires_grad=True
        )
        self._init_states = nn.Parameter(
            torch.randn(2 * num_mixtures, 2 * hidden_size), requires_grad=True
        )

        # Initialize the parameters
        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> GenerativeOutput:
        """Forward pass and get the parameters in the generative process.

        Args:
            x (torch.Tensor): Output from the encoder of shape `[B, D]`.

        Returns:
            GenerativeOutput: Parameters the generative process.
        """
        assert x.ndim == 2 and x.size(-1) == self.input_dim, (
            f"Invalid input shape {list(x.shape)}, "
            f"expect `[B, {self.input_dim}]`."
        )
        s_mean, s_vars = self.forward_s(x=x)
        y_mean, y_vars = self.forward_y(s=s_mean, x=x)

        return GenerativeOutput(
            s_mean=s_mean, s_vars=s_vars, y_mean=y_mean, y_vars=y_vars
        )

    def forward_s(self, x: torch.Tensor) -> _tensor_2_t:
        """Returns the mean and variances of the latent state distribution.

        Args:
            x (torch.Tensor): Output from the encoder of shape `[B, D]`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The mean and variance vectors
                of the latent state distribution.
        """
        shape = [x.size(0), self.num_mixtures, self.horizon, self.hidden_size]
        s_mean = torch.zeros(shape, device=x.device, dtype=x.dtype)
        s_vars = torch.ones_like(s_mean)

        init_state = self._dynamic["input_head"].forward(x)
        for i in range(self.num_mixtures):
            rnn: nn.LSTM = self._dynamic["lstm"][i]
            h, c = self._init_states[i : i + 2].chunk(2, dim=-1)
            hx = (
                h.unsqueeze(-2).expand(-1, x.size(0), -1).contiguous(),
                c.unsqueeze(-2).expand(-1, x.size(0), -1).contiguous(),
            )

            # forward pass the initial state
            s_mean[:, i, 0] = self._dynamic["mean_head"].forward(init_state)
            s_vars[:, i, 0] = torch.add(
                self._dynamic["vars_head"].forward(init_state),
                torch.finfo(x.dtype).eps,
            )

            # forward pass the LSTM with free-running
            inp = torch.cat([s_mean[:, i, 0], x], dim=-1)
            for t in range(1, self.horizon):
                inp = inp.unsqueeze(-2).contiguous()
                out, hx = rnn.forward(inp, hx=hx)
                inp = out.squeeze(-2)
                s_mean[:, i, t] = self._dynamic["mean_head"].forward(inp)
                s_vars[:, i, t] = torch.add(
                    self._dynamic["vars_head"].forward(inp),
                    torch.finfo(x.dtype).eps,
                )
                inp = torch.cat([s_mean[:, i, t], x], dim=-1)

        return s_mean, s_vars

    def forward_y(self, s: torch.Tensor, x: torch.Tensor) -> _tensor_2_t:
        """Returns the mean and variance of the emission distribution.

        Args:
            s (torch.Tensor): Latent state tensor of shape `[B, *, H, D]`.
            x (torch.Tensor): Output from the encoder of shape `[B, D]`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The mean and variance vectors
                of the emission distribution.
        """
        # Forward pass the mean head
        if s.ndim == 4:
            x = x[:, None, None, :].expand(-1, s.size(1), s.size(2), -1)
        elif s.ndim == 3:
            x = x[:, None, :].expand(-1, s.size(1), -1)
        else:
            raise ValueError("Invalid `s` shape, expect 3 or 4 dimensions.")
        y_mean = self.emission_mean.forward(torch.cat([s, x], dim=-1))

        y_vars = torch.square(self.emission_vars) + TIKHONOV_REGULARIZATION
        if y_mean.ndim == 4:
            y_vars = y_vars.unsqueeze(-2).unsqueeze(-2).expand_as(y_mean)
        elif y_mean.ndim == 3:
            y_vars = y_vars.unsqueeze(-2).expand_as(y_mean)
        else:
            raise ValueError(
                "Invalid `y_mean` shape, expect 3 or 4 dimensions."
            )

        return y_mean, y_vars

    def predictive(
        self, x: torch.Tensor, cumulate: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns the mean and covariance of the predictive distribution.

        Args:
            x (torch.Tensor): Input feature tensor of shape
                :math:`[batch_size, x_dims]`.
            cumulate (bool, optional): Flags to cumulate the predictions.
                Defaults to `False`.

        Returns:
            Tuple: The mean and covariance matrix of the marginal distribution.
        """
        assert x.ndim == 2 and x.size(-1) == self.input_dim, (
            f"Invalid input shape {list(x.shape)}, "
            f"expect `[B, {self.input_dim}]`."
        )
        params = self.forward(x=x)
        y_means = params.y_mean

        weight = self.emission_mean.weight[..., 0 : self.hidden_size]
        y_covars = torch.matmul(
            weight.matmul(torch.diag_embed(params.s_vars)),
            weight.transpose(-2, -1),
        )
        y_covars = torch.add(torch.diag_embed(params.y_vars), y_covars)
        if cumulate:
            y_means = y_means.cumsum(dim=-2)
            y_covars = y_covars.cumsum(dim=-3)

        return y_means, y_covars

    def reset_parameters(self) -> None:
        """Reset the parameters of the generative network."""
        variance_scaling(self._init_states)
        for lstm in self._dynamic["lstm"]:
            assert isinstance(lstm, nn.LSTM)
            for name, param in lstm.named_parameters():
                if "weight_ih" in name:
                    variance_scaling(param)
                elif "weight_hh" in name:
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    nn.init.zeros_(param)
                else:
                    raise ValueError(
                        f"Initialization for {name} is undefined."
                    )

    @property
    def emission_mean(self) -> nn.Linear:
        """nn.Linear: The emission layers for the generative network."""
        return self._emission_mean

    @property
    def emission_vars(self) -> nn.Sequential:
        """nn.Sequential: The emission log-variance layer for the emission."""
        return self._emission_vars


class InferenceNet(nn.Module):
    """Inference network for join posterior distribution.

    .. note::

        The inference network consists of several feed-forward neural networks
        for each of the random variables in the joint posterior distribution.
        The key network is a RNN that parameterizes the smoothing distribution
        of the latent states given the observations and the input.
    """

    # ----------- public attributes ----------- #
    hidden_size: int
    """int: Hidden layer size of the inference network modules."""
    horizon: int
    """int: Number of time steps in frames to predict."""
    num_mixtures: int
    """int: Number of mixture components in the generative model."""

    # ----------- private attributes ----------- #
    _smooth: nn.ModuleDict
    """nn.ModuleDict: Modules for the smooth distribution of latent states."""
    _z_proxy: nn.Sequential
    """nn.Sequential: Network approximates the distribution `p(z|x)`."""
    _init_states: nn.Parameter
    """nn.Parameter: Learnable initial latent states."""

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        horizon: int,
        num_mixtures: int,
        output_dim: int = 3,
    ) -> None:
        super().__init__()

        # save arguments
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.horizon = horizon
        self.num_mixtures = num_mixtures
        self.output_dim = output_dim

        # build neural network modules
        self._smooth = nn.ModuleDict(
            {
                "lstm": nn.LSTM(
                    input_size=input_dim + output_dim,
                    hidden_size=hidden_size,
                    batch_first=True,
                    bidirectional=True,
                ),
                "mean_head": linear(
                    2 * hidden_size, hidden_size, init_scale=1e-10
                ),
                "vars_head": nn.Sequential(
                    linear(2 * hidden_size, hidden_size, init_scale=1e-10),
                    nn.Softplus(),
                ),
            }
        )
        self._z_proxy = nn.Sequential(
            linear(input_dim, 4 * hidden_size),
            nn.SiLU(),
            linear(4 * hidden_size, 2 * hidden_size),
            nn.SiLU(),
            linear(2 * hidden_size, num_mixtures),
        )

        # Initialize the learnable initial cell state
        self._init_states = nn.Parameter(
            torch.randn(2, 2 * hidden_size), requires_grad=True
        )

        # Initialize the parameters
        self.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        generator: GenerativeNet,
        train_variance: bool = False,
    ) -> Dict[str, torch.Tensor]:
        if not self.training:
            raise RuntimeError(
                "Inference network is only used during training"
            )
        assert y.ndim == 3 and y.size(-1) == self.output_dim, (
            f"Invalid input shape {list(y.shape)}, "
            f"expect `[B, H, {self.output_dim}]`."
        )
        y_diff = torch.cat([y[..., 0:1, :], torch.diff(y, n=1, dim=1)], dim=1)

        # step 1: construct posterior distributions
        x = x.float().contiguous()
        q_s_mean, q_s_vars = self.forward_s(x=x, y=y)
        p_s_mean, p_s_vars = generator.forward_s(x=x)
        y_mean, y_vars = generator.forward_y(s=q_s_mean, x=x)
        if not train_variance:
            p_s_vars = torch.ones_like(p_s_vars, requires_grad=False)
            y_vars = None

        # step 2: compute the reconstruction loss
        reconstruction_loss = linear_gaussian_reconstruction_loss(
            y=y_diff,
            x_vars=q_s_vars,
            y_mean=y_mean,
            y_vars=y_vars,
            weight=generator.emission_mean.weight[:, 0 : self.hidden_size],
        ).sum(dim=-1)

        # step 3: compute z-posteriors
        q_s_mean = q_s_mean.unsqueeze(1).expand(-1, self.num_mixtures, -1, -1)
        q_s_vars = q_s_vars.unsqueeze(1).expand(-1, self.num_mixtures, -1, -1)
        with torch.no_grad():
            q_z = self.evaluate_z(
                prior_mean=p_s_mean,
                prior_vars=p_s_vars,
                posterior_mean=q_s_mean,
                posterior_vars=q_s_vars,
            )
            kl_div_z = kl_divergence_categorical(
                log_p=torch.log(q_z + 1e-10),
                log_q=torch.zeros_like(q_z) - math.log(self.num_mixtures),
            )
            assert torch.sum(q_z, dim=-1).allclose(torch.tensor(1.0))

        # step 4: compute the kl-divergence of latent mixture variables
        p_dist = dist.MultivariateNormal(
            loc=p_s_mean,
            covariance_matrix=torch.diag_embed(p_s_vars),
        )
        q_dist = dist.MultivariateNormal(
            loc=q_s_mean,
            covariance_matrix=torch.diag_embed(q_s_vars),
        )
        kl_div_s = dist.kl.kl_divergence(p=q_dist, q=p_dist).sum(dim=-1)
        kl_div_s = torch.sum(q_z * kl_div_s, dim=-1)

        # optional: return the mean ADE for debugging recognition performance
        with torch.no_grad():
            mean_diff = torch.norm(
                y[..., 0:2] - y_mean.cumsum(dim=-2)[..., 0:2],
                p=2,
                dim=-1,
            ).mean(dim=-1)

        return {
            "mean_diff": mean_diff,
            "reconstruction_loss": reconstruction_loss,
            "kl_div_s": kl_div_s,
            "kl_div_z": kl_div_z,
            "z_logits": torch.log(q_z + 1e-10),
        }

    def forward_z(self, x: torch.Tensor) -> torch.Tensor:
        return self._z_proxy.forward(x)

    def forward_s(self, x: torch.Tensor, y: torch.Tensor) -> _tensor_2_t:
        """Forward pass the recognition LSTM network with teacher forcing.

        Args:
            x (torch.Tensor): Output from the encoder of shape `[B, D]`.
            y (torch.Tensor): Future observation of shape `[B, H, O]`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The mean and variance vectors.
        """
        assert y.ndim == 3 and y.size(-1) == self.output_dim, (
            f"Invalid input shapes:  y.shape = {list(y.shape)}, "
            f"expect `[B, H, {self.output_dim}]`."
        )

        rnn: nn.LSTM = self._smooth["lstm"]
        h, c = self._init_states.chunk(2, dim=-1)
        hx = (
            h.unsqueeze(-2).expand(-1, x.size(0), -1).contiguous(),
            c.unsqueeze(-2).expand(-1, x.size(0), -1).contiguous(),
        )

        inp = torch.cat([x.unsqueeze(-2).expand(-1, y.size(-2), -1), y], -1)
        out, hx = rnn.forward(input=inp, hx=hx)
        s_mean = self._smooth["mean_head"].forward(out)
        s_vars = torch.add(
            self._smooth["vars_head"].forward(out),
            torch.finfo(x.dtype).eps,
        )

        return s_mean, s_vars

    @torch.jit.ignore
    def evaluate_z(
        self,
        prior_mean: torch.Tensor,
        prior_vars: torch.Tensor,
        posterior_mean: torch.Tensor,
        posterior_vars: torch.Tensor,
        prior_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns the optimal variational distribution for latent assignments.

        Args:
            prior_mean (torch.Tensor): The mean of the prior distribution.
            prior_vars (torch.Tensor): The variances of the prior distribution.
            posterior_mean (torch.Tensor): The posterior mean vector.
            posterior_vars (torch.Tensor): The posterior variance vector.
            prior_logits (Optional[torch.Tensor], optional): The prior logits.
                Defaults to `None`.

        Returns:
            torch.Tensor: The optimal variational assignment log-probabilities.
        """
        if prior_logits is None:
            # if not provided, use uniform prior
            prior_logits = -math.log(self.num_mixtures)

        diff = prior_mean - posterior_mean
        logits = -0.5 * torch.sum(
            diff.pow(2).div(prior_vars).sum(dim=-1)
            + posterior_vars.div(prior_vars).sum(dim=-1)
            + prior_vars.div(posterior_vars).log().sum(dim=-1),
            dim=-1,
        )
        # NOTE: for numerical stability and avoid dominating terms
        logits = logits - torch.max(logits, dim=-1, keepdim=True).values

        output = torch.divide(
            torch.exp(logits + prior_logits),
            torch.sum(torch.exp(logits + prior_logits), dim=-1, keepdim=True),
        )

        return output

    def reset_parameters(self) -> None:
        """Reset the parameters of the inference network."""
        variance_scaling(self._init_states)
        lstm = self._smooth["lstm"]
        assert isinstance(lstm, nn.LSTM)
        for name, param in lstm.named_parameters():
            if "weight_ih" in name:
                variance_scaling(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
            else:
                raise ValueError(f"Initialization for {name} is undefined.")


class Decoder(nn.Module):
    """Variational decoder network for the SeNeVA model."""

    alpha: float
    """float: Hyperparameter for the focal loss."""
    gamma: float
    """float: Hyperparameter for focusing strength in focal loss."""
    hidden_size: int
    """int: Hidden layer size of the network modules."""

    _generative_net: GenerativeNet
    """GenerativeNet: Generative network for joint prior distribution."""
    _inference_net: InferenceNet
    """InferenceNet: Inference network for joint posterior distribution."""

    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        num_mixtures: int,
        horizon: int,
        output_dim: int = 3,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ) -> None:
        super().__init__()

        # save arguments
        self.alpha = alpha
        self.gamma = gamma
        self.hidden_size = hidden_size

        # build network modules
        self._generative_net = GenerativeNet(
            input_dim=input_dim,
            hidden_size=hidden_size,
            num_mixtures=num_mixtures,
            horizon=horizon,
            output_dim=output_dim,
        )
        self._inference_net = InferenceNet(
            input_dim=input_dim,
            hidden_size=hidden_size,
            horizon=horizon,
            num_mixtures=num_mixtures,
            output_dim=output_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        sampling: bool = True,
        iou_radius: float = 1.4,
        iou_threshold: float = 0.0,
        num_modals: int = 6,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of the variational decoder network.

        Args:
            x (torch.Tensor): Output from the encoder of shape `[B, D]`.
            sampling (bool, optional): Flags to enable sampling. If `False`,
                use distribution means. Defaults to `False`.
            iou_radius (float, optional): Radius of the circular buffer for
                the selected samples in meters. Defaults to :math:`1.4`.
            iou_threshold (float): Threshold for the IoU between the selected
                samples and the target samples in percents. Defaults to `0.0`.
            num_modals (int, optional): Number of modalities to sample.
                Defaults to :math:`6`.
        """
        num_batch: int = x.size(0)
        res = {}

        # for inference, use the generative network to compute the output
        with torch.no_grad():
            mixture_probs = self.inference_net.forward_z(x=x).softmax(-1)
            y_means, y_covars = self.generative_net.predictive(x, True)

            if sampling:
                predictions, probabilities = [], []
                for i in range(num_batch):
                    # select the top-k components
                    sorted_probs, indices = torch.sort(
                        mixture_probs[i], descending=True
                    )
                    cumulated_probs = torch.cumsum(sorted_probs, dim=-1)
                    slices = indices[cumulated_probs <= 0.90]
                    if slices.size(0) == 0:
                        slices = indices[0:1]
                    # slices = indices[0:1]
                    probs = sorted_probs[slices]
                    means = y_means[i, slices]
                    covars = y_covars[i, slices]

                    # step 1: generate samples at the last time step
                    yT_dist = dist.MultivariateNormal(
                        loc=means[..., -1, 0:2],
                        covariance_matrix=covars[..., -1, 0:2, 0:2],
                    )
                    goal_samples = yT_dist.sample((100,)).view(-1, 2)
                    goal_samples = torch.cat(
                        [goal_samples, y_means[i, :, -1, 0:2].view(-1, 2)]
                    )

                    # step 2: compute mixture probabilities
                    sample_probs: torch.Tensor = yT_dist.log_prob(
                        goal_samples.unsqueeze(1)
                    )
                    sample_probs = torch.exp(sample_probs).view(
                        goal_samples.size(0), *means.shape[:-2]
                    )
                    sample_probs = torch.sum(
                        probs.unsqueeze(0) * sample_probs, dim=-1
                    )

                    # step 3: apply NMS sampling
                    cand, cand_probs = [], []
                    valid_mask = torch.ones_like(sample_probs)
                    while len(cand) < num_modals:
                        # select the sample with the highest probability
                        idx = torch.argmax(
                            sample_probs * valid_mask.float(), dim=-1
                        )
                        mask = torch.zeros(
                            *sample_probs.shape,
                            device=x.device,
                            dtype=torch.bool,
                        )
                        mask[idx] = True
                        selected_sampels = goal_samples[mask]
                        cand.append(selected_sampels.unsqueeze(-2))
                        cand_probs.append(sample_probs[mask].unsqueeze(-1))

                        # compute IoU between the selected samples and others
                        iou = compute_circular_iou(
                            selected_samples=selected_sampels.unsqueeze(0),
                            other_samples=goal_samples.unsqueeze(0),
                            selected_radius=iou_radius,
                            other_radius=iou_radius,
                        )[0]
                        iou_mask = iou > iou_threshold

                        # update the valid mask
                        valid_mask[iou_mask] = -1.0
                        valid_mask[mask] = -1.0
                    cand = torch.cat(cand, dim=-2)
                    cand_probs = torch.cat(cand_probs, dim=-1).softmax(dim=-1)

                    # step 2: apply homogeneous reparameterization sampling
                    mixture_mean = torch.sum(
                        probs.view(*means.shape[:-2], 1, 1) * means,
                        dim=-3,
                        keepdim=True,
                    )
                    diff = means - mixture_mean
                    mixture_covar = torch.sum(
                        probs.view(*covars.shape[:-3], 1, 1, 1)
                        * (
                            covars
                            + diff.unsqueeze(-1).matmul(diff.unsqueeze(-2))
                        ),
                        dim=-4,
                        keepdim=True,
                    )
                    mixture_mean.squeeze_(dim=-3)
                    mixture_covar.squeeze_(dim=-4)

                    # compute the Cholesky decomposition and determine p
                    mixture_L: torch.Tensor = torch.linalg.cholesky(
                        mixture_covar
                    )
                    p = torch.matmul(
                        torch.inverse(mixture_L[..., -1, 0:2, 0:2]).unsqueeze(
                            0
                        ),
                        torch.unsqueeze(
                            cand.permute(1, 0, 2)
                            - mixture_mean[..., -1, 0:2].unsqueeze(0),
                            dim=-1,
                        ),
                    )

                    # interpolate changing parameters
                    steps = torch.linspace(
                        0, 1, mixture_mean.size(-2) + 1, device=x.device
                    ).view(
                        [1] * (p.ndim - 2) + [mixture_mean.size(-2) + 1, 1, 1]
                    )
                    p = p.unsqueeze(-3) * steps[..., 1:, :, :]
                    rest_cand = torch.matmul(
                        mixture_L[..., :-1, 0:2, 0:2].unsqueeze(0),
                        p[..., :-1, 0:2, 0:1],
                    ).squeeze(-1)
                    rest_cand = rest_cand + mixture_mean[None, ..., :-1, 0:2]
                    rest_cand = rest_cand.transpose(0, 1).contiguous()

                    # concatenate the samples
                    prediction = torch.cat([rest_cand, cand.unsqueeze(-2)], -2)
                    predictions.append(prediction)
                    mixture_dist = dist.MultivariateNormal(
                        loc=mixture_mean, covariance_matrix=mixture_covar
                    )
                    prob: torch.Tensor = mixture_dist.log_prob(
                        value=prediction.transpose(0, 1)
                    )
                    probabilities.append(
                        prob.sum(dim=-1).softmax(dim=-1).transpose(0, 1)
                    )
                res["predictions"] = torch.cat(predictions, dim=0)
                res["probabilities"] = torch.cat(probabilities, dim=0)
            else:
                res["predictions"] = y_means
                res["probabilities"] = mixture_probs

        return res

    @property
    def generative_net(self) -> GenerativeNet:
        """GenerativeNet: The generative network."""
        return self._generative_net

    @property
    def inference_net(self) -> InferenceNet:
        """InferenceNet: The inference network."""
        return self._inference_net
