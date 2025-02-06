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
"""Sequential Neural Varaitional Agent (SeNeVA) model."""
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torchmetrics import MeanMetric, Metric, MetricCollection, MinMetric

from seneva.data.base import PolylineData
from seneva.model.components.decoder import Decoder
from seneva.model.components.encoder import Encoder
from seneva.utils.logging import get_logger

# Constants
LOGGER = get_logger(__name__)


class SeNeVANetwork(nn.Module):
    """Sequential Neural Varaitional Agent (SeNeVA) model."""

    # ----------- public attributes ----------- #
    predict_actions: bool
    """bool: Whether to predict actions or waypoints."""

    # ----------- private attributes ----------- #
    _encoder: Encoder
    """Encoder: Traffic feature encoder module."""
    _decoder: Decoder
    """Decoder: Probabilistic state-space model for action generation."""

    def __init__(
        self,
        # encoder arguments
        map_in_features: int,
        motion_in_features: int,
        map_hidden_size: int,
        motion_hidden_size: int,
        # decoder arguments
        decoder_hidden_size: int,
        num_components: int,
        output_dim: int = 2,
        # default arguments
        dropout: float = 0.1,
        encoder_num_heads: int = 8,
        prediction_horizon: int = 30,
        num_subgraph_layers: int = 3,
        num_global_layers: int = 3,
        alpha: float = 0.25,
        gamma: float = 0.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim

        # build network modules
        self._encoder = Encoder(
            map_in_dims=map_in_features,
            map_hidden_size=map_hidden_size,
            map_num_layers=num_subgraph_layers,
            motion_in_dims=motion_in_features,
            motion_hidden_size=motion_hidden_size,
            motion_num_layers=num_subgraph_layers,
            dropout=dropout,
            num_heads=encoder_num_heads,
            num_global_layers=num_global_layers,
        )
        self._decoder = Decoder(
            input_dim=self.encoder.out_feature,
            hidden_size=decoder_hidden_size,
            num_mixtures=num_components,
            horizon=prediction_horizon,
            output_dim=output_dim,
            alpha=alpha,
            gamma=gamma,
        )

    def forward(
        self,
        data: PolylineData,
        num_modals: int = 6,
        iou_radius: float = 1.4,
        iou_threshold: float = 0.0,
        sampling: bool = False,
    ) -> Dict[str, torch.Tensor]:
        # forward and obtain current state
        x = self._encoder.forward(data)
        with torch.no_grad():
            trajectory = data.get("y_motion", None)

        # forward and obtain prediction
        decoder_output = self._decoder.forward(
            x=x,
            num_modals=num_modals,
            iou_radius=iou_radius,
            iou_threshold=iou_threshold,
            sampling=sampling,
        )
        predictions = decoder_output["predictions"]
        scores = decoder_output["probabilities"]
        predictions = torch.cat(
            [predictions, torch.zeros_like(predictions[..., 0:1])], dim=-1
        )

        # return network output
        return {
            **{k: v for (k, v) in decoder_output.items() if "loss" in k},
            "predictions": predictions,
            "probabilities": scores,
            "target": trajectory,
            "batch": data.get("y_motion_batch"),
        }

    def fit_bayes(
        self,
        data: PolylineData,
        kl_factor: float,
        train_variance: bool = False,
    ) -> Dict[str, torch.Tensor]:
        # prepare target data
        y = data.y_motion[..., 0 : self.output_dim]  # shape: (B, H, O)

        x = self.encoder.forward(data=data)
        outputs = self.decoder.inference_net.forward(
            x=x,
            y=y,
            generator=self.decoder.generative_net,
            train_variance=train_variance,
        )
        reconstruction_loss = outputs["reconstruction_loss"]
        reconstruction_loss = reconstruction_loss.mean()
        kl_div_s = outputs["kl_div_s"].mean()
        kl_div_z = outputs["kl_div_z"].mean()
        # variance_loss = outputs["variance_loss"].mean()
        z_logits = outputs["z_logits"]
        loss = reconstruction_loss + kl_factor * kl_div_s

        return {
            "loss": loss,
            "mean_diff": outputs["mean_diff"].mean().item(),
            "reconstruction_loss": reconstruction_loss.item(),
            "kl_loss_s": kl_div_s.mean().item(),
            "kl_loss_z": kl_div_z.mean().item(),
            "z_logits": z_logits,
            "entropy": torch.mean(
                -torch.sum(torch.exp(z_logits) * z_logits, dim=-1)
            ),
        }

    def fit_proxy(self, data: PolylineData) -> Dict[str, torch.Tensor]:
        # prepare target data
        y = data.y_motion[..., 0 : self.output_dim]  # shape: (B, H, O)

        with torch.no_grad():
            x = self.encoder.forward(data=data)
            p_s_mean, p_s_vars = self.decoder.generative_net.forward_s(x=x)
            q_s_mean, q_s_vars = self.decoder.inference_net.forward_s(x=x, y=y)
            q_s_mean = q_s_mean.unsqueeze(1).expand_as(p_s_mean)
            q_s_vars = q_s_vars.unsqueeze(1).expand_as(p_s_vars)
            target = self.decoder.inference_net.evaluate_z(
                prior_mean=p_s_mean,
                prior_vars=p_s_vars,
                posterior_mean=q_s_mean,
                posterior_vars=q_s_vars,
            )
        z_hat = self.decoder.inference_net.forward_z(x=x)
        cross_entropy = nn.functional.cross_entropy(
            input=z_hat, target=target, reduction="none"
        )
        pt = torch.exp(-cross_entropy)
        loss = torch.mean(
            self.decoder.alpha * (1 - pt) ** self.decoder.gamma * cross_entropy
        )
        return {"loss": loss, "z_proxy_loss": loss.item()}

    @property
    def encoder(self) -> Encoder:
        """Encoder: Traffic feature encoder module."""
        return self._encoder

    @property
    def decoder(self) -> Decoder:
        """Decoder: Probabilistic state-space model for action generation."""
        return self._decoder


class SeNeVALightningModule(LightningModule):
    """Lightning module wrapper for the SeNeVA model."""

    network: SeNeVANetwork
    """SeNeVANetwork: SeNeVA model network."""

    def __init__(
        self,
        network_kwargs: Dict[str, Any],
        optimizer_kwargs: Dict[str, Any],
        metric_collection: Optional[
            Union[MetricCollection, Dict[str, Metric]]
        ] = None,
        monitor: str = "val/MR_best",
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        if isinstance(metric_collection, dict):
            metric_collection = MetricCollection(metric_collection)

        # switch to manual optimization
        self.automatic_optimization = False

        # save arguments
        self.save_hyperparameters(ignore=["metric_collection", "monitor"])
        self.network = SeNeVANetwork(**network_kwargs)
        self.train_steps: int = 0

        # initialize train loss trackers
        self.train_losses = MetricCollection(
            metrics={
                "loss": MeanMetric(),
                "mean_diff": MeanMetric(),
                "reconstruction_loss": MeanMetric(),
                "kl_loss_z": MeanMetric(),
                "kl_loss_s": MeanMetric(),
                "z_proxy_loss": MeanMetric(),
            }
        )

        if metric_collection is not None:
            assert isinstance(metric_collection, MetricCollection), TypeError(
                f"Invalid metric collection type. Expected {MetricCollection},"
                f" but got {type(metric_collection)}."
            )
            # metric trackers for evaluation
            self.metric_names = list(metric_collection.keys())
            self.train_metrics = metric_collection.clone(prefix="train/")
            self.val_metrics = metric_collection.clone(prefix="val/")
            self.test_metrics = metric_collection.clone(prefix="test/")

            # best tracker for tracking best validation score so far
            assert monitor in self.val_metrics, f"Invalid monitor: {monitor}."
            self.monitor = monitor
            self._monitor_agg = MinMetric()

    def forward(
        self, data: PolylineData, *args, **kwargs
    ) -> Dict[str, torch.Tensor]:
        return self.network.forward(data, *args, **kwargs)

    def on_train_start(self) -> None:
        self.train_losses.reset()
        self.train_metrics.reset()
        # NOTE: by default lightning executes validation step sanity checks
        # before training starts, so it's worth to make sure that the
        # validation metrics don't store results from these checks
        self.val_metrics.reset()

    def training_step(self, data: PolylineData) -> STEP_OUTPUT:
        kl_factor = len(data) / len(self.trainer.train_dataloader.dataset)

        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()

        total_loss = 0.0
        losses = {}
        # train the bayesian model
        self.network.zero_grad()
        output = self.network.fit_bayes(
            data=data,
            kl_factor=kl_factor,
            train_variance=self.train_steps
            > 0.5 * self.hparams.optimizer_kwargs.get("max_steps", 100000),
        )
        output["loss"].backward()
        optimizer.step()
        losses.update(output)
        total_loss += output["loss"].item()
        self.log(
            "status/posterior_entropy",
            output["entropy"],
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

        # train the proxy model
        if self.train_steps > self.hparams.optimizer_kwargs.get(
            "max_steps", 100000
        ):
            self.network.zero_grad()
            output = self.network.fit_proxy(data=data)
            output["loss"].backward()
            optimizer.step()
            losses.update(output)
            total_loss += output["loss"].item()

        # log the current training learning rate
        for i, param_group in enumerate(optimizer.param_groups):
            self.log(
                f"lr/param_group_{i+1}",
                param_group["lr"],
                on_step=True,
                on_epoch=False,
                sync_dist=True,
            )

        # update and log loss trackers
        losses["loss"] = total_loss
        for key, value in losses.items():
            if key in self.train_losses:
                self.train_losses[key](value)
                self.log(
                    f"train/{key}",
                    value,
                    on_step=True,
                    on_epoch=True,
                    sync_dist=True,
                    prog_bar=True,
                )

        # update metrics
        self.eval()
        with torch.no_grad():
            output = self.network.forward(data=data)
            for name, metric in self.train_metrics.items():
                metric(
                    preds=output["predictions"][..., 0:3],
                    target=output["target"][..., 0:3],
                    anchor=data.anchor,
                    batch=output["batch"],
                )
                self.log(
                    name,
                    metric,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )
        self.train()
        self.log("status/kl_factor", kl_factor, on_step=True)

        self.train_steps += 1
        scheduler.step()

    def on_validation_start(self) -> None:
        self.val_metrics.reset()

    def validation_step(
        self, data: PolylineData, *args: Any, **kwargs: Any
    ) -> None:
        output = self.forward(data=data)

        # update and log validation metrics
        for name, metric in self.val_metrics.items():
            metric(
                preds=output["predictions"][..., 0:3],
                target=output["target"][..., 0:3],
                anchor=data.anchor,
                batch=output["batch"],
            )
            self.log(
                name,
                metric,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )

    def on_validation_epoch_end(self) -> None:
        # update and log the best validation score tracker
        monitor = self.val_metrics[self.monitor].compute()
        self._monitor_agg(monitor)
        self.log(
            f"val/{self.monitor}_best",
            self._monitor_agg.compute(),
            sync_dist=True,
            prog_bar=True,
        )

    def on_test_start(self) -> None:
        self.test_metrics.reset()

    def test_step(self, data: PolylineData, *args: Any, **kwargs: Any) -> None:
        output = self.forward(data=data)

        # update and log test metrics
        for name, metric in self.test_metrics.items():
            metric(
                preds=output["predictions"][..., 0:3],
                target=output["target"][..., 0:3],
                anchor=data.anchor,
                batch=output["batch"],
            )
            self.log(
                name,
                metric,
                on_step=True,
                on_epoch=True,
                sync_dist=True,
            )

    def configure_optimizers(self) -> Any:
        # NOTE: create parameter groups and only apply weight decay to
        # weights in linear or convolution layers. See:
        # https://discuss.pytorch.org/t/weight-decay-only-for-weights-of-nn-linear-and-nn-conv/114348/6
        weight_decay = self.hparams.optimizer_kwargs.get("weight_decay", 0.0)
        weight_decay_params, other_params = set(), set()
        for mn, m in self.named_modules():
            for pn, _ in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if "bias" in pn:
                    # all biases will not be decayed
                    other_params.add(fpn)
                elif "weight" in pn and isinstance(
                    m, (nn.Linear, nn.MultiheadAttention, nn.LSTM)
                ):
                    # weights of whitelist modules will be weight decayed
                    weight_decay_params.add(fpn)
                elif "weight" in pn and isinstance(
                    m,
                    (
                        nn.LayerNorm,
                        nn.BatchNorm1d,
                        nn.BatchNorm2d,
                        nn.BatchNorm3d,
                        nn.Embedding,
                    ),
                ):
                    # weights of blacklist modules will NOT be weight decayed
                    other_params.add(fpn)
        other_params.add("network._decoder._generative_net._emission_vars")
        other_params.add("network._decoder._generative_net._init_states")
        other_params.add("network._decoder._inference_net._init_states")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = weight_decay_params & other_params
        union_params = weight_decay_params | other_params
        if not len(inter_params) == 0:
            raise RuntimeError(
                f"Parameters {str(inter_params):s}"
                "appear both weight_decay_params and other_params."
            )
        if len(param_dict.keys() - union_params) != 0:
            raise RuntimeError(
                f"Parameters {str(param_dict.keys() - union_params):s} "
                "were not separated into weight_decay_params or other_params."
            )
        param_groups = [
            {
                "params": [
                    param_dict[pn] for pn in sorted(list(weight_decay_params))
                ],
                "lr": self.hparams.optimizer_kwargs.get("lr", 1e-3),
                "betas": self.hparams.optimizer_kwargs.get(
                    "betas", (0.9, 0.999)
                ),
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    param_dict[pn] for pn in sorted(list(other_params))
                ],
                "lr": self.hparams.optimizer_kwargs.get("lr", 1e-3),
                "betas": self.hparams.optimizer_kwargs.get(
                    "betas", (0.9, 0.999)
                ),
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(param_groups)
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer=optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LambdaLR(
                    optimizer=optimizer,
                    lr_lambda=lambda step: step
                    / self.hparams.optimizer_kwargs.get("warmup_steps", 1),
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer=optimizer,
                    T_max=self.hparams.optimizer_kwargs.get(
                        "max_steps", 100000
                    ),
                    eta_min=self.hparams.optimizer_kwargs.get("min_lr", 3e-7),
                ),
                torch.optim.lr_scheduler.LambdaLR(
                    optimizer=optimizer,
                    lr_lambda=lambda step: self.hparams.optimizer_kwargs.get(
                        "min_lr", 3e-7
                    )
                    / self.hparams.optimizer_kwargs.get("lr", 1e-3),
                ),
            ],
            milestones=[
                self.hparams.optimizer_kwargs.get("warmup_steps", 1),
                self.hparams.optimizer_kwargs.get("max_steps", 100000),
            ],
            last_epoch=-1,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
