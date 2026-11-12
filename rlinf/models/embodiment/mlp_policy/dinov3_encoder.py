# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Frozen, spatially pooled DINOv3 features for residual SAC."""

import hashlib
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class DINOv3Config:
    """Local DINOv3-S/16 assets and ordered RGB camera preprocessing."""

    repo_path: str
    weights_path: str
    weights_sha256: str | None = None
    camera_keys: tuple[str, ...] = ("main_images", "wrist_images")
    image_height: int = 224
    image_width: int = 320
    spatial_pool_height: int = 2
    spatial_pool_width: int = 2

    @property
    def output_dim(self) -> int:
        return (
            len(self.camera_keys)
            * 384
            * self.spatial_pool_height
            * self.spatial_pool_width
        )

    def validate(self) -> None:
        """Reject incompatible camera layouts, image sizes and asset identities."""
        if not self.repo_path or not self.weights_path:
            raise ValueError("DINOv3 requires explicit repo_path and weights_path.")
        if (
            isinstance(self.camera_keys, str)
            or not self.camera_keys
            or any(not isinstance(key, str) or not key for key in self.camera_keys)
            or len(set(self.camera_keys)) != len(self.camera_keys)
        ):
            raise ValueError(
                "DINOv3 camera_keys must be a non-empty sequence of unique keys."
            )
        for name in (
            "image_height",
            "image_width",
            "spatial_pool_height",
            "spatial_pool_width",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(
                    f"Expected positive integer DINOv3 {name}, got {value}."
                )
        if self.image_height % 16 or self.image_width % 16:
            raise ValueError("DINOv3 image dimensions must be divisible by 16.")
        if (
            self.spatial_pool_height > self.image_height // 16
            or self.spatial_pool_width > self.image_width // 16
        ):
            raise ValueError("DINOv3 pooling grid must fit inside the patch grid.")
        if self.weights_sha256 is not None and (
            len(self.weights_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.weights_sha256.lower())
        ):
            raise ValueError(
                "DINOv3 weights_sha256 must contain 64 hexadecimal characters."
            )


def _load_backbone(config: DINOv3Config) -> nn.Module:
    repo = Path(config.repo_path).expanduser().resolve()
    weights = Path(config.weights_path).expanduser().resolve()
    if not (repo / "dinov3/hub/backbones.py").is_file():
        raise FileNotFoundError(
            f"Expected a local official DINOv3 repository at {repo}."
        )
    if not weights.is_file():
        raise FileNotFoundError(f"Expected DINOv3-S/16 weights at {weights}.")
    if config.weights_sha256 is not None:
        checksum = hashlib.sha256()
        with weights.open("rb") as handle:
            while block := handle.read(16 * 1024 * 1024):
                checksum.update(block)
        digest = checksum.hexdigest()
        if digest != config.weights_sha256.lower():
            raise ValueError(
                f"Expected DINOv3 SHA256 {config.weights_sha256}, got {digest}."
            )
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    package = importlib.import_module("dinov3")
    if repo not in Path(package.__file__).resolve().parents:
        raise ImportError(f"Expected DINOv3 from {repo}, got {package.__file__}.")
    backbone = importlib.import_module("dinov3.hub.backbones").dinov3_vits16(
        pretrained=False
    )
    backbone.load_state_dict(
        torch.load(weights, map_location="cpu", weights_only=True), strict=True
    )
    return backbone


class FrozenDINOv3Encoder(nn.Module):
    """Encode ordered RGB views as FP32, camera-major spatial patch features.

    Inputs are uint8 RGB or floating RGB in [0, 1] or [0, 255], shaped
    [batch, height, width, 3]. The rollout worker owns this frozen encoder;
    only its output is stored in replay and sent to the learner.
    """

    def __init__(self, config: DINOv3Config, *, backbone: nn.Module | None = None):
        super().__init__()
        config.validate()
        self.config = config
        self.backbone = (
            _load_backbone(config) if backbone is None else backbone
        ).float()
        self.backbone.requires_grad_(False)
        self.register_buffer(
            "image_mean",
            torch.tensor((0.485, 0.456, 0.406)).reshape(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor((0.229, 0.224, 0.225)).reshape(1, 3, 1, 1),
            persistent=False,
        )
        self.eval()

    def train(self, mode: bool = True) -> "FrozenDINOv3Encoder":
        """Keep the encoder and backbone in evaluation mode."""
        super().train(False)
        return self

    @torch.no_grad()
    def forward(self, env_obs: dict[str, Any]) -> torch.Tensor:
        """Return [batch, cameras * 384 * pool_height * pool_width] features."""
        views = []
        batch_size = None
        for key in self.config.camera_keys:
            if key not in env_obs or env_obs[key] is None:
                raise ValueError(
                    f"Expected DINOv3 RGB camera {key!r} in the observation."
                )
            image = torch.as_tensor(env_obs[key], device=self.image_mean.device)
            if image.ndim != 4 or image.shape[-1] != 3 or min(image.shape) < 1:
                raise ValueError(
                    f"Expected {key} shaped [B,H,W,3], got {tuple(image.shape)}."
                )
            if batch_size is not None and image.shape[0] != batch_size:
                raise ValueError("Expected equal DINOv3 camera batch sizes.")
            batch_size = image.shape[0]
            byte_image = image.dtype == torch.uint8
            if not byte_image and not image.is_floating_point():
                raise ValueError(
                    f"Expected uint8 or floating RGB for {key}, got {image.dtype}."
                )
            image = image.permute(0, 3, 1, 2).float()
            if not torch.isfinite(image).all() or image.min() < 0 or image.max() > 255:
                raise ValueError(f"Expected finite RGB values in [0, 255] for {key}.")
            if byte_image or image.max() > 1:
                image = image / 255.0
            image = F.interpolate(
                image,
                size=(self.config.image_height, self.config.image_width),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            ).clamp_(0, 1)
            views.append(image)
        images = torch.stack(views, dim=1).flatten(0, 1)
        images = (images - self.image_mean) / self.image_std
        with torch.autocast(device_type=images.device.type, enabled=False):
            patches = self.backbone.forward_features(images)["x_norm_patchtokens"]
        height, width = self.config.image_height // 16, self.config.image_width // 16
        expected = (batch_size * len(views), height * width, 384)
        if tuple(patches.shape) != expected:
            raise ValueError(
                f"Expected DINOv3 patch tokens {expected}, got {tuple(patches.shape)}."
            )
        grid = (
            patches.float().reshape(expected[0], height, width, 384).permute(0, 3, 1, 2)
        )
        pooled = F.adaptive_avg_pool2d(
            grid, (self.config.spatial_pool_height, self.config.spatial_pool_width)
        )
        return pooled.reshape(batch_size, -1).detach()
