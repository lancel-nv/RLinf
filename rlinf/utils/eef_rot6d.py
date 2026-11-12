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

"""Normalization and canonical network inputs for ten-dimensional EEF actions."""

import json
from pathlib import Path

import torch
from torch import nn

from rlinf.utils.torch_rot6d import canonicalize_rot6d


class EEFNormalizer(nn.Module):
    """Own pi05 quantile statistics for xyz, column rot6d, and gripper coordinates."""

    def __init__(self, path: str) -> None:
        super().__init__()
        stats = json.loads(Path(path).read_text())["norm_stats"]
        for key in ("state", "actions"):
            low = torch.tensor(stats[key]["q01"], dtype=torch.float32)
            high = torch.tensor(stats[key]["q99"], dtype=torch.float32)
            if (
                low.shape != (10,)
                or high.shape != (10,)
                or not torch.isfinite(low).all()
                or not torch.isfinite(high).all()
                or (high < low).any()
            ):
                raise ValueError(
                    f"Expected finite ordered 10-D EEF {key} quantiles in {path}."
                )
            self.register_buffer(key + "_offset", (high + low + 1e-6) / 2)
            self.register_buffer(key + "_scale", (high - low + 1e-6) / 2)

    def normalize(self, value: torch.Tensor, key: str) -> torch.Tensor:
        """Map raw dataset coordinates to pi05 quantile-normalized coordinates."""
        return (value - getattr(self, key + "_offset")) / getattr(self, key + "_scale")

    def unnormalize(self, value: torch.Tensor, key: str) -> torch.Tensor:
        """Restore raw dataset coordinates without clipping out-of-range values."""
        return value * getattr(self, key + "_scale") + getattr(self, key + "_offset")

    def network_input(self, value: torch.Tensor, key: str) -> torch.Tensor:
        """Keep normalized xyz/gripper and canonicalize rotation in physical coordinates."""
        rotation = canonicalize_rot6d(self.unnormalize(value, key)[..., 3:9])
        return torch.cat((value[..., :3], rotation, value[..., 9:]), -1)
