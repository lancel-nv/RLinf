# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Validate the published SFT assets and materialize a local EXPO configuration."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def sha256_file(path: Path) -> str:
    """Hash an asset without loading the complete checkpoint into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--rollout-render-backend",
        required=True,
        help="SAPIEN PCI device, e.g. pci:0000:03:00.0",
    )
    parser.add_argument(
        "--eval-render-backend",
        required=True,
        help="SAPIEN PCI device of the evaluation GPU",
    )
    args = parser.parse_args()

    from omegaconf import OmegaConf

    source = Path(__file__).resolve().parent
    reference = json.loads((source / "reference.json").read_text())
    checkpoint = args.checkpoint.expanduser().resolve()
    run_dir = args.run_dir.expanduser().resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Expected a new or empty run directory, found {run_dir}")
    norm = (
        checkpoint / "assets/maniskill_peginsertionside_1mm_eef_rot6d/norm_stats.json"
    )
    weights = checkpoint / "model_state_dict/full_weights.pt"
    manifest = json.loads(norm.with_name("manifest.json").read_text())
    for path, expected in [
        (norm, reference["norm_stats_sha256"]),
        (weights, reference["weights_sha256"]),
    ]:
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"Expected SHA256 {expected} for {path}, found {actual}")
    if manifest["norm_stats_sha256"] != reference["norm_stats_sha256"]:
        raise ValueError(
            "Expected manifest to identify the reference normalization statistics"
        )

    cfg = OmegaConf.load(source / "train.yaml")
    cfg.actor.model.eef_rot6d_norm_stats_path = str(norm)
    cfg.rollout.residual_base_model.model_path = str(checkpoint)
    cfg.rollout.residual_base_model.openpi_data.norm_stats_path = str(norm)
    cfg.runner.logger.log_path = str(run_dir / "results")
    for mode, backend in [
        ("train", args.rollout_render_backend),
        ("eval", args.eval_render_backend),
    ]:
        cfg.env[mode].init_params.render_backend = backend
        cfg.env[mode].video_cfg.video_base_dir = str(run_dir / "video" / mode)
    if "REPLACE_" in OmegaConf.to_yaml(cfg):
        raise ValueError("Expected all template paths to be materialized")
    run_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, run_dir / "train.yaml", resolve=True)
    repo = source.parents[2]
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo, text=True
    )
    (run_dir / "assets.json").write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "hf_repo_id": reference["hf_repo_id"],
                "hf_revision": reference["hf_revision"],
                "code_commit": revision,
                "git_status": status,
                "weights_sha256": reference["weights_sha256"],
                "norm_stats_sha256": reference["norm_stats_sha256"],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
