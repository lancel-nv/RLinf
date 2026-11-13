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

"""Evaluate each 100k-update checkpoint on the same 50 scenes as the baseline."""

import argparse
import fcntl
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path


def verify_result(output: Path, baseline: Path) -> dict:
    """Verify episode coverage, paired scene hashes, and per-episode videos."""
    result = json.loads((output / "summary.json").read_text())
    episodes = json.loads((output / "episodes.json").read_text())
    expected = set(range(430000, 430050))
    if len(episodes) != 50 or {row["seed"] for row in episodes} != expected:
        raise ValueError(f"Expected 50 distinct evaluation seeds in {output}")
    successes = sum(row["success"] for row in episodes)
    if (
        result["episodes"] != 50
        or result["successes"] != successes
        or result["success_rate"] != successes / 50
    ):
        raise ValueError(f"Inconsistent episode and summary success counts in {output}")
    actual = json.loads((output / "initial_states.json").read_text())
    paired = json.loads((baseline / "initial_states.json").read_text())
    if [x["initial_state_sha256"] for x in actual] != [
        x["initial_state_sha256"] for x in paired
    ]:
        raise ValueError(f"Initial scenes differ from baseline in {output}")
    for row in episodes:
        path = output / f"seed_{row['seed']}.mp4"
        video = json.loads(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=nb_frames,width,height,r_frame_rate",
                    "-of",
                    "json",
                    str(path),
                ],
                text=True,
            )
        )["streams"][0]
        if (
            int(video["nb_frames"]),
            video["width"],
            video["height"],
            video["r_frame_rate"],
        ) != (row["steps"] + 1, 768, 384, "10/1"):
            raise ValueError(f"Expected all frames including terminal frame in {path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--gpu", default="0", help="Physical GPU index or UUID for evaluation"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Evaluate checkpoints already present, then exit",
    )
    args = parser.parse_args()

    import torch
    from omegaconf import OmegaConf
    from torch.utils.tensorboard import SummaryWriter

    from rlinf.utils.logging import get_logger

    logging.basicConfig(level=logging.INFO)
    logger = get_logger()
    run = args.run_dir.resolve()
    cfg = OmegaConf.load(run / "train.yaml")
    baseline = run / "eval50/sft2000"
    baseline_result = verify_result(baseline, baseline)
    checkpoints = (
        Path(cfg.runner.logger.log_path)
        / cfg.runner.logger.experiment_name
        / "checkpoints"
    )
    required = set(range(100000, int(cfg.runner.max_critic_updates) + 1, 100000))
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["PYTHONUNBUFFERED"] = "1"
    with (
        (run / "eval50/queue.lock").open("w") as lock,
        SummaryWriter(str(run / "eval50/tensorboard")) as writer,
    ):
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        writer.add_scalar("eval50/success_rate", baseline_result["success_rate"], 0)
        verified = {}
        while True:
            finished_before_scan = (run / "train.exitcode").exists()
            for state_path in sorted(
                checkpoints.glob("global_step_*/actor/residual_state_rank_0.pt")
            ):
                # The runner writes this marker after the actor finishes saving.
                if not (state_path.parent.parent / "collection_metrics.json").exists():
                    continue
                updates = int(
                    torch.load(state_path, map_location="cpu", weights_only=True)[
                        "budget"
                    ]["updates"]
                )
                if updates not in required or updates in verified:
                    continue
                output = run / "eval50" / f"expo{updates}"
                if not (output / "summary.json").exists():
                    command = [
                        sys.executable,
                        str(Path(__file__).with_name("evaluate.py")),
                        "--config",
                        str(run / "train.yaml"),
                        "--checkpoint",
                        str(state_path.parent),
                        "--updates",
                        str(updates),
                        "--output-dir",
                        str(output),
                    ]
                    subprocess.run(command, env=env, check=True)
                verified[updates] = verify_result(output, baseline)
                if verified[updates]["critic_updates"] != updates:
                    raise ValueError(f"Expected {updates} critic updates in {output}")
                writer.add_scalar(
                    "eval50/success_rate", verified[updates]["success_rate"], updates
                )
                writer.flush()
                logger.info(
                    "updates=%d SR=%d/50", updates, verified[updates]["successes"]
                )
                (run / "eval50/results.json").write_text(
                    json.dumps(verified, indent=2) + "\n"
                )
            if set(verified) == required or args.once:
                return
            exitcode = run / "train.exitcode"
            if finished_before_scan:
                raise RuntimeError(
                    f"Training exited with {exitcode.read_text().strip()}; missing evaluations: {sorted(required - set(verified))}"
                )
            time.sleep(20)


if __name__ == "__main__":
    main()
