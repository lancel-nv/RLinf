# Reproduce EXPO with EEF actions and π₀.₅ pooled features

[中文](README.zh-CN.md)

Reproduce the original 1 mm preinsert EXPO experiment from a published, frozen
2000-step π₀.₅ checkpoint. Evaluate the baseline, train 800,000 critic updates,
and evaluate every 100,000 updates on 50 paired scenes while saving every video.

This recipe targets **`feat/expo`**, whose runtime reference is `f918557b`.
It reproduces the original experimental implementation. The separate
`feat/expo-prefix-eef` adaptation on top of PR #1671 has different heads and
warmup behavior; the results below do not validate that adaptation.
The recipe files must be present in your checkout; the reference commit identifies
the runtime implementation, not the later addition of this recipe.

The historical source snapshot is
`logs/peg_preinsert_1mm_sft2000_pi05pool_expo_20261111_v1/code` in the author's
workspace. [reference.json](reference.json) records its implementation hashes,
model hashes and results. Compared with that snapshot, the runtime at `f918557b`
adds joint-controller support in `peg_preinsert.py`; the EEF path keeps the same
kinematics solver and operations. All other existing Python runtime files under
`rlinf/` match the snapshot.

## 1. Install and select GPUs

Use Linux with NVIDIA GPU simulation and Vulkan rendering. The historical run used
**two RTX 5880 Ada GPUs with 48 GB each**, Python 3.11.14, PyTorch 2.11.0/CUDA 13.0,
ManiSkill 3.0.0b22 and SAPIEN 3.0.1. See
[environment-reference.json](environment-reference.json) for the installed versions.
The commands below install the complete historical Python dependency set from
[requirements-reference.txt](requirements-reference.txt) into a dedicated environment.
Use an x86_64 Linux host with `uv`, a C/C++ compiler, Linux headers for the `evdev`
build, FFmpeg, and NVIDIA Vulkan driver libraries installed.

```bash
git clone --branch feat/expo https://github.com/lancel-nv/RLinf.git
cd RLinf
uv venv --python 3.11.14 .venv-expo
source .venv-expo/bin/activate
uv pip install --no-deps -r examples/embodiment/reproduce_expo/requirements-reference.txt
uv pip install --reinstall --no-deps rlinf-transformer-openpi==4.53.2
uv pip install --no-deps -e .
command -v ffprobe
export REPO_DIR="$PWD"
export REPRO="$REPO_DIR/examples/embodiment/reproduce_expo"
export PYTHONPATH="$REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS=cpu TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONUNBUFFERED=1
unset SAPIEN_DISABLE_RAY_TRACING RAY_ADDRESS
```

The full dependency list is installed with `--no-deps` to retain the historical
runtime despite older Torch constraints in upstream OpenPI/LeRobot metadata.
Reinstall the OpenPI Transformers fork last: both distributions share an import
namespace. The historical imported version is **4.53.2**, despite the separate
`transformers` distribution metadata reporting 4.57.6; installing the standard
package last causes a tokenizer-version conflict. Install RLinf from this checkout without resolving
dependencies again. The general-purpose `requirements/install.sh` remains available
for other experiments; this recipe pins the original experiment environment.

Use an NVIDIA driver compatible with CUDA 13.0; `ffprobe` from FFmpeg must be on
`PATH`. OpenPI downloads the PaliGemma tokenizer on first use into
`OPENPI_DATA_HOME`, so keep network access available. If Vulkan needs an explicit
driver selection, export its NVIDIA ICD JSON path as `VK_ICD_FILENAMES`.

## 2. Download the pretrained checkpoint

The [public SFT checkpoint](https://huggingface.co/lancel-nv/pi05-peg-preinsert-1mm-eef-sft2000)
is pinned below to an immutable revision. Its published weight SHA256 and downloaded
normalization statistics match the historical experiment.

```bash
export HF_MODEL_ID="lancel-nv/pi05-peg-preinsert-1mm-eef-sft2000"
export HF_REVISION="96c29a03adf29a750e8d40f243f8ba65887f1188"
export CKPT="$REPO_DIR/checkpoints/peg-preinsert-1mm-eef-sft2000"
hf download "$HF_MODEL_ID" --revision "$HF_REVISION" --local-dir "$CKPT"
```

The downloaded model repository contains these required paths:

```text
model_state_dict/full_weights.pt
assets/maniskill_peginsertionside_1mm_eef_rot6d/norm_stats.json
assets/maniskill_peginsertionside_1mm_eef_rot6d/manifest.json
```

`full_weights.pt` is about 13.4 GB. Keep the normalization statistics from this exact
checkpoint: they define both EEF decoding and residual authority. The provided
`prepare.py` verifies the weights and normalization SHA256 against `reference.json`
and records the local code revision. The manifest records the 400-episode GPU
source-data provenance; the dataset itself is not consumed by these RL commands.

## 3. Prepare a fresh experiment

Set physical GPU IDs below: the first trains the residual model and the second runs
π₀.₅ plus simulation. Evaluation shares the first GPU. The helper resolves physical
PCI addresses for SAPIEN so CUDA device remapping does not select the wrong renderer.

```bash
export RUN_DIR="$REPO_DIR/logs/expo-eef-pi05pool-review"
export ACTOR_GPU=0
export ROLLOUT_GPU=1
export EVAL_GPU="$ACTOR_GPU"
nvidia-smi --query-gpu=index,pci.bus_id,name,memory.total --format=csv
pci_backend() {
  nvidia-smi -i "$1" --query-gpu=pci.bus_id --format=csv,noheader |
    sed 's/^00000000:/0000:/; s/^/pci:/' | tr '[:upper:]' '[:lower:]'
}
python "$REPRO/prepare.py" --checkpoint "$CKPT" --run-dir "$RUN_DIR" \
  --rollout-render-backend "$(pci_backend "$ROLLOUT_GPU")" \
  --eval-render-backend "$(pci_backend "$EVAL_GPU")"
export OPENPI_DATA_HOME="$RUN_DIR/cache"
```

This writes `$RUN_DIR/train.yaml` and `assets.json` and refuses to overwrite a
nonempty run directory. It only substitutes local paths and renderer devices into
[train.yaml](train.yaml). All hyperparameters remain fixed.

The task starts with the peg already held. The peg is 210 × 40 × 40 mm; the hole
cross-section is 42 × 42 mm, giving 1 mm clearance per side. The tip starts 50 mm
from the mouth with ±30 mm lateral/height jitter and ±3° rotational jitter.
Cabinet poses are randomized within the task's initialization constraints.
Execution and IK use the GPU; each episode lasts at most 100 control steps at
10 Hz. This is the preinsert task, not the full table-pickup task.

| Setting | Value |
|---|---|
| Frozen base | π₀.₅, SFT step 2000; 5 denoising steps |
| Input features | Mean over valid image, prompt and discrete-state prefix tokens; 2048 dimensions |
| Residual inputs | Pooled features, 10D EEF rot6d state and base action chunk; no privileged state |
| Actions | Normalized 10D EEF rot6d, 10-step chunks; edit scale 0.02 |
| Candidates / edits | 8 / 8; target-Q selection |
| Reward | Sparse 0/1; chunk sum; γ = 0.99 per chunk |
| Termination | Success terminates; timeout continues bootstrap |
| Batch / UTD | 256 / 20 |
| Collection warmup | 4,000 action-chunk transitions |
| Learner warmup | 20,000 critic-only updates |
| Actor / alpha update | Once per 20 critic updates after warmup |
| Total updates | 800,000 critic; 39,000 actor |
| Q ensemble | 10 heads; random min of 2 for selection/backup |
| Learning rates | Actor, critic and alpha: 3e-4; target τ = 0.005 |
| Entropy | Initial alpha 1e-5; automatic target; no entropy in critic backup |
| Training seed / collection seed | 1234 / 440000 |
| Replay window | 50,000 samples |

A critic update means one optimizer step over a minibatch, regardless of the ten
Q heads. A transition represents an action chunk, not one physical control step.
This recipe intentionally retains the original critic-only warmup.

## 4. Evaluate the SFT baseline

Evaluate the unchanged base before training, in five batches of ten environments:

```bash
CUDA_VISIBLE_DEVICES="$EVAL_GPU" python "$REPRO/evaluate.py" \
  --config "$RUN_DIR/train.yaml" --output-dir "$RUN_DIR/eval50/sft2000"
cat "$RUN_DIR/eval50/sft2000/summary.json"
```

The historical result is 21/50 (42%). Seeds are **430000–430049**. At each chunk,
policy noise uses `900000 + batch_offset * 100 + chunk_index`. Success is counted
at any active step before the first termination or the 100-step limit. Each finished
episode immediately updates `episodes.json` and `progress.json`; `summary.json`
includes the final SR and Wilson 95% interval. `initial_states.json` enables exact
scene comparison with the RL evaluations. Keep all these files.

## 5. Train and evaluate every 100,000 updates

Start both processes from the same prepared shell, preferably inside `tmux`:

```bash
CUDA_VISIBLE_DEVICES="$ACTOR_GPU,$ROLLOUT_GPU" \
  bash "$REPRO/train.sh" "$RUN_DIR" > "$RUN_DIR/train.log" 2>&1 &
TRAIN_PID=$!
python "$REPRO/evaluate_checkpoints.py" --run-dir "$RUN_DIR" \
  --gpu "$EVAL_GPU" > "$RUN_DIR/eval50/queue.log" 2>&1 &
EVAL_PID=$!
printf 'training PID=%s, evaluation PID=%s\n' "$TRAIN_PID" "$EVAL_PID"
tail -f "$RUN_DIR/train.log" "$RUN_DIR/eval50/queue.log"
```

`train.sh` starts fresh and records its exit code. The queue waits for complete
checkpoints, reads the actual critic-update budget, and evaluates each checkpoint
on the baseline scenes. It saves 50 side-by-side main/wrist videos per checkpoint,
including the terminal frame, and checks frame counts with `ffprobe`.
The historical training took about **10.5 hours**; shared evaluation can affect
throughput and asynchronous collection timing. This is not a runtime guarantee.

For evaluation after training has finished, run the queue once instead:

```bash
python "$REPRO/evaluate_checkpoints.py" --run-dir "$RUN_DIR" \
  --gpu "$EVAL_GPU" --once
```

Do not run two queues on the same directory. Existing complete evaluations are
verified and reused. If an evaluation is interrupted, move its partial output
directory aside before retrying; partial runs are not silently overwritten.
A baseline `summary.json` is required before starting the queue.

To evaluate one checkpoint explicitly:

```bash
export ACTOR_CHECKPOINT="/absolute/path/to/global_step_N/actor"
CUDA_VISIBLE_DEVICES="$EVAL_GPU" python "$REPRO/evaluate.py" \
  --config "$RUN_DIR/train.yaml" --checkpoint "$ACTOR_CHECKPOINT" \
  --updates 800000 --output-dir "$RUN_DIR/eval50/expo800000-manual"
```

`global_step_N` is a runner iteration, not a critic-update count. The evaluator
checks `residual_state_rank_0.pt` against `--updates`. Keep the complete `actor`
directory, including `model_state_dict/full_weights.pt` and the residual budget.
The original EXPO policy stores its target heads in that model state; do not load
an adaptation checkpoint from `feat/expo-prefix-eef` into this evaluator.

## 6. Inspect metrics, videos and historical results

View training and fixed-scene evaluation curves together:

```bash
tensorboard --logdir "$RUN_DIR" --host 127.0.0.1 --port 6013
# On your laptop:
ssh -N -L 6013:127.0.0.1:6013 USER@TRAINING_HOST
```

Open `http://localhost:6013`. Training events are under `results/`; fixed-scene
SR is `eval50/success_rate` under `eval50/tensorboard`. `eval50/results.json`
collects verified SRs. Individual videos are `eval50/sft2000/seed_*.mp4` and
`eval50/expo<updates>/seed_*.mp4`, with 50 videos in each completed evaluation.

```bash
tar -czf "$RUN_DIR/sft2000-50videos.tar.gz" \
  -C "$RUN_DIR/eval50" sft2000
tar -czf "$RUN_DIR/expo800000-50videos.tar.gz" \
  -C "$RUN_DIR/eval50" expo800000
# On your laptop:
scp USER@TRAINING_HOST:/absolute/path/to/run/expo800000-50videos.tar.gz .
```

The archive commands package the baseline and final videos with their JSON metadata. Replace the host and
absolute run path in the download command; use the analogous baseline archive
name to download the baseline videos.

| Critic updates | Successes / 50 | SR |
|---:|---:|---:|
| 0 (SFT) | 21 | 42% |
| 100,000 | 36 | 72% |
| 200,000 | 42 | 84% |
| 300,000 | 39 | 78% |
| 400,000 | 40 | 80% |
| 500,000 | 33 | 66% |
| 600,000 | 38 | 76% |
| 700,000 | 31 | 62% |
| 800,000 | 35 | 70% |

These are one historical training seed and 50 paired evaluation scenes, not a
multi-seed estimate or a guarantee of exact SR on another stack. The best checkpoint
was 200,000 updates (84%); the final checkpoint was 800,000 (70%). Report both when
comparing methods. GPU physics, package versions and asynchronous timing can change
a new run. Compare initial-scene hashes within each reproduction and retain the
code revision, asset hashes, logs and complete evaluation outputs.

## 7. Validation scope

This recipe was exercised on 2026-11-13 in a newly installed virtual environment
on two RTX 5880 Ada GPUs. [validation.json](validation.json) records the checks:

- Public HF weight SHA256 and CLI download of the normalization statistics verified.
  The GPU tests used the local checkpoint with the same published weight SHA256;
  the full 13.4 GB file was not downloaded a second time.
- SFT baseline: **21/50 (42%)**. Historical final EXPO checkpoint: **35/50 (70%)**.
  All 100 episode outcomes and all initial scene hashes match the historical run.
  All 100 newly generated videos passed frame-count verification.
- Short asynchronous training: **400 critic updates, 19 actor updates**, exit code 0,
  with a complete saved checkpoint. The short test uses 32 warmup transitions,
  20 warmup updates, 20 critic-only updates and a replay minimum of 32; its save
  interval is 400. The published 800,000-update configuration is unchanged.
- Checkpoint discovery, result verification and TensorBoard output through the
  evaluation queue's `--once` path passed. The long-running wait loop was not
  exercised across a new 800,000-update run.
- **19 CPU EXPO tests passed** (129 deselected), together with pre-commit, shell
  syntax and EN/ZH command parity checks.

```bash
python "$REPRO/prepare.py" --help
python "$REPRO/evaluate.py" --help
python "$REPRO/evaluate_checkpoints.py" --help
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$REPO_DIR" \
  python -m pytest tests/unit_tests/test_sac_residual.py -k expo -q
```

These checks cover installation, GPU inference and simulation, and the training
and checkpoint paths. A complete 800,000-update training run was **not repeated**
for this recipe; the curve above is the original experiment. This documentation
and the portability scripts were prepared with an AI assistant.
