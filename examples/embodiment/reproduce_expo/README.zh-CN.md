# 复现 EEF 动作与 π₀.₅ 池化特征的 EXPO 实验

[English](README.md)

本说明从已发布的 2000-step π₀.₅ checkpoint 开始，复现原来的 1 mm 预插入
EXPO 实验：评测基座、训练 80 万次 critic 更新，并每 10 万次更新评测相同的
50 个场景，保存全部视频。

本说明对应 **`feat/expo`**，运行代码参考版本为 `f918557b`。
它对应原来的实验实现。基于 PR #1671 的 `feat/expo-prefix-eef` 适配版本使用
不同的网络头和 warmup 行为；下方实验结果不能证明新适配版本已通过训练验证。
请使用包含本复现目录的代码版本；参考 commit 标识运行代码，而非后来新增的文档。

作者工作区的历史源码快照在
`logs/peg_preinsert_1mm_sft2000_pi05pool_expo_20261111_v1/code`。
[reference.json](reference.json) 记录实现与模型的哈希及历史结果。
相对该快照，`f918557b` 的 `peg_preinsert.py` 增加了关节控制支持，EEF 路径
仍使用同一运动学求解器与运算；`rlinf/` 下其他原有 Python 运行文件一致。

## 1. 安装并选择 GPU

使用支持 NVIDIA GPU 仿真与 Vulkan 渲染的 Linux 环境。历史实验使用
**两张 48 GB RTX 5880 Ada**、Python 3.11.14、PyTorch 2.11.0/CUDA 13.0、
ManiSkill 3.0.0b22 和 SAPIEN 3.0.1。完整版本记录见
[environment-reference.json](environment-reference.json)。以下命令将
[requirements-reference.txt](requirements-reference.txt) 中的完整历史 Python 依赖
安装到独立虚拟环境。使用 x86_64 Linux，并预先安装 `uv`、C/C++ 编译器、
用于构建 `evdev` 的 Linux 头文件、FFmpeg 和 NVIDIA Vulkan 驱动库。

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

使用 `--no-deps` 安装完整清单，保留历史运行版本，避免 OpenPI/LeRobot 元数据中
旧 Torch 约束触发降级。最后重装 OpenPI Transformers fork，因为两个发行包共享导入命名空间。
历史实际导入版本为 **4.53.2**，虽然独立 `transformers` 包的元数据为 4.57.6；
如果最后安装标准包，会出现 tokenizer 版本冲突。再从当前 checkout 安装 RLinf，不重新解析依赖。
其他实验仍可使用通用的 `requirements/install.sh`；本说明固定原实验环境。

NVIDIA 驱动需要支持 CUDA 13.0，FFmpeg 的 `ffprobe` 需要在 `PATH` 中。
OpenPI 首次运行会将 PaliGemma tokenizer 下载至 `OPENPI_DATA_HOME`，此时需要网络。
如果主机需要显式指定 Vulkan 驱动，请将 NVIDIA ICD JSON 路径设为 `VK_ICD_FILENAMES`。

## 2. 下载预训练 checkpoint

[SFT checkpoint 已公开](https://huggingface.co/lancel-nv/pi05-peg-preinsert-1mm-eef-sft2000)，
以下命令固定到不可变 revision。公开权重的 SHA256 和实际下载的归一化统计
均已核对，与历史实验一致。

```bash
export HF_MODEL_ID="lancel-nv/pi05-peg-preinsert-1mm-eef-sft2000"
export HF_REVISION="96c29a03adf29a750e8d40f243f8ba65887f1188"
export CKPT="$REPO_DIR/checkpoints/peg-preinsert-1mm-eef-sft2000"
hf download "$HF_MODEL_ID" --revision "$HF_REVISION" --local-dir "$CKPT"
```

下载的模型仓库包含以下必需文件：

```text
model_state_dict/full_weights.pt
assets/maniskill_peginsertionside_1mm_eef_rot6d/norm_stats.json
assets/maniskill_peginsertionside_1mm_eef_rot6d/manifest.json
```

`full_weights.pt` 大约 13.4 GB。必须使用该 checkpoint 配套的归一化统计：它同时
决定 EEF 解码与 residual 修正权限。`prepare.py` 会按 `reference.json` 校验
权重和归一化文件的 SHA256，并记录本地代码版本。
manifest 记录 400 条 GPU 示范数据的来源；这里的 RL 命令不读取该数据集。

## 3. 准备新的实验目录

设置实际物理 GPU 编号：第一张训练 residual，第二张运行 π₀.₅ 和仿真，评测
共用第一张。辅助函数取得 SAPIEN 的物理 PCI 地址，避免 CUDA 设备重映射后
选错渲染卡。

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

该命令写入 `$RUN_DIR/train.yaml` 和 `assets.json`，拒绝覆盖非空目录。
它仅替换 [train.yaml](train.yaml) 中的本地路径和渲染设备，保留全部训练超参。

任务初始化时已经夹住棍子。棍子为 210 × 40 × 40 mm，孔截面为 42 × 42 mm，
即每侧 1 mm 余量。尖端距孔口 50 mm，横向与高度随机 ±30 mm，旋转随机 ±3°。
柜子位姿在任务初始化约束内随机。执行与 IK 均在 GPU 上，每局最多 100 个控制步，
频率 10 Hz。这是预插入任务，与从桌面抓取的完整任务不同。

| 参数 | 数值 |
|---|---|
| 冻结基座 | π₀.₅ SFT 2000 step；5 步去噪 |
| 视觉特征 | 对有效图像、prompt、本体离散状态 prefix token 求均值，2048 维 |
| Residual 输入 | 池化特征、10 维 EEF rot6d 本体状态和基座动作 chunk；不使用特权状态 |
| 动作 | 归一化 10 维 EEF rot6d，chunk 长度 10；edit scale 0.02 |
| 候选 / 编辑 | 8 / 8；target Q 选择 |
| 奖励 | Sparse 0/1，chunk 内求和；每 chunk 的 γ = 0.99 |
| 终止 | 成功终止；超时继续 bootstrap |
| Batch / UTD | 256 / 20 |
| 采集 warmup | 4,000 条动作 chunk transition |
| Learner warmup | 20,000 次仅 critic 更新 |
| Actor / alpha | warmup 后每 20 次 critic 更新一次 |
| 总更新次数 | critic 800,000；actor 39,000 |
| Q ensemble | 10 个头；选择和 backup 使用随机两个头的最小值 |
| 学习率 | actor、critic、alpha 均为 3e-4；target τ = 0.005 |
| 熵 | initial alpha 1e-5，自动 target；critic backup 不加熵 |
| 训练 / 采集 seed | 1234 / 440000 |
| Replay 窗口 | 50,000 个样本 |

一次 critic 更新指一个 minibatch 上的一次 optimizer step，不乘 Q 头数。
一条 transition 对应一个动作 chunk，不是一个物理控制步。
本复现保留历史实验的 critic-only warmup。

## 4. 评测 SFT 基座

训练前先评测原始基座，使用 5 批、每批 10 个并行环境：

```bash
CUDA_VISIBLE_DEVICES="$EVAL_GPU" python "$REPRO/evaluate.py" \
  --config "$RUN_DIR/train.yaml" --output-dir "$RUN_DIR/eval50/sft2000"
cat "$RUN_DIR/eval50/sft2000/summary.json"
```

历史结果为 21/50（42%），场景 seeds 为 **430000–430049**。
每个 chunk 的推理噪声 seed 为 `900000 + batch_offset * 100 + chunk_index`。
在首次终止或 100 步上限前的有效步骤中出现成功，即计为成功。
每完成一局立即更新 `episodes.json` 和 `progress.json`；最终 `summary.json`
记录 SR 与 Wilson 95% 区间。保留 `initial_states.json`，用于与 RL 评测逐场景核对。

## 5. 训练并每 10 万次更新评测

在同一个已配置的 shell 中启动两个进程，建议使用 `tmux`：

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

`train.sh` 从头训练并记录退出码。评测队列等待 checkpoint 完整保存，读取实际
critic 更新计数，再评测与基座相同的场景。每个 checkpoint 保存 50 个主相机与
腕部相机并排视频，包含终止帧，并用 `ffprobe` 检查帧数。
历史训练耗时约 **10.5 小时**；共卡评测会影响吞吐与异步采集时序，不能保证同样耗时。

若训练已经结束，可改为只评测当前已有 checkpoint：

```bash
python "$REPRO/evaluate_checkpoints.py" --run-dir "$RUN_DIR" \
  --gpu "$EVAL_GPU" --once
```

同一目录不要启动两个评测队列。已完成的评测会被核验并复用。若评测中断，重试前
先将对应的部分输出目录移开；脚本不会悄悄覆盖未完成的结果。
启动队列前需要已有基座 `summary.json`。

也可以指定一个 checkpoint 单独评测：

```bash
export ACTOR_CHECKPOINT="/absolute/path/to/global_step_N/actor"
CUDA_VISIBLE_DEVICES="$EVAL_GPU" python "$REPRO/evaluate.py" \
  --config "$RUN_DIR/train.yaml" --checkpoint "$ACTOR_CHECKPOINT" \
  --updates 800000 --output-dir "$RUN_DIR/eval50/expo800000-manual"
```

`global_step_N` 中的 N 是 runner 迭代，不是 critic 更新次数。脚本会读取
`residual_state_rank_0.pt`，核对是否与 `--updates` 一致。保留完整 `actor`
目录，包括 `model_state_dict/full_weights.pt` 和 residual budget。
原实现将 target heads 保存在该模型状态内；不要向此入口加载
`feat/expo-prefix-eef` 适配分支的 checkpoint。

## 6. 查看指标、视频与历史结果

将训练曲线与固定场景评测结果放在同一 TensorBoard 中：

```bash
tensorboard --logdir "$RUN_DIR" --host 127.0.0.1 --port 6013
# On your laptop:
ssh -N -L 6013:127.0.0.1:6013 USER@TRAINING_HOST
```

打开 `http://localhost:6013`。训练事件位于 `results/`；固定场景评测曲线
`eval50/success_rate` 位于 `eval50/tensorboard`。`eval50/results.json` 汇总已核验的 SR。
视频位于 `eval50/sft2000/seed_*.mp4` 和 `eval50/expo<updates>/seed_*.mp4`，
每次完整评测各有 50 个。

```bash
tar -czf "$RUN_DIR/sft2000-50videos.tar.gz" \
  -C "$RUN_DIR/eval50" sft2000
tar -czf "$RUN_DIR/expo800000-50videos.tar.gz" \
  -C "$RUN_DIR/eval50" expo800000
# On your laptop:
scp USER@TRAINING_HOST:/absolute/path/to/run/expo800000-50videos.tar.gz .
```

打包命令分别打包基座与最终模型的视频及 JSON 元数据。下载时替换实际主机和绝对路径；下载基座
视频时，将文件名改为对应的基座压缩包。

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

以上是一个训练 seed、50 个配对评测场景的历史结果，不代表多 seed 均值，也不保证
其他软硬件环境得到完全一致的 SR。最佳 checkpoint 为 20 万次更新（84%），最终
80 万次为 70%，比较方法时应同时报告。GPU 物理、依赖版本和异步时序均可能使新实验
发生变化。应核对同一轮复现中的初始场景哈希，并保留代码版本、资产哈希、日志与完整评测产物。

## 7. 验证范围

2026-11-13 已在全新安装的虚拟环境、两张 RTX 5880 Ada 上验证本流程。
[validation.json](validation.json) 记录以下实测结果：

- 已核对公开 HF 权重 SHA256，并通过 CLI 下载归一化统计。
  GPU 测试使用与公开权重 SHA256 相同的本地 checkpoint，没有重复下载完整的 13.4 GB 文件。
- SFT 基座 **21/50（42%）**；历史最终 EXPO checkpoint **35/50（70%）**。
  全部 100 局成败结果和初始场景哈希与历史一致，新生成的 100 个视频均通过帧数检查。
- 异步短训练完成 **400 次 critic 更新、19 次 actor 更新**，退出码为 0，并完整保存 checkpoint。
  短测试使用 32 条 warmup transition、20 次 warmup 更新、20 次 critic-only 更新，
  replay 最小样本数为 32，保存间隔为 400。发布的 80 万更新配置保持原参数。
- 通过评测队列的 `--once` 路径验证 checkpoint 发现、结果核验及 TensorBoard 输出。
  本轮没有在一次新的 80 万更新训练中验证持续等待队列。
- **19 项 CPU EXPO 测试通过**（其余 129 项未选中），pre-commit、shell 语法及中英文命令一致性检查通过。

```bash
python "$REPRO/prepare.py" --help
python "$REPRO/evaluate.py" --help
python "$REPRO/evaluate_checkpoints.py" --help
CUDA_VISIBLE_DEVICES="" PYTHONPATH="$REPO_DIR" \
  python -m pytest tests/unit_tests/test_sac_residual.py -k expo -q
```

这些检查覆盖安装、GPU 推理与仿真、训练及 checkpoint 路径。
本轮**没有重新训练完整的 80 万次更新**，上方曲线来自原始实验。
本说明与可移植脚本由 AI 助手协助整理。
