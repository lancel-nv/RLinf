Soft Actor-Critic (SAC)
==================================

1. 引言
---------------

Soft Actor-Critic (SAC) 是目前应用最广泛的强化学习 (RL) 算法之一。  
它包含两个核心部分：  

- Actor （策略模型）: 根据当前状态生成动作。  
- Critic （Q值模型）: 评估当前状态与所选动作的价值。  

SAC 是一种用于连续动作控制的 off-policy 深度强化学习算法。
它基于最大熵强化学习框架，用熵扩展标准的强化学习优化目标，以平衡利用与探索。
SAC 利用熵正则化的 Bellman 方程和自动温度调控，同时学习一个随机策略和多个Q函数。
因其样本效率和稳定性，SAC 在机器人控制领域有广泛应用。

更多细节请参考原始论文  
`SAC <https://arxiv.org/abs/1801.01290>`_.


2. 目标函数
----------------------

设策略为 :math:`\pi`。则 :math:`\pi` 的 Q 值函数定义为: :math:`Q^{\pi}(s, a)`。
在 SAC 里，Q 值函数满足如下松弛的 Bellman 公式： 

.. math::

   Q^{\pi}(s, a) = \mathbb{E}_{s' \sim P, a \sim \pi} \left[
      r(s, a) + \gamma (Q^{\pi}(s', a') + H(\pi(\cdot|s')))
   \right]
   = \mathbb{E}_{s' \sim P, a \sim \pi} \left[
      r(s, a) + \gamma (Q^{\pi}(s', a') - \alpha \log \pi(a'|s'))
   \right].

这里 :math:`\gamma` 是折扣因子, :math:`H` 是策略的熵, and :math:`\alpha` 是温度参数（平衡熵和期望回报）。

因此，第 i 个 Q 值函数的损失函数 :math:`Q_{\phi_{i}}` 如下：

.. math::

   L(\phi_{i}, D) = \mathbb{E}_{(s, a, r, s', d) \sim D} \left[
      \frac{1}{2} \left(
         Q_{\phi_{i}}(s, a) - (r + \gamma (1 - d)(\min_{i} Q_{\overline{\phi_{\text{targ}, i}}}(s', a') - \alpha \log \pi_{\theta}(a'|s')))
      \right)^2
   \right],

这里 :math:`D` 是回放缓冲区, :math:`\overline{\phi_{\text{targ}, i}}` 是目标 Q 函数的参数，:math:`a'` 是从当前策略 :math:`\pi_{\theta}` 采样的动作。

策略 :math:`\pi_{\theta}` 最大化熵和期望回报的期望。因此，策略损失如下：

.. math::

   L(\theta, D) = \mathbb{E}_{s \sim D, a \sim \pi_{\theta}} \left[
      \alpha \log \pi_{\theta}(a|s) - \min_{i} Q_{\phi_i}(s, a)
   \right].

实践上，温度参数 :math:`\alpha` 是可学习的参数。其损失函数如下：

.. math::

   L(\alpha) = - \alpha (H_{\text{targ}} - H(\pi(\cdot, d))), 

这里 :math:`H_{\text{targ}}` 是一个超参数，表示熵的目标大小，通常设成动作维度的相反数。 


3. 配置
-----------------

目前，在我们的框架中，SAC 仅支持具身任务。  
算法配置如下所示：  

.. code-block:: yaml

   algorithm:
      update_epoch: 32
      group_size: 1
      agg_q: min # ["min", "mean"]. 选择如何聚合多个Q函数的值。


      adv_type: embodied_sac
      loss_type: embodied_sac
      loss_agg_func: "token-mean"
      
      bootstrap_type: standard # [standard, always]. 是否累积下一步 Q 值的判断准则。
      tau: 0.01  # 松弛更新目标Q值网络的比例
      target_update_freq: 1  # 目标Q函数的更新频率
      entropy_tuning:
         alpha_type: softplus  # ["softplus","exp","fixed_alpha"]
         initial_alpha: 0.01  # 初始温度值
         target_entropy: -4  # 目标熵（默认 -action_dim）
         optim:
            lr: 3.0e-4  # 温度参数学习率
            lr_scheduler: torch_constant
            clip_grad: 10.0
      
      # 回放缓冲区设置
      replay_buffer:
         enable_cache: True # 启用内存缓存以减少I/O开销
         cache_size: 6000 # 内存缓存的轨迹数量
         sample_window_size: 6000 # 滑动采样窗口大小
         min_buffer_size: 2  # 开始更新策略时缓冲区数据量最小值（以Trajectory为单位）

4. 冻结 VLA 的 Residual SAC
----------------------------

Residual SAC 在冻结的 OpenPI action chunk 上学习有界修正。ManiSkill 示例使用 π₀.₅、Panda 关节控制和 ``PegInsertionSideWideClearance-v1``，训练小型 FP32 网络，由 rollout worker 运行冻结的 VLA，不需要训练 RLT token。

默认情况下，actor 的输入包括归一化 proprioception、冻结 VLA prefix embedding 的 masked mean，以及原始基模动作。下文的 DINOv3 示例会同时替换 actor 和 critic 的视觉特征。Critic 接收投影后的特征和叠加残差后的动作。两组特征投影都由 critic optimizer 更新，actor 更新时将其 detach。每次决策只采样一个基模 proposal，并将它与特征一起存入 replay。

动作与目标值
~~~~~~~~~~~~

残差在归一化的模型动作空间中叠加：``action = base_action + edit_scale * tanh(z)``。Rollout worker 再通过基模的 output transforms 解码成环境动作。Replay 保存归一化后的组合动作，不保存解码后的环境动作作为 critic 输入。Log-prob 包含一次 tanh Jacobian 和一次 scale Jacobian。

示例不裁剪归一化动作，因此保留超出 [-1, 1] 的基模 proposal。设置 ``actor.model.action_clipping: final`` 会裁剪组合后的归一化动作；此时熵仍描述裁剪前的残差分布。环境自身的执行器限幅仍然生效。

Critic target 将 chunk 内奖励求和，每次决策只应用一次 ``gamma``。真正的 termination 停止 bootstrap；默认情况下，时间限制从 final observation 继续 bootstrap。Actor 评分和 target 计算都从 critic ensemble 中无放回随机抽取两个 critic，取其最小值。所有 critic head 使用同一个 target 学习。是否在 target 中加入熵由 ``algorithm.backup_entropy`` 单独控制。

超参数
~~~~~~

示例参考 ``groot/core/rl/algorithms/sac_residual`` 的共享算法配置，不复现 N2 专用 warmup 配方及其成功率。这里使用的 π₀.₅ 特征、Panda 动作坐标和仿真奖励均与原实验不同。

.. list-table::
   :header-rows: 1
   :widths: 35 20 45

   * - 设置
     - 默认值
     - 含义
   * - Action chunk
     - 10 × 8
     - 十步 Panda 关节控制动作。
   * - 残差幅度 / 初始 log std
     - 0.2 / -1.5
     - 初始均值为零；log std 限制在 [-5, 2]。
   * - 隐藏层 / critic ensemble
     - 3 × 256 / 10
     - Linear、LayerNorm 和 ReLU；随机两个 critic 取最小值。
   * - State / prefix 投影
     - 128 / 256
     - 分别投影 9 维状态和 2048 维冻结 prefix 特征。
   * - Batch / 学习率
     - 256 / 3e-4
     - Actor、critic 和温度均使用 Adam；betas 为 (0.9, 0.999)，epsilon 为 1e-8。
   * - 梯度范数上限 / 初始 alpha
     - 1.0 / 2e-4
     - 按 optimizer 分别裁剪梯度；alpha 使用 exp 参数化。
   * - Gamma / Polyak tau
     - 0.99 / 0.005
     - 按 chunk 折扣；每次 critic 更新后更新 target。
   * - Actor 和 alpha 更新间隔
     - 20
     - 完成第 20 次 critic 更新后首次更新，之后每 20 次更新一次。
   * - Replay warmup / 初始更新次数
     - 1024 / 2000
     - 按 action-chunk transition 计数，不按 episode 或底层动作步数计数。
   * - UTD / 每轮更新上限
     - 20 / 10000
     - Warmup 后每新增一条 transition 获得 20 次更新额度；超出上限的额度保留。
   * - Entropy backup
     - False
     - 与共享参考配置默认值一致；消融实验需显式开启。

当 ``target_entropy: null`` 时，目标熵按执行 chunk 的所有坐标计算为 ``sum(log(edit_scale_i) - 0.5)``。对于 10 × 8 维动作和 0.2 的幅度，约为 -168.755033。参考值 -189.849412 对应 10 × 9 维动作，不能原样照搬。超过 ``sum(log(2 * edit_scale_i))`` 的目标不可达，配置验证会拒绝。也可用八元素列表指定每个动作维度的残差幅度，并在 chunk 内重复使用。

示例沿用 RLT 环境的 ``reward_mode: only_success`` 和 ``use_rel_reward: False``：成功插入奖励 +1，其余步骤奖励 0，包括未成功的失败和超时。成功后终止 episode，chunk 内后续步骤的奖励为 0。Critic 对每个 action chunk 内的奖励求和。设置 ``reward.use_reward_model: False`` 时，环境奖励原样传递；``reward.env_reward_weight`` 仅在混合 reward-model 输出时生效。

运行 ManiSkill 示例
~~~~~~~~~~~~~~~~~~~~~~~~

使用现有 OpenPI + ManiSkill 环境，并提供兼容的 π₀.₅ checkpoint，以及同一 Panda 关节控制数据集的归一化统计。通用基模可以加载，但若基模 proposal 无法提供有效动作，有界残差不保证能学会插孔。Dataconfig 名称中包含 ``rlt`` 是因为复用该数据集的 transforms；``openpi.use_rlt`` 处于关闭状态。

正式权重沿用 :doc:`../../examples/embodied/rlt` 的 Stage 1/SFT 产物：将 ``rollout.residual_base_model.model_path`` 指向训练输出的 ``checkpoints/global_step_<step>/actor`` 目录，保留其 ``model_state_dict/full_weights.pt``。示例中的 3000 仅表示所选训练步数，应替换为实际保存的 checkpoint。归一化统计显式使用同一 SFT 数据集的 ``norm_stats.json``，不要为正式训练生成临时统计。带 RLT head 的 Stage 1 checkpoint 也可提供 π₀.₅ backbone；此配方关闭 RLT head，仅加载基模用于冻结推理。

.. code-block:: bash

   source .venv-maniskill/bin/activate
   export EMBODIED_PATH="$(pwd)/examples/embodiment"
   export PYTHONPATH="$(pwd):${PYTHONPATH}"
   python examples/embodiment/train_embodied_agent.py \
     --config-name maniskill_sac_residual_openpi_pi05 \
     rollout.residual_base_model.model_path=/path/to/maniskill_rlt_stage1_sft_openpi_pi05/checkpoints/global_step_3000/actor \
     rollout.residual_base_model.openpi_data.norm_stats_path=/path/to/maniskill_peginsertionside_joint/norm_stats.json

示例使用一个 learner、FSDP ``no_shard``、FP32 残差网络和冻结的 BF16 π₀.₅ rollout 模型。当前配方不支持梯度累积、offload、人工干预和编译 rollout，任务仍包含完整的抓取与插入。

异步采集与训练
~~~~~~~~~~~~~~~~~~~~~~~~

需要让采集与训练重叠执行时，使用双 GPU 异步示例。该配置选择 10 mm 间隙的 ``PegInsertionSideHardClearance-v1``，将 learner 放在 GPU 0，将环境和冻结的 π₀.₅ rollout 放在 GPU 1。提供 1000-step SFT checkpoint，以及同一 10 mm 数据集的归一化统计：

.. code-block:: bash

   python examples/embodiment/train_async.py \
     --config-name maniskill_sac_residual_openpi_pi05_async \
     rollout.residual_base_model.model_path=/path/to/maniskill_pi05_10mm_sft/checkpoints/global_step_1000/actor \
     rollout.residual_base_model.openpi_data.norm_stats_path=/path/to/maniskill_peginsertionside_10mm_joint/norm_stats.json

入口使用 ``AsyncEmbodiedRunner`` 和原生异步 SAC 的后台轨迹接收线程。Learner 从 replay 更新时，采集继续运行。残差网络、loss、0/1 奖励、batch size 256、UTD 20，以及每 20 次 critic 更新一次 actor/alpha 的间隔，均与同步配方共用。每次 learner 调用结束后在后台同步权重；首次同步会等待初始或恢复的权重准确加载。

异步示例将每次 learner 调用限制为最多 400 次 critic 更新，随后返回以记录日志并同步权重，未用完的额度留到后续调用。Warmup 前或没有更新额度时，runner 等待且不推进步数。示例设置 ``runner.max_critic_updates: 800000``，learner 按剩余额度截短最后一次调用，runner 在累计完成 80 万次 critic batch 更新后保存并退出，总数包含 warmup。此上限替代 ``max_epochs`` / ``max_steps`` 的 runner 步数停止条件，仅支持异步 residual SAC 入口。进度表按 critic 更新计数；checkpoint 名称、``val_check_interval`` 和 ``save_interval`` 仍按实际发生更新的 learner 调用计数。恢复后按累计更新数计算上限，不会额外获得 80 万次额度。将该字段设为 null 可恢复按 runner 步数停止。达到 critic 更新上限时仍可能有未使用的 UTD 额度。

若要先训练 critic，再更新 residual actor，设置 ``algorithm.residual_schedule.critic_warmup_updates``，指定开始阶段仅训练 critic 的 batch 更新次数，默认值为 0。这期间 critic、共享输入投影和 target network 正常更新；actor MLP、温度及两者的 optimizer 和学习率调度器保持不变。之后 actor 和温度按配置的 critic 更新间隔更新。该参数与 ``warmup_transitions`` （学习前需采集的动作 chunk 条数）和 ``warmup_updates`` （初始更新额度）分别控制不同阶段。

DINOv3 EEF 特权输入配方先采集至少 4,000 条 transition，获得 20,000 次初始更新额度，并设置 ``critic_warmup_updates: 20000``。默认每 20 次 critic 更新一次 actor，因此首次 actor 更新发生在第 20,020 次 critic 更新。全新训练累计 800,000 次 critic 更新时，actor 和温度各更新 39,000 次。Critic 预热期间继续异步采集。

夹持 peg 的预插入训练应在 ``env.train`` 和 ``env.eval`` 中均设置 ``init_params.reconfiguration_freq: 1``，并使用 ``auto_reset: false`` 的整批 reset。重建 GPU 物理场景可清除上一局的接触和约束历史，代价是增加 reset 耗时。开启重建后，未显式指定 seed 的整批 reset 从 wrapper 独立维护的序列生成各环境的 seed；序列由 worker seed 初始化。显式 reset seed 原样传给模拟器，并可复现地重启该序列；固定 reset 模式保留固定 seed。部分 reset 沿用 ManiSkill 的 RNG 行为，不能同时重建整个场景。比较 checkpoint 时保持评测 seed 一致。

若需要按 critic 更新次数精确保存，将 ``runner.save_critic_interval: 100000`` 与 ``max_critic_updates`` 一起设置。此选项覆盖按 runner 步数保存的间隔，在每 10 万次 critic batch 更新及最终预算处保存。即使剩余更新额度会跨过里程碑，learner 也会在该处结束本次调用。checkpoint 目录名仍采用 runner 步数；读取 ``actor/residual_state_rank_0.pt`` 中的 ``budget.updates`` 确认实际 critic 步数。独立评测进程可对每份 checkpoint 使用 50 个固定种子，每得到一个结果即计算成功数除以已完成回合数。保存选项本身不调度评测，需要单独配置评测进程。

使用 DINOv3 视觉特征
~~~~~~~~~~~~~~~~~~~~

需要让 residual SAC 直接使用图像特征时，可使用 DINOv3 异步示例。Actor 和 critic 共用冻结 DINOv3-S/16 特征的可训练投影。冻结的 π₀.₅ 仍使用原来的 prompt 和 SFT 归一化统计生成基座动作；DINOv3 视觉编码器只接收 RGB 图像。

Rollout worker 将 ``main_images`` 和 ``wrist_images`` 通过 bicubic 插值缩放为 224 × 320，并应用 ImageNet 归一化。每个相机的 384 通道 patch 网格池化为 2 × 2，按相机顺序拼接后得到 3072 维特征。共用的视觉投影将其映射到 256 维；9 维状态仍投影到 128 维。Replay 只保存池化后的特征，learner 不运行 DINOv3，也不保存 RGB 帧。投影参数由 critic optimizer 更新，actor 更新时将其 detach。

每个 rollout 节点都需要本地官方 DINOv3 源码仓库和对应的 S/16 预训练权重。加载过程不自动下载资源。可选的 ``rollout.residual_dinov3.weights_sha256`` 会在加载前校验权重文件：

.. code-block:: bash

   python examples/embodiment/train_async.py \
     --config-name maniskill_sac_residual_openpi_pi05_dinov3_async \
     rollout.residual_dinov3.repo_path=/path/to/dinov3 \
     rollout.residual_dinov3.weights_path=/path/to/dinov3_vits16_pretrain.pth \
     rollout.residual_base_model.model_path=/path/to/maniskill_pi05_10mm_sft/checkpoints/global_step_1000/actor \
     rollout.residual_base_model.openpi_data.norm_stats_path=/path/to/maniskill_peginsertionside_10mm_joint/norm_stats.json

此示例保留 10 mm 任务、0/1 奖励、``edit_scale: 0.02``、``initial_alpha: 1e-5``、batch size 256、UTD 20 和 80 万次 critic 更新上限。Critic target 不包含 entropy backup。从 2048 维 prefix 特征切换时，需要重新初始化 residual 模型；原有投影和 replay 与 3072 维 DINOv3 特征不兼容。恢复 DINOv3 实验时，应保持相机顺序、预处理、backbone 权重、基座 checkpoint 和归一化统计一致。

使用 EEF 状态与 Rot6d 动作
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

10 mm EEF 任务使用 ``maniskill_sac_residual_openpi_pi05_dinov3_eef_async`` 配置和 ``pi05_maniskill_eef_rot6d`` SFT checkpoint。SFT 前需要重新采集示范，并基于完整新数据重新计算 pi05 分位数归一化统计。Joint 或四元数版本的 checkpoint 不具备相同的状态和动作语义。

使用 ``toolkits/lerobot/collect_maniskill_peg_lerobot_eef.py`` 采集经过成功筛选的 GPU 示范。MPlib 在 CPU 上规划，几何信息从 GPU 场景位姿读取；最终 EEF 动作通过 GPU 仿真和 GPU IK 执行并记录。评测和 rollout 也设置 ``sim_backend: gpu``。三条路径校验相同的 ``pd_ee_delta_pose`` 控制器：根坐标系增量、不累积目标，以及 ``alpha: 1.0`` 的 GPU Levenberg–Marquardt 求解器。此 EEF 配置会拒绝 CPU 执行。

采集器从棍子放在桌面的状态开始，记录抓取、搬运和插入的完整任务。所有移动阶段共用 ``--tcp-speed-m-s``\ （默认 ``0.04`` m/s）和 ``--tcp-angular-speed-rad-s``\ （默认 ``0.25`` rad/s）。采集器将关节参考目标转换为 TCP 位姿，按控制频率对每段移动路径重新采样，每个子段的时长取平移所需时间与旋转所需时间的较大值。因此，插入阶段参考点更密不会额外增加控制步数。在 10 Hz 下，默认每条指令最多平移 4 mm、旋转 0.025 rad（约 1.43°）。

使用默认速度采集新的完整任务数据集：

.. code-block:: bash

   PYTHONPATH=. python toolkits/lerobot/collect_maniskill_peg_lerobot_eef.py \
     --repo-id local/peg_fulltask_uniform_tcp \
     --sim-backend gpu --control-freq 10 --fps 10 --sim-freq 100 \
     --tcp-speed-m-s 0.04 --tcp-angular-speed-rad-s 0.25

夹爪闭合保留原有停留时长。移动阶段结束时，TCP 到达距终点 1 mm、姿态误差 0.01 rad 以内后才进入下一阶段；终点追踪使用相同限速，受阻时最多执行到回合时限。角速度限制、路径拐角、末尾不足一个周期的距离和接触都可能使平移减速。指令限速不保证 GPU IK 和接触条件下的实测速度恒定。每条示范的 ``tcp_speed_by_phase`` 记录各阶段指令与实测的线速度、角速度统计，终点追踪单独记为 ``*_settle`` 阶段。回合元数据和最终 manifest 均记录 ``tcp_motion``；``--resume`` 会拒绝缺少该记录或速度设置不同的数据。SFT 应使用新采集的数据，并重新计算归一化统计。

状态和模型动作均为 10 维。状态包含世界坐标系 TCP xyz、旋转矩阵前两列，以及以米为单位的夹爪总开度。动作包含归一化的 root-frame 平移增量、root-aligned 增量旋转矩阵前两列，以及夹爪指令。两列分别展开，顺序为 ``[r00, r10, r20, r01, r11, r21]``；不能通过逐元素相减构造旋转增量。输出变换将旋转解码为 XYZ Euler 增量，并除以当前安装的 ManiSkill 控制器的 ``rot_lower=-0.1``，最终向环境发送 7 维指令。

SFT 归一化之前，先对原始状态和动作的旋转执行 Gram–Schmidt 正交化。每次进入 actor 或 critic MLP 前，residual policy 先反归一化旋转分量，再进行正交化。这覆盖状态、actor 的基座动作输入，以及 critic 的叠加残差动作输入，使表示同一旋转的非退化 6D 向量得到一致的网络旋转特征。退化向量采用确定性的正交轴回退。其余分量保留归一化值。Replay 和残差采样仍使用模型归一化坐标；熵项计算的是旋转投影前 100 维残差的采样密度，并非 SO(3) 上的密度。

将 ``actor.model.eef_rot6d_norm_stats_path`` 和 ``rollout.residual_base_model.openpi_data.norm_stats_path`` 指向同一个新 ``norm_stats.json``。训练和评测环境均设置 ``wrap_obs_mode: openpi_eef_rot6d``、``control_mode: pd_ee_delta_pose``。启动校验会拒绝 joint/EEF 模式混配、维度冲突和统计不一致。π₀.₅ 内部仍补齐到 32 维，但 residual SAC 只使用 10 个执行动作各自的前 10 维。``edit_scale: 0.02`` 对应的自动目标熵为 ``100 * (log(0.02) - 0.5) = -441.2023``。Critic target 仍不包含 entropy backup。

特权 Actor/Critic 与几何尺寸
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

使用 ``maniskill_sac_residual_openpi_pi05_dinov3_eef_privileged_async``，可以在现有 DINOv3 和 EEF 输入之外，为 actor 和 critic 同时提供仿真真值。示例设置 ``actor.model.actor_use_privileged_state: true``，两者共用 31→128 特权投影。全部共享投影由 critic 优化器训练，actor 更新时对投影输出执行 detach。Actor 还以冻结 VLA 的基座动作为条件；冻结 VLA 本身仍只读取原有输入。

精简后的特权状态共 31 维，按以下顺序排列：

.. list-table:: 任务相对坐标中的特权观测
   :header-rows: 1
   :widths: 30 10 60

   * - 字段
     - 维度
     - 表示
   * - ``peg_head_hole``
     - 9
     - Peg 前端相对孔的 xyz（毫米）和 rot6d。
   * - ``margins``
     - 3
     - 带符号的成功条件余量：``[x + 0.015, radius - abs(y), radius - abs(z)] * 1000``。
   * - ``tcp_peg``
     - 9
     - TCP 相对 peg 主体的 xyz（毫米）和 rot6d。
   * - 接触力
     - 9
     - 左手指与 peg、右手指与 peg、peg 与盒子的接触力向量，均在孔坐标轴下表达。
   * - ``is_grasping``
     - 1
     - 当前是否抓持，值为 0 或 1。

位置和余量统一乘以 1000，2 mm 的对准偏差因此表示为 2；余量大于等于零表示满足对应成功条件。孔的随机尺寸体现在横向余量中。Peg 与孔、TCP 与 peg 的相对旋转均使用 rot6d，进入共享投影前执行 Gram–Schmidt 正交化。接触力采用 ``sign(f) * log1p(abs(f) / 10 N)``。输入中不再包含机器人连杆状态、绝对位姿、速度、单独的几何参数、关节目标或成功标志。这些换算不改变 SFT 归一化统计。

配置 ``env.train.critic_privileged_obs: true``、``env.eval.critic_privileged_obs: true`` 和 ``actor.model.privileged_state_dim: 31``，示例已经设置。Replay 保存每个动作块两端对应的特权状态，包括最终观测。Dense 示例将 500 步作为采集边界，从最终观测继续 bootstrap。Actor 在 rollout 和评测时也必须提供特权观测，缺失时会报错，不会自动补零。设置 ``actor.model.actor_use_privileged_state: false`` 可以保留仅 critic 使用特权状态的模式，此时 actor 可在没有特权观测时推理。

本示例在训练和评测中保留随机尺寸。Peg 全长在 170–250 mm 范围内采样，正方形截面的边长在 30–50 mm 范围内采样；孔的边长为 peg 边长加 20 mm，每侧间隙仍为 10 mm。Peg 和盒子的复位位姿继续随机，孔在盒内居中并随盒子移动。成功判据不变：peg 前端在孔坐标系中满足 x >= -15 mm，且 y/z 位于孔内。

后续进行固定尺寸对照时，可在训练和评测中同时设置 ``init_params.fixed_peg_half_length: 0.105`` 与 ``init_params.fixed_peg_half_width: 0.02``，对应 210×40×40 mm 的 peg、60×60 mm 的孔截面和 210 mm 的盒子深度，此时需要从孔口深入 90 mm。当前示例省略这两个字段，使用随机尺寸。

切换该配置需要重新初始化 residual learner 和 replay，因为 critic 输入及 Q-head 形状与无特权版本不同。旧 415 维版本的投影和 replay 也与新的 31 维布局不兼容。启用 actor 特权输入还会扩大 editor 的输入维度，因此不能直接续训仅 critic 使用特权状态的 checkpoint。现有 1000-step EEF SFT 基座及其配套统计仍可使用。示例保留异步 80 万次更新、batch size 256、UTD 20、``edit_scale: 0.02``、``initial_alpha: 1e-5``，critic target 不加入 entropy backup。尺寸变更不会自动重训 SFT 或替换其统计。

从夹持后的孔前状态开始
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

只训练对准和插入时，使用 ``examples/embodiment/config/env/maniskill_peg_preinsert.yaml`` 中的环境配置。Reset 将棍子放在闭合的夹爪中，尖端沿插入轴距盒子前表面 50 mm。尖端在孔坐标系的 Y/Z 方向分别加入 ±30 mm 的均匀随机偏移，XYZ 欧拉角分别加入 ±3° 的均匀随机扰动。盒子保留原有的 XY 位置和 yaw 采样范围；不可达的柜子位姿会重新采样，尖端偏移保持不变。该预设固定棍子尺寸为 210×40×40 mm，孔在盒子内居中，截面为 60×60 mm。

设置 ``init_params.hole_clearance: 0.001`` 可将每侧间隙缩小到 1 mm。棍子宽度保持 40 mm 时，孔截面为 42×42 mm。该参数同时更新碰撞几何和成功判据使用的孔半径；在 ``env.train.init_params`` 和 ``env.eval.init_params`` 中使用相同值。省略此参数时，沿用环境变体的默认间隙。复用其他间隙采集的示范时，应记录源数据的几何尺寸。

初始化的三个参数位于 ``init_params.preinsert``：

.. code-block:: yaml

   preinsert:
     gap_m: 0.05
     lateral_jitter_m: 0.03
     rotation_jitter_deg: 3.0

调整 RL 配置时，将相同参数块放入 ``env.train.init_params`` 和 ``env.eval.init_params``，并使采集与训练使用相同的几何尺寸。距离沿孔轴从盒子前表面计算，而不是从孔中心计算：该预设的初始尖端坐标为 X = −155 mm。原有成功条件保持不变，即 X ≥ −15 mm 且 Y/Z 位于孔内，因此从初始位置到成功需要前进约 140 mm。预设时间限制为 100 步，控制频率为 10 Hz，沿用分阶段 dense reward。

初始化通过 GPU IK 求解 Panda 位姿，只设置被重置环境中的机器人和棍子状态，不推进其他环境的物理时间。抓持点位于棍子局部 X = −0.6 倍半长处。第一次物理步刷新接触缓存前，接触观测保持为零；之后根据真实手指接触力判定抓持。

采集五条示范并回放保存的动作：

.. code-block:: bash

   PYTHONPATH=. python toolkits/lerobot/collect_maniskill_peg_preinsert.py \
     --config examples/embodiment/config/env/maniskill_peg_preinsert.yaml \
     --output /path/to/new_preinsert_preview \
     --num-episodes 5 --seed 310000

控制器先对准，再推进棍子。每条保存的轨迹必须在采集时成功，并且在相同 seed 下重新 reset、解码保存的动作、通过 GPU 仿真回放后再次成功；初始化和专家 TCP 增量转关节目标均使用 GPU IK。输出包含 LeRobot ``dataset/``、回放 ``videos/``、动作与状态 ``traces/``、``env.yaml`` 和 ``manifest.json``。每条保留的示范都保存侧视、主相机和腕部相机视频。Manifest 记录尝试过的 seed、成功情况及 replay 状态差异；这些结果用于验证专家示范，不代表 SFT 或 RL policy 的成功率。

默认状态和动作使用 10 维 EEF/rot6d 表示。正式 SFT 前，重新采集训练数据并计算归一化统计；随后将训练得到的模型及配套统计用于 residual learner 的基座。

关节模式下，复制该环境配置并设置 ``wrap_obs_mode: rlt_openpi_joint``、``init_params.control_mode: pd_joint_delta_pos`` 和 ``critic_privileged_obs: false``，然后通过 ``--config`` 指向复制后的文件。同一个 collector 会保存 9 维关节状态和 8 维归一化控制动作。初始化与专家 TCP 增量转关节目标均使用 GPU IK，实际执行和 replay 使用关节 PD 控制，夹爪保持闭合。关节和 EEF 两种表示都需要重新采集示范、计算 π₀.₅ 归一化统计并训练匹配的 SFT；两种 checkpoint 不能互换。

恢复与监控
~~~~~~~~~~~~~~~~~~~~~~~~

Checkpoint 保存 optimizer 状态、学习得到的温度、target model、replay、更新额度和 learner RNG。通过 ``runner.resume_dir`` 恢复时，actor 更新间隔和待执行的 UTD 额度从保存的计数继续。恢复后沿用当前配置的 ``max_updates_per_iteration``，因此同步 checkpoint 可以改用较短的异步 learner 调用。UTD、采集预热、初始更新额度和仅训练 critic 的预热设置必须与保存时一致。旧 checkpoint 缺少仅训练 critic 的字段时按 0 处理。Runner 步数也会恢复，设置后续运行长度时需要考虑步数含义的变化。恢复范围是 learner 状态，不包括运行中的仿真状态和 rollout RNG。

上述示例均启用 TensorBoard。关注 ``env/success_once``、``train/critic/td_error_abs``、``train/actor/entropy``、``train/sac/alpha``、``train/sac/critic_updates`` 和 ``train/sac/pending_updates``。比较运行结果时，同时查看采集的 transition 数、critic 更新数和实际耗时。CPU 检查覆盖异步更新调度和回放接收；异步 GPU 吞吐量与 ManiSkill 学习曲线尚未验证。

异步 residual SAC 的 ``env/collected_success_rate`` 表示累计采集成功回合数除以已上报的全部已完成采集回合数，包含 warm-up。取值范围为 0 到 1，每个回合只计一次，不按回合长度或 worker 的 batch 大小加权。``env/success_once`` 仍表示当前 rollout batch 的成功率。评测回合、replay 采样和旧数据淘汰均不改变累计计数。首批回合结果上报后才显示此指标，仍在执行的回合暂不计入。

runner 将已上报的计数保存在 actor checkpoint 同级的 ``collection_metrics.json`` 中，并在 resume 时恢复。旧 checkpoint 若没有此文件，计数从恢复训练时开始，同时输出警告。

分阶段插孔 Dense Reward
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

特权 EEF 示例使用 ``reward_mode: peg_staged_dense``，最小化接近棍子、对齐孔口、完成插入三项距离成本。训练和评测都设置 ``use_rel_reward: false``、``ignore_terminations: false``。成功当步奖励为零并终止 episode。500 步时间限制截断采集，``algorithm.bootstrap_time_limits: true`` 让 critic 从 reset 前的最终观测继续 bootstrap。任务成功条件和 ``eval/success_once`` 沿用原有插孔判定。

接近目标是 TCP 在棍子局部坐标系中的一个包含边界的区域，原点位于棍子中心，+X 指向插入端。使用成功的 GPU EEF 示范统计边界：

.. code-block:: bash

   PYTHONPATH=. python toolkits/lerobot/analyze_maniskill_peg_grasps.py \
     --dataset /path/to/maniskill_peginsertionside_10mm_eef_rot6d \
     --output /path/to/grasp_statistics.json \
     --range-margin-m 0.005

工具用 GPU 仿真和 GPU IK 重放保存的动作与 seed，核对 EEF 状态，只保留成功且连续至少三次检测到抓持的轨迹。先计算每条轨迹的夹持位置中位数，再对各轨迹中位数逐坐标取 10% 和 90% 分位数，并向两侧扩展指定容差。将 ``grasp_min_peg_m`` 和 ``grasp_max_peg_m`` 填入 ``env.train.peg_dense_reward`` 与 ``env.eval.peg_dense_reward``。示例根据 400 条示范得到的区域约为 X=[-75.26, -49.72]、Y=[-5.15, 5.04]、Z=[-6.08, 6.00] mm。区域只用于接近距离：一旦实际满足 ``is_grasping``，即使 TCP 位于区域外，也解锁对齐项。

三项距离都以米计算：

* 接近距离为 TCP 到区域最近点的 L2 距离，区域内为零。尺度 ``reach_scale_m`` 为 0.05。
* 对齐距离同时包含棍子前端到孔轴的 YZ 偏移和插入轴方向误差。在孔坐标系中，对拼接向量 ``[head_y, head_z, L * (peg_axis - hole_axis)]`` 求 L2 范数，其中 L 是当前棍子半长，孔轴为 [1, 0, 0]。这样两部分都具有米的量纲，反向的插入轴也会受到惩罚。尺度 ``alignment_scale_m`` 为 0.05。
* 剩余插入距离为孔坐标系中的 ``max(-0.015 - head_x, 0)``，尺度 ``depth_scale_m`` 为 0.10。只有保持抓取、横向距离不超过 ``alignment_tolerance_m``\ （2 mm）、轴向夹角不超过 ``axis_tolerance_deg``\ （5°）时才解锁深度项。

每项已解锁距离采用 ``c = d / (d + scale)`` 归一化，位于 [0, 1)。未解锁的对齐项或深度项固定罚 1：

.. math::

   r = -\left(c_{\mathrm{reach}}
       + \begin{cases}c_{\mathrm{align}} & \text{grasped}\\1 & \text{otherwise}\end{cases}
       + \begin{cases}c_{\mathrm{depth}} & \text{aligned and grasped}\\1 & \text{otherwise}\end{cases}\right).

成功时三项成本全部置零。奖励始终位于 [-3, 0]。同一几何状态下，解锁把固定成本 1 替换成小于 1 的实际成本，因此解锁本身不会降低奖励。阶段取决于当前接触与位姿，松手或失去对齐会重新锁定相应成本。距离增大时成本平滑增加，不会因硬裁剪而完全失去距离信号。

每个控制步支付当前状态的成本。Residual SAC 将十步 chunk 的奖励相加，仅在 critic target 中应用一次 gamma=0.99。Episode 结束后的冻结步奖励为零。未成功的静止状态会持续支付负成本，成功终止后不再支付后续成本。这是距离成本目标，不是势能差：episode return 是沿途成本之和，取决于耗时和路径；实现中没有需要维护或 reset 的势能基线，也没有奖励侧的折扣。

Episode 指标提供 ``dense_reward``、``dense_grasp_cost``、``dense_alignment_cost`` 和 ``dense_depth_cost``，便于核对分项之和。``dense_stage`` 为 0=接近、1=对齐、2=插入、3=成功；``dense_grasped`` 和 ``dense_aligned`` 表示门槛状态。距离诊断包括 ``grasp_region_error_mm``、``alignment_error_mm``\ （仅横向）、``alignment_distance_mm``\ （组合 L2）、``alignment_axis_error_deg``、``depth_remaining_mm`` 和 ``insertion_depth_mm``。它们是当前状态快照，``episode/return`` 累加奖励。

保留 ``reward.use_reward_model: false``，让负奖励原样进入 replay。切换奖励定义时重新初始化 residual learner 和 replay，保留 SFT checkpoint 与归一化参数。除 return 外，单独检查 500 步内的实际 SR 和完成时间；继续 bootstrap 的训练时域与评测时限并不相同。

EXPO 候选动作选择
~~~~~~~~~~~~~~~~~

冻结基座能在同一状态下提出多个动作 chunk 时，可以用 EXPO 让 critic 从中选择。本实现参考本地 gr00t 的 ``algorithms/expo``，沿用 residual SAC 的 replay、异步采集、优化器和有界修正网络，调整动作选择方式及 actor 目标。

每次观测先由冻结基座独立采样 8 个动作 chunk，再对每个 chunk 采样一次修正，共得到 16 个候选动作。EXPO 随机抽取两个 target critic，用两者较小的 Q 给所有候选打分，执行得分最高的动作。选择阶段的 editor 使用 target 状态／视觉投影。多个基座候选共用图像／prompt 的 prefix 编码和 DINOv3 特征，但 8 次去噪依次执行，因此采集开销高于只有一个基座候选的 residual SAC。修正上限相对于各自的基座候选定义；切换到另一个基座候选时，动作变化可以超过一次有界 residual 修正。

Replay 保存实际执行的归一化动作，以及真实下一观测的全部候选，后者形状为 ``[num_candidates, chunk_dim]``。超时转移使用 reset 前最终观测的候选。计算 critic 目标时，先在这些缓存候选上重新选择动作，再独立抽取一对 target critic 做 Bellman backup：

.. math::

   y = r + \gamma (1-d)\min_{j\in J'}Q^{\mathrm{target}}_j(s',a^*).

成功终止 bootstrap，超时默认继续 bootstrap。EXPO 要求 ``algorithm.backup_entropy`` 为 false。Actor 围绕 replay 中实际执行的动作学习修正，使用停止梯度的 online 特征，以全部 online critic 的平均 Q 和 SAC 熵项构造目标。Alpha 继续自动学习，edit scale 的 Jacobian 和 target entropy 计算与 residual SAC 相同。

异步配置为 ``maniskill_expo_openpi_pi05_dinov3_eef_async.yaml``，使用冻结的 SFT 2000-step 基座、DINOv3 和 EEF 本体状态，输入 rotation 经 rot6d 正交化，不使用特权信息。任务从夹爪已抓住棍子的状态开始：尖端距孔口 5 cm，横向扰动为 3 cm，旋转扰动为 3 度。棍子尺寸固定为 210 x 40 x 40 mm，孔的单侧余量为 1 mm，使用稀疏 0/1 reward。主要参数如下：

.. list-table:: EXPO 训练默认参数
   :header-rows: 1

   * - 参数
     - 值
   * - ``actor.model.num_candidates`` / ``actor.model.num_edits``
     - 8 / 8，对前 8 个基座候选各采样一次修正
   * - ``actor.model.edit_scale`` / 初始 alpha
     - 0.02 / 1e-5
   * - Batch size / UTD / actor 更新间隔
     - 256 / 20 / critic warmup 后每 20 次 critic 更新一次 actor
   * - Warmup
     - 采集 4000 个 chunk transition，再做 20000 次仅 critic 更新
   * - Critic 总预算
     - 800000 次，含 warmup；对应 39000 次 actor 更新
   * - Checkpoint 与评测
     - 每 100000 次 critic 更新，评测 50 局并保存视频

启动前设置基座 checkpoint、两处归一化文件路径，以及 DINOv3 源码和权重路径。两处归一化路径必须指向同一份 SFT 统计。在已有 OpenPI 与 ManiSkill 环境中，从仓库根目录执行：

.. code-block:: bash

   export EMBODIED_PATH="$PWD/examples/embodiment"
   PYTHONPATH="$PWD" python examples/embodiment/train_async.py \
     --config-path "$EMBODIED_PATH/config" \
     --config-name maniskill_expo_openpi_pi05_dinov3_eef_async \
     rollout.residual_base_model.model_path=/path/to/sft_2000/actor \
     rollout.residual_base_model.openpi_data.norm_stats_path=/path/to/norm_stats.json \
     actor.model.eef_rot6d_norm_stats_path=/path/to/norm_stats.json \
     rollout.residual_dinov3.repo_path=/path/to/dinov3 \
     rollout.residual_dinov3.weights_path=/path/to/dinov3_vits16_pretrain.pth

命令在 GPU 0 上启动 learner，在 GPU 1 上运行冻结基座推理和仿真。``runner.val_critic_interval`` 和 ``runner.save_critic_interval`` 都按已完成的 critic 更新计数，包含 warmup。Learner 在指定更新次数返回，评测等待对应权重同步至 rollout 后开始。EXPO 模型统一持有 target 投影和 target critic；它们随权重同步和 checkpoint 保存，但不进入优化器。通过 ``runner.resume_dir`` 加载 EXPO checkpoint，会同时恢复 replay、优化器、target 权重、更新额度和 RNG 状态。

EXPO 在评测时仍会采样修正动作和 critic 子集。严格配对比较应在独立 evaluator 中同时固定场景种子和 policy 采样种子；仅固定场景种子不能固定候选选择。内置的 50 局评测会在不同 checkpoint 间推进环境种子序列，并将每批并行回合保存为一个平铺视频。TensorBoard 保留共享的 ``train/sac/critic_updates`` 命名空间、``train/actor_q_to_entropy_grad_ratio`` 和累计 ``env/collected_success_rate``。``train/critic/target_edit_fraction`` 表示 Bellman 目标选择修正候选的比例。这些训练指标不能替代独立评测的 SR。


选择动作表示和特征
~~~~~~~~~~~~~~~~~~

下表列出算法、动作表示和冻结特征源的异步配置组合。将配置名传给 ``examples/embodiment/train_async.py`` 的 ``--config-name`` 参数即可选择对应组合。

.. list-table:: Residual 学习配置
   :header-rows: 1

   * - 算法
     - 状态／动作表示
     - 特征
     - 配置名
   * - Residual SAC
     - 关节
     - π₀.₅ prefix 均值
     - ``maniskill_sac_residual_openpi_pi05_async``
   * - Residual SAC
     - 关节
     - DINOv3
     - ``maniskill_sac_residual_openpi_pi05_dinov3_async``
   * - Residual SAC
     - EEF rot6d
     - π₀.₅ prefix 均值
     - ``maniskill_sac_residual_openpi_pi05_eef_async``
   * - Residual SAC
     - EEF rot6d
     - DINOv3
     - ``maniskill_sac_residual_openpi_pi05_dinov3_eef_async``
   * - EXPO
     - 关节
     - π₀.₅ prefix 均值
     - ``maniskill_expo_openpi_pi05_async``
   * - EXPO
     - 关节
     - DINOv3
     - ``maniskill_expo_openpi_pi05_dinov3_async``
   * - EXPO
     - EEF rot6d
     - π₀.₅ prefix 均值
     - ``maniskill_expo_openpi_pi05_eef_async``
   * - EXPO
     - EEF rot6d
     - DINOv3
     - ``maniskill_expo_openpi_pi05_dinov3_eef_async``

关节配置使用 9 维本体状态和 8 维动作；EEF 配置使用 10 维状态和动作，其中旋转采用正交化的 rot6d。必须提供与所选表示匹配的 SFT checkpoint 和 ``norm_stats.json``；EEF 还要求 ``actor.model.eef_rot6d_norm_stats_path`` 指向同一份统计。现有 residual SAC 示例默认使用完整 10 mm 任务，EXPO 示例默认使用 1 mm 夹持预插入任务。对比算法前，应统一场景尺寸、episode 时限、reset 设置和 warmup。

未设置 ``rollout.residual_dinov3`` 或将其设为 null 时，actor 和 critic 使用 π₀.₅ 全部有效 prefix 输出的 2048 维 masked mean，包括图像、prompt 和离散本体状态 tokens，排除 padding。这是全局均值，不保留逐相机的空间池化网格。启用 DINOv3 时，每个相机先保留 2 × 2 网格，再拼接成 3072 维特征。两种 backbone 均冻结；共享的状态与视觉投影由 critic 优化器训练。切换表示或特征源时，应重新初始化 residual learner 和 replay，因为输入维度与缓存特征发生变化。

有界 edit 定义在 SFT 归一化动作坐标中，因此 ``edit_scale: 0.02`` 在关节和 EEF 空间中不代表相同的物理修正权限。对于十步 chunk，自动目标熵在关节配置下为 ``80 * (log(0.02) - 0.5) = -352.9618``，EEF 配置下为 ``100 * (log(0.02) - 0.5) = -441.2023``。两种配置的 critic backup 均不加入熵项。
