Soft Actor-Critic (SAC)
==================================

1. Introduction
---------------

Soft Actor-Critic (SAC) is one of the most widely used reinforcement learning (RL) algorithms.  
It consists of two key components:

- Actor (Policy Model): generates actions based on the current state.
- Critic (Q-value Model): evaluates the value of current observations and the chosen actions.

Soft Actor-Critic (SAC) is an off-policy deep reinforcement learning algorithm for continuous control. 
It is based on the maximum entropy reinforcement learning framework, 
which augments the standard reward objective with an entropy term to encourage exploration and robustness. 
SAC simultaneously learns a stochastic policy and two Q-functions, 
using entropy-regularized Bellman backups and automatic temperature tuning. 
Due to its sample efficiency and stability, 
SAC has been widely applied in robotics and continuous control benchmarks. 

For more details, see the original SAC paper 
`SAC <https://arxiv.org/abs/1801.01290>`_.


2. Objective Function
----------------------

Let the policy be :math:`\pi`. Then the Q function for :math:`\pi` is defined as: :math:`Q^{\pi}(s, a)`. In SAC, the Q function satisfies the following soft Bellman equation:

.. math::

   Q^{\pi}(s, a) = \mathbb{E}_{s' \sim P, a \sim \pi} \left[
      r(s, a) + \gamma (Q^{\pi}(s', a') + H(\pi(\cdot|s')))
   \right]
   = \mathbb{E}_{s' \sim P, a \sim \pi} \left[
      r(s, a) + \gamma (Q^{\pi}(s', a') - \alpha \log \pi(a'|s'))
   \right].

Here :math:`\gamma` is the discount factor, :math:`H` is the entropy of the policy, and :math:`\alpha` is the temperature parameter that determines the relative importance of the entropy term against the reward.

Therefore, the loss for the i-th Q-function :math:`Q_{\phi_{i}}` is as follows:

.. math::

   L(\phi_{i}, D) = \mathbb{E}_{(s, a, r, s', d) \sim D} \left[
      \frac{1}{2} \left(
         Q_{\phi_{i}}(s, a) - (r + \gamma (1 - d)(\min_{i} Q_{\overline{\phi_{\text{targ}, i}}}(s', a') - \alpha \log \pi_{\theta}(a'|s')))
      \right)^2
   \right],

where :math:`D` is the replay buffer, :math:`\overline{\phi_{\text{targ}, i}}` are the parameters of the target Q-network, and :math:`a'` is sampled from the current policy :math:`\pi_{\theta}`.

The policy :math:`\pi_{\theta}` is to maximize the expected Q value and entropy. Therefore, the policy loss is defined as follows:

.. math::

   L(\theta, D) = \mathbb{E}_{s \sim D, a \sim \pi_{\theta}} \left[
      \alpha \log \pi_{\theta}(a|s) - \min_{i} Q_{\phi_i}(s, a)
   \right].

In practice, the temperature coefficient :math:`\alpha` is learnable. Then the alpha loss is defined as follows:

.. math::

   L(\alpha, D) = - \alpha (H_{\text{targ}} - H(\pi(\cdot, d))), 

where :math:`H_{\text{targ}}` is a hyperparameter representing the target value for entropy. It is typically set to negative action dimension. 


3. Configuration
-----------------

Currently, SAC is supported only for embodied tasks in our framework.  
The algorithm configuration is defined as follows:

.. code-block:: yaml

   algorithm:
      update_epoch: 32
      group_size: 1
      agg_q: min # ["min", "mean"]. Option to aggregate multiple Q-values.


      adv_type: embodied_sac
      loss_type: embodied_sac
      loss_agg_func: "token-mean"
      
      bootstrap_type: standard # [standard, always]. Bootstrap Q-values according to terminations and truncations. "standard" only bootstraps when truncations, while "always" bootstraps when truncations or terminations.
      gamma: 0.8 # Discount factor.
      tau: 0.01  # Soft update coefficient for target networks
      target_update_freq: 1  # Frequency of target network updates
      entropy_tuning:
         alpha_type: softplus  # ["softplus","exp","fixed_alpha"]
         initial_alpha: 0.01  # Initial temperature value
         target_entropy: -4  # Target entropy (-action_dim)
         optim:
            lr: 3.0e-4  # Learning rate for temperature parameter
            lr_scheduler: torch_constant
            clip_grad: 10.0
      
      # Replay buffer settings
      replay_buffer:
         enable_cache: True # Enable memory cache to reduce I/O overhead
         cache_size: 6000  # number of trajectories cached in memory
         sample_window_size: 6000  # number of latest trajectories to sample from for replay buffer
         min_buffer_size: 2  # Minimum buffer size before training starts (in number of trajectories)

4. Residual SAC With a Frozen VLA
---------------------------------

Use residual SAC to learn bounded corrections to a frozen OpenPI action chunk.
The ManiSkill example uses π₀.₅, Panda joint control and
``PegInsertionSideWideClearance-v1``. It trains small FP32 heads while the
rollout worker runs the frozen VLA. It does not require RLT token training.

By default, the actor conditions on normalized proprioception, a masked mean of
the frozen VLA prefix embeddings, and the original base action. The DINOv3
example below replaces the visual features for both actor and critic. The critic receives the
projected features and the composed action. Both projections belong to the critic
optimizer; actor updates detach them. One frozen proposal is sampled per decision
and stored with its features in replay.

Actions and Targets
~~~~~~~~~~~~~~~~~~~

Apply the residual in normalized model-action coordinates:
``action = base_action + edit_scale * tanh(z)``. The rollout worker then
decodes the composed action through the base model's output transforms.
Replay retains the normalized composed action, not the decoded environment
action. The density includes the tanh and scale Jacobians exactly once.

The example leaves normalized actions unclipped. This preserves proposals
outside [-1, 1]. Setting ``actor.model.action_clipping: final`` clips the
composed normalized action; its entropy still describes the residual before
clipping. The environment's own actuator bounds remain in effect.

Critic targets sum rewards within the executed chunk and apply ``gamma``
once per decision. True terminations stop bootstrap; by default, time limits
bootstrap from the final observation. Both actor scoring and target evaluation take the minimum
of two critics sampled without replacement. All critic heads learn from the same
target. Entropy backup is controlled separately by ``algorithm.backup_entropy``.

Hyperparameters
~~~~~~~~~~~~~~~

The example follows the shared ``sac_residual`` recipe in
``groot/core/rl/algorithms/sac_residual``. It does not reproduce the
N2-specific warmup recipe or its reported success rates. The π₀.₅ features,
Panda action coordinates and simulator reward are different.

.. list-table::
   :header-rows: 1
   :widths: 35 20 45

   * - Setting
     - Default
     - Meaning
   * - Action chunk
     - 10 × 8
     - Ten Panda joint-control actions.
   * - Residual scale / initial log std
     - 0.2 / -1.5
     - Zero initial mean; log std is clamped to [-5, 2].
   * - Hidden layers / critic ensemble
     - 3 × 256 / 10
     - Linear, LayerNorm and ReLU; random two-critic minimum.
   * - State / prefix projections
     - 128 / 256
     - Learned projections of 9-D state and 2048-D frozen prefix features.
   * - Batch / learning rates
     - 256 / 3e-4
     - Adam for actor, critic and temperature; betas (0.9, 0.999), epsilon 1e-8.
   * - Gradient norm / initial alpha
     - 1.0 / 2e-4
     - Clip each optimizer's parameters separately; exponential alpha.
   * - Gamma / Polyak tau
     - 0.99 / 0.005
     - Per-chunk discount; target update after every critic update.
   * - Actor and alpha interval
     - 20
     - First update after 20 completed critic updates, then every 20.
   * - Replay warmup / initial updates
     - 1024 / 2000
     - Counts action-chunk transitions, not episodes or low-level actions.
   * - UTD / per-iteration cap
     - 20 / 10000
     - Twenty updates per new transition after warmup; capped work remains pending.
   * - Entropy backup
     - False
     - Matches the shared reference default; enable explicitly for that ablation.

With ``target_entropy: null``, resolve the target as
``sum(log(edit_scale_i) - 0.5)`` over all executed chunk coordinates.
For 10 × 8 actions with scale 0.2, this is approximately -168.755033.
The reference's -189.849412 is for 10 × 9 actions and must not be copied unchanged.
A target above ``sum(log(2 * edit_scale_i))`` is rejected as unreachable.
Per-action scales can be supplied as an eight-element list and repeat across
the chunk.

The example uses the RLT environment's ``reward_mode: only_success`` with
``use_rel_reward: False``: successful insertion receives +1, and all other
steps receive 0, including unsuccessful failures and timeouts. Success
terminates the episode, and subsequent steps in the chunk receive 0.
The critic sums these rewards within each action chunk. With
``reward.use_reward_model: False``, environment rewards pass through unchanged;
``reward.env_reward_weight`` applies only when mixing reward-model outputs.

Run the ManiSkill Example
~~~~~~~~~~~~~~~~~~~~~~~~~

Use the existing OpenPI + ManiSkill environment. Supply a compatible π₀.₅
checkpoint and the normalization statistics for the same Panda joint-control
dataset. A generic base checkpoint can load, but bounded residuals do not
guarantee it can learn insertion without a useful base proposal. The dataconfig
name contains ``rlt`` because it reuses that dataset's transforms;
``openpi.use_rlt`` is disabled.

For production runs, follow the Stage 1/SFT artifact layout in :doc:`../../examples/embodied/rlt`. Set ``rollout.residual_base_model.model_path`` to the training output directory ``checkpoints/global_step_<step>/actor``, retaining its ``model_state_dict/full_weights.pt``. The example selects step 3000; replace it with an actual saved checkpoint. Explicitly use ``norm_stats.json`` from the same SFT dataset instead of generating temporary statistics for training. A Stage 1 checkpoint with an RLT head can also supply the π₀.₅ backbone; this recipe disables the RLT head and loads the base model for frozen inference.

.. code-block:: bash

   source .venv-maniskill/bin/activate
   export EMBODIED_PATH="$(pwd)/examples/embodiment"
   export PYTHONPATH="$(pwd):${PYTHONPATH}"
   python examples/embodiment/train_embodied_agent.py \
     --config-name maniskill_sac_residual_openpi_pi05 \
     rollout.residual_base_model.model_path=/path/to/maniskill_rlt_stage1_sft_openpi_pi05/checkpoints/global_step_3000/actor \
     rollout.residual_base_model.openpi_data.norm_stats_path=/path/to/maniskill_peginsertionside_joint/norm_stats.json

The example uses one learner with FSDP ``no_shard``, FP32 heads and a frozen
BF16 π₀.₅ rollout model. Gradient accumulation, offload, interventions and
compiled rollout are not supported by this recipe.
It retains the simulator's full grasp-and-insert task.

Run Collection and Learning Asynchronously
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For overlapping collection and learning, use the two-GPU asynchronous example.
It selects ``PegInsertionSideHardClearance-v1`` with 10 mm clearance, places the
learner on GPU 0, and places the environment and frozen π₀.₅ rollout on GPU 1.
Provide the 1000-step SFT checkpoint and normalization statistics from the same
10 mm dataset:

.. code-block:: bash

   python examples/embodiment/train_async.py \
     --config-name maniskill_sac_residual_openpi_pi05_async \
     rollout.residual_base_model.model_path=/path/to/maniskill_pi05_10mm_sft/checkpoints/global_step_1000/actor \
     rollout.residual_base_model.openpi_data.norm_stats_path=/path/to/maniskill_peginsertionside_10mm_joint/norm_stats.json

This entry uses ``AsyncEmbodiedRunner`` and native asynchronous SAC's background
trajectory receiver. Collection continues while the learner updates from replay.
The residual network, losses, 0/1 reward, batch size 256, UTD 20 and actor/alpha
interval of 20 critic updates are shared with the synchronous recipe. Weight
synchronization runs in the background after each learner call; the initial
synchronization waits for the exact initial or resumed weights.

The asynchronous example caps each learner call at 400 critic updates, then
returns for logging and weight synchronization. Pending updates carry over to
later calls. Before warmup or when no update credits remain, the runner waits
without advancing its step. The example sets ``runner.max_critic_updates: 800000``:
the learner clips the last call to the remaining budget, and the runner saves
and exits at exactly 800,000 critic batch updates, including warmup. This limit
replaces the runner-step stopping condition from ``max_epochs`` / ``max_steps``;
it is supported by the asynchronous residual SAC entry only. The progress table
counts critic updates. Checkpoint names, ``val_check_interval`` and
``save_interval`` count learner calls that perform updates. On resume, the limit
is cumulative, not an additional allowance. Set it to null to use the runner-step
limit. Unused UTD credits may remain when the critic limit is reached.

To train the critic before updating the residual actor, set
``algorithm.residual_schedule.critic_warmup_updates`` to the number of initial
critic batch updates. Its default is zero. During this interval, the critic,
shared input projections and target network update; the actor MLP, temperature
and their optimizers and learning-rate schedules remain unchanged. Afterwards,
the actor and temperature update at the configured critic interval. This is
separate from ``warmup_transitions`` (collected action-chunk rows required before
learning) and ``warmup_updates`` (initial update credits).

The privileged DINOv3 EEF recipe collects at least 4,000 transitions and grants
20,000 initial updates, with ``critic_warmup_updates: 20000``. With the default
interval of 20, the first actor update is critic update 20,020. A fresh run of
800,000 critic updates therefore performs 39,000 actor and temperature updates.
Collection continues asynchronously during critic warmup.

For held-peg preinsert training, set ``init_params.reconfiguration_freq: 1`` in
both ``env.train`` and ``env.eval`` and use full resets with ``auto_reset: false``.
Reconstructing the GPU physics scene removes contact and constraint history from
the preceding episode, at the cost of additional reset time. Unseeded full resets
with reconfiguration enabled draw per-environment seeds from a wrapper-owned
sequence initialized by the worker seed. Explicit reset seeds remain unchanged
and restart this sequence reproducibly; fixed-reset mode retains its fixed seed.
Partial resets retain ManiSkill's RNG handling and cannot rebuild the full scene.
Keep explicit evaluation seeds fixed when comparing checkpoints.

For checkpoints at exact critic-update milestones, set
``runner.save_critic_interval: 100000`` together with ``max_critic_updates``.
This overrides the runner-step save interval and saves at every 100,000 critic
batch updates and at the final budget. The learner ends a call at the next
milestone even when the pending update count would cross it. Checkpoint directory
names still use runner steps; read ``budget.updates`` from
``actor/residual_state_rank_0.pt`` to identify the critic step. An independent
evaluation process can evaluate each saved policy on 50 fixed seeds, reporting
successes divided by completed episodes after each result. This save setting
does not schedule evaluations itself; configure the evaluation process separately.

Use DINOv3 Visual Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To condition residual SAC on images independently of the VLA prefix, use the
DINOv3 asynchronous example. Both actor and critic share a trainable projection
of frozen DINOv3-S/16 features. The frozen π₀.₅ model still generates base actions
using its original prompt and SFT normalization statistics; the DINOv3 visual
encoder receives only RGB images.

The rollout worker resizes ``main_images`` and ``wrist_images`` to 224 × 320 with
bicubic interpolation and applies ImageNet normalization. It pools each camera's
384-channel patch grid to 2 × 2, producing 3072 features in camera order. The
shared visual projection maps these to 256 dimensions; the 9-dimensional state
still projects to 128 dimensions. Only the pooled features enter replay, so the
learner does not run DINOv3 or store RGB frames. The critic optimizer updates the
projections, while actor updates detach them.

Provide a local official DINOv3 repository and matching S/16 pretrained weights
on each rollout node. Assets are loaded locally without downloading. The optional
``rollout.residual_dinov3.weights_sha256`` checks the weight file before loading:

.. code-block:: bash

   python examples/embodiment/train_async.py \
     --config-name maniskill_sac_residual_openpi_pi05_dinov3_async \
     rollout.residual_dinov3.repo_path=/path/to/dinov3 \
     rollout.residual_dinov3.weights_path=/path/to/dinov3_vits16_pretrain.pth \
     rollout.residual_base_model.model_path=/path/to/maniskill_pi05_10mm_sft/checkpoints/global_step_1000/actor \
     rollout.residual_base_model.openpi_data.norm_stats_path=/path/to/maniskill_peginsertionside_10mm_joint/norm_stats.json

This example retains the 10 mm task, 0/1 reward, ``edit_scale: 0.02``,
``initial_alpha: 1e-5``, batch size 256, UTD 20 and 800,000 critic-update budget.
The critic target excludes the entropy backup. Start a new residual model when
switching from the 2048-dimensional prefix features; its projection and replay
are incompatible with the 3072-dimensional DINOv3 features. To resume a DINOv3
run, keep the same camera order, preprocessing, backbone weights, base checkpoint
and normalization statistics.

Use EEF States and Rot6d Actions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For the 10 mm EEF task, select
``maniskill_sac_residual_openpi_pi05_dinov3_eef_async`` and a
``pi05_maniskill_eef_rot6d`` SFT checkpoint. Collect new demonstrations and
recompute the full-dataset pi05 quantile statistics before SFT. A joint or
quaternion checkpoint does not supply the same state/action contract.

Use ``toolkits/lerobot/collect_maniskill_peg_lerobot_eef.py`` to collect
success-filtered GPU demonstrations. It uses CPU MPlib planning with geometry
read from GPU poses, then executes and records EEF actions through GPU
simulation and GPU IK. Set ``sim_backend: gpu`` for evaluation and rollout too.
All three paths check the same ``pd_ee_delta_pose`` controller: root-frame
increments, no target accumulation, and the GPU Levenberg–Marquardt solver
with ``alpha: 1.0``. CPU execution is rejected for this EEF configuration.

The collector starts with the peg on the table and records the complete grasp,
transport and insertion task. All movement phases share ``--tcp-speed-m-s``
(default ``0.04`` m/s) and ``--tcp-angular-speed-rad-s`` (default ``0.25`` rad/s).
It converts joint reference targets to TCP poses and resamples each movement
path at the control frequency, using the larger of translation time and rotation
time for each segment. Dense insertion waypoints therefore do not introduce
extra control steps. At 10 Hz, the defaults allow at most 4 mm of translation
and 0.025 rad (about 1.43 degrees) of rotation per command.

For a new full-task dataset with these defaults, run:

.. code-block:: bash

   PYTHONPATH=. python toolkits/lerobot/collect_maniskill_peg_lerobot_eef.py \
     --repo-id local/peg_fulltask_uniform_tcp \
     --sim-backend gpu --control-freq 10 --fps 10 --sim-freq 100 \
     --tcp-speed-m-s 0.04 --tcp-angular-speed-rad-s 0.25

Gripper closing keeps its explicit hold duration. Phase endpoints are reached
within 1 mm and 0.01 rad before advancing; endpoint tracking uses the same speed
limits and stops at the episode horizon if blocked. Angular limits, path corners,
short final intervals and contact can reduce translation speed. The commanded
limits do not guarantee constant measured speed under GPU IK and contact.
Each episode's ``tcp_speed_by_phase`` records commanded and measured translation
and rotation speed summaries, with endpoint tracking in separate ``*_settle``
phases. The episode metadata and final manifest record ``tcp_motion``;
``--resume`` rejects episodes with missing or different timing settings. Use a
fresh dataset and recompute its normalization statistics before SFT.

Both state and model action have 10 coordinates. The state contains world TCP
xyz, the first two columns of its rotation matrix, and total finger opening in
metres. The action contains normalized root-frame translation increments, the
first two columns of the root-aligned delta rotation, and the gripper command.
Flatten each column separately: ``[r00, r10, r20, r01, r11, r21]``. Do not
subtract rotation columns to construct a delta. The output transform decodes
the rotation to XYZ Euler increments and divides by the installed ManiSkill
controller's ``rot_lower=-0.1``; the environment receives seven coordinates.

Gram–Schmidt canonicalizes raw state and action rotations before SFT
normalization. Before each actor or critic MLP, the residual policy reverses
rotation normalization and canonicalizes the columns again. This covers the
state, the actor's base-action input and the critic's composed-action input.
Equivalent nondegenerate 6D vectors therefore give the networks the same
rotation features. Degenerate vectors use a deterministic orthogonal-axis
fallback. The other coordinates retain their normalized values. Replay and
residual sampling remain in normalized model coordinates; the entropy term is
the density of the sampled 100-dimensional edit before rotation projection,
not a density on SO(3).

Set ``actor.model.eef_rot6d_norm_stats_path`` and
``rollout.residual_base_model.openpi_data.norm_stats_path`` to the same new
``norm_stats.json``. Both training and evaluation use
``wrap_obs_mode: openpi_eef_rot6d`` and ``control_mode: pd_ee_delta_pose``.
Startup validation rejects mixed joint/EEF modes, incompatible dimensions and
mismatched statistics. The model still uses 32 padded pi05 coordinates
internally; only the first 10 coordinates of each of the 10 executed actions
enter residual SAC. With ``edit_scale: 0.02``, the automatic target entropy is
``100 * (log(0.02) - 0.5) = -441.2023``. Critic targets continue to omit entropy
backup.

Privileged Actor/Critic and Geometry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use ``maniskill_sac_residual_openpi_pi05_dinov3_eef_privileged_async`` to give
both actor and critic simulator state in addition to the existing DINOv3 and
EEF inputs. Set ``actor.model.actor_use_privileged_state: true`` as in this
example. Both networks share the 31-to-128 privileged projection; the critic
optimizer trains all shared projections, and actor updates detach their outputs.
The actor also conditions on the frozen VLA's base action. The frozen VLA itself
continues to receive only its original inputs.

The compact state has 31 dimensions, ordered as follows:

.. list-table:: Task-relative privileged observations
   :header-rows: 1
   :widths: 30 10 60

   * - Field
     - Dimensions
     - Representation
   * - ``peg_head_hole``
     - 9
     - Peg-head xyz in millimetres and rot6d, relative to the hole.
   * - ``margins``
     - 3
     - Signed success margins: ``[x + 0.015, radius - abs(y), radius - abs(z)] * 1000``.
   * - ``tcp_peg``
     - 9
     - TCP xyz in millimetres and rot6d, relative to the peg body.
   * - Contact forces
     - 9
     - Left-finger/peg, right-finger/peg, and peg/box force vectors in hole axes.
   * - ``is_grasping``
     - 1
     - Current grasp flag, 0 or 1.

Positions and margins are multiplied by 1000, so a 2 mm alignment error enters
as 2. Nonnegative margins satisfy the corresponding success condition. Random
hole size is reflected in the lateral margins. Peg/hole relative orientation
uses rot6d, as does TCP/peg orientation; both undergo Gram–Schmidt before the
shared projection. Forces use ``sign(f) * log1p(abs(f) / 10 N)``.
The state omits robot-link states, absolute poses, velocities, standalone
geometry, joint targets and success flags. These conversions do not change the
SFT normalization statistics.

Set ``env.train.critic_privileged_obs`` and ``env.eval.critic_privileged_obs``
to ``true``, and ``actor.model.privileged_state_dim`` to ``31``. The example
sets these fields. Replay carries the matching privileged state at both ends
of each chunk, including final observations. The dense recipe treats its
500-step limit as a collection boundary and bootstraps from the final
observation.
Actor rollout and evaluation now require privileged observations too. Missing
state raises an error rather than supplying zeros. Set
``actor.model.actor_use_privileged_state: false`` to retain a critic-only
privileged model whose actor can infer without these observations.

The example retains random geometry in both training and evaluation. Peg full
length is sampled from 170–250 mm and its square side from 30–50 mm. The hole
side is the peg side plus 20 mm, retaining 10 mm clearance on each side. Peg
and box reset poses remain random; the hole is centered in the box and moves
with it. The success rule is unchanged: the peg head must reach hole-frame
x >= -15 mm with y/z inside the hole.

For a later fixed-geometry comparison, set both
``init_params.fixed_peg_half_length: 0.105`` and
``init_params.fixed_peg_half_width: 0.02`` on both splits. These values give a
210 × 40 × 40 mm peg, a 60 × 60 mm hole cross-section and a 210 mm box depth;
the required insertion depth is then 90 mm. Omit both fields to retain random
geometry as in the current example.

Start a fresh residual learner and replay for this configuration; the critic
input and Q-head shapes differ from the non-privileged checkpoint. The old
415-dimensional privileged projection and replay are incompatible with the
31-dimensional layout. Enabling actor
privilege also widens the editor input, so a critic-only checkpoint cannot be
resumed directly with actor privilege enabled. The existing
1000-step EEF SFT base and its matching statistics remain usable. This example
keeps the asynchronous 800,000-update budget, batch size 256, UTD 20,
``edit_scale: 0.02``, ``initial_alpha: 1e-5`` and critic targets without entropy
backup. Changing geometry does not retrain SFT or replace its statistics.

Start With the Peg Held Near the Hole
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To train only alignment and insertion, use the environment settings in
``examples/embodiment/config/env/maniskill_peg_preinsert.yaml``. Reset places
the peg between closed fingers with its tip 50 mm outside the front box face.
The tip has independent uniform Y/Z offsets of ±30 mm in hole coordinates;
its XYZ Euler perturbations are independently uniform within ±3 degrees.
The box retains its original XY and yaw sampling ranges; unreachable cabinet
placements are resampled while keeping the tip offsets fixed. This preset fixes
the peg at 210 × 40 × 40 mm and the centered hole at 60 × 60 mm.

Set ``init_params.hole_clearance: 0.001`` to reduce the per-side clearance
to 1 mm. With the same 40 mm peg width, this produces a 42 × 42 mm hole.
The override updates both the collision geometry and the hole radius used by
the success check. Apply it to both ``env.train.init_params`` and
``env.eval.init_params``. Omitting it preserves the environment variant's
default clearance. Record the source geometry when reusing demonstrations
collected with a different clearance.

The three initialization settings live in ``init_params.preinsert``:

.. code-block:: yaml

   preinsert:
     gap_m: 0.05
     lateral_jitter_m: 0.03
     rotation_jitter_deg: 3.0

Copy the same block into ``env.train.init_params`` and
``env.eval.init_params`` when adapting an RL recipe, and use matching geometry
in collection and training. The gap is axial distance from the front face,
not from the hole center: this preset starts at hole-frame head X = −155 mm.
The existing success boundary remains X ≥ −15 mm with Y/Z inside the hole,
so the tip must travel about 140 mm from reset to success. The preset horizon
is 100 steps at 10 Hz, and it retains the staged dense reward.

Initialization solves the Panda pose with GPU IK and sets the selected
robot and peg states without advancing other environments. The grasp point
is at peg-local X = −0.6 times its half-length. Contact observations remain
zero until the first physical step refreshes the simulator's contact buffer;
subsequent grasp detection uses actual finger contact forces.

Collect five demonstrations and replay their saved actions with:

.. code-block:: bash

   PYTHONPATH=. python toolkits/lerobot/collect_maniskill_peg_preinsert.py \
     --config examples/embodiment/config/env/maniskill_peg_preinsert.yaml \
     --output /path/to/new_preinsert_preview \
     --num-episodes 5 --seed 310000

The controller aligns the peg before advancing it. Each saved trajectory must
succeed both during collection and after resetting to the same seed and
replaying its decoded actions with GPU physics. Both the reset pose and
expert TCP-to-joint conversion use GPU IK. The output
contains a LeRobot ``dataset/``, replay ``videos/``, action/state ``traces/``,
``env.yaml`` and ``manifest.json``. Every accepted episode has a video with
side, main-camera and wrist-camera views. The manifest records attempted
seeds, successes and replay state differences; these are expert checks,
not measurements of an SFT or RL policy.

The default representation is 10-D EEF/rot6d. For joint-space collection, copy
this environment configuration, set ``wrap_obs_mode: rlt_openpi_joint``,
``init_params.control_mode: pd_joint_delta_pos`` and
``critic_privileged_obs: false``, and pass the copied path to ``--config``.
The same collector then saves 9-D joint state and 8-D normalized controller
actions. GPU IK converts the expert's bounded TCP increments into joint targets;
execution and replay use joint PD control. The gripper stays closed.

For either representation, collect fresh demonstrations and recompute π₀.₅
normalization statistics before SFT. Use the resulting model and its matching
statistics as the residual learner's base. EEF and joint checkpoints are not
interchangeable.

Resume and Monitor
~~~~~~~~~~~~~~~~~~

Checkpoints include optimizer states, learned temperature, target model, replay,
update budget and learner RNG state. Resume through ``runner.resume_dir``;
the actor-update cadence and pending UTD budget continue from the saved counts.
The current ``max_updates_per_iteration`` remains in effect on resume, so a
synchronous checkpoint can use shorter asynchronous learner calls. UTD, collection warmup, initial update credits and critic-only warmup
must match the saved schedule. Older checkpoints without the critic-only field
are treated as having zero critic-only updates. The runner step is also restored;
account for the different step meaning when setting the remaining run length.
This restores learner state, not the live simulator state or rollout RNG.

TensorBoard is enabled in these examples. Watch ``env/success_once``,
``train/critic/td_error_abs``, ``train/actor/entropy``, ``train/sac/alpha``,
``train/sac/critic_updates`` and ``train/sac/pending_updates``. Compare runs by
collected transitions and critic updates as well as wall time. CPU checks cover
asynchronous update scheduling and replay ingestion; asynchronous GPU throughput
and ManiSkill learning curves have not yet been validated.

For asynchronous residual SAC, ``env/collected_success_rate`` reports cumulative
successful collection episodes divided by all reported completed collection
episodes, including warm-up. It ranges from 0 to 1 and counts each episode once,
regardless of its length or the worker batch size. ``env/success_once`` continues
to report the current rollout batch. Evaluation episodes, replay sampling and
replay eviction do not change the cumulative count. The metric appears after the
first episode report; in-flight episodes are not yet included.

The runner saves the reported counts in ``collection_metrics.json`` beside the
actor checkpoint and restores them on resume. Resuming an older checkpoint
without this file starts the counter at resume and emits a warning.

Staged Dense Peg Reward
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The privileged EEF recipe uses ``reward_mode: peg_staged_dense`` to minimize
three distance costs: reaching the peg, aligning with the hole, and completing
insertion. Set ``use_rel_reward: false`` and ``ignore_terminations: false`` on
both splits. Success ends the episode with reward zero. The 500-step time limit
truncates collection; ``algorithm.bootstrap_time_limits: true`` continues the
critic target from the final observation before reset. Task success and
``eval/success_once`` keep the original insertion condition.

The reach target is an inclusive TCP region in peg-local axes, with the origin
at the peg center and +X toward the insertion tip. Estimate its bounds from
successful GPU EEF demonstrations:

.. code-block:: bash

   PYTHONPATH=. python toolkits/lerobot/analyze_maniskill_peg_grasps.py \
     --dataset /path/to/maniskill_peginsertionside_10mm_eef_rot6d \
     --output /path/to/grasp_statistics.json \
     --range-margin-m 0.005

The tool replays saved actions and seeds with GPU physics and GPU IK, checks
EEF states against the dataset, and retains successful replays with at least
three consecutive grasp detections. It takes a coordinate median per episode,
then the 10th and 90th percentiles across episodes, expanding both bounds by
the margin. Copy ``grasp_min_peg_m`` and ``grasp_max_peg_m`` into
``env.train.peg_dense_reward`` and ``env.eval.peg_dense_reward``. The example's
400-demonstration region is approximately X=[-75.26, -49.72], Y=[-5.15, 5.04],
Z=[-6.08, 6.00] mm. This region defines reach distance only: actual
``is_grasping`` contact unlocks alignment even when TCP lies outside the region.

All distances are in metres. The three costs are:

* Reach distance is the L2 distance from TCP to the nearest point in the region,
  zero inside. Its scale ``reach_scale_m`` is 0.05.
* Alignment distance combines the peg tip's YZ displacement from the hole axis
  with axis-direction error. Specifically, take the L2 norm of the concatenated
  vector ``[head_y, head_z, L * (peg_axis - hole_axis)]`` in hole coordinates,
  where L is the current peg half-length and the hole axis is [1, 0, 0]. This
  expresses both terms in metres and penalizes a reversed insertion axis.
  ``alignment_scale_m`` is 0.05.
* Remaining insertion distance is ``max(-0.015 - head_x, 0)`` in hole axes.
  ``depth_scale_m`` is 0.10. Depth unlocks only while grasped, lateral distance
  is at most ``alignment_tolerance_m`` (2 mm), and axis angle is at most
  ``axis_tolerance_deg`` (5 degrees).

Normalize each unlocked distance as ``c = d / (d + scale)``, in [0, 1).
A locked alignment or depth term costs exactly one. Thus:

.. math::

   r = -\left(c_{\mathrm{reach}}
       + \begin{cases}c_{\mathrm{align}} & \text{grasped}\\1 & \text{otherwise}\end{cases}
       + \begin{cases}c_{\mathrm{depth}} & \text{aligned and grasped}\\1 & \text{otherwise}\end{cases}\right).

Success overrides all three costs to zero. Rewards stay in [-3, 0]. At fixed
geometry, unlocking a term replaces a cost of one with a value below one, so
unlocking alone cannot reduce reward. The gates use current contact and pose;
dropping the peg or losing alignment relocks the corresponding costs. Larger
distances increase costs smoothly without saturating at a hard cutoff.

Each control step pays its state costs. Residual SAC sums the ten control-step
rewards in a chunk and applies gamma=0.99 once in the critic target. Steps after
episode end are frozen with reward zero. A stationary unsuccessful state keeps
paying a negative cost, and successful termination stops future costs. This is
a distance-cost objective, not a potential difference: episode return is the
sum of the costs paid along the trajectory and depends on its duration and path.
There is no potential baseline or reward-side discount to reset or maintain.

Episode metrics expose ``dense_reward``, ``dense_grasp_cost``,
``dense_alignment_cost`` and ``dense_depth_cost`` for checking the decomposition.
``dense_stage`` is 0=reach, 1=align, 2=insert, 3=success. ``dense_grasped`` and
``dense_aligned`` show the gates. Distance diagnostics are
``grasp_region_error_mm``, ``alignment_error_mm`` (lateral only),
``alignment_distance_mm`` (combined L2), ``alignment_axis_error_deg``,
``depth_remaining_mm`` and ``insertion_depth_mm``. These are state snapshots;
``episode/return`` accumulates rewards.

Keep ``reward.use_reward_model: false`` to pass the negative rewards unchanged.
Use a fresh residual learner and replay when changing reward definitions, while
retaining the SFT checkpoint and its normalization statistics. Check actual SR
within 500 steps and completion time as well as return; continuing bootstrap
optimizes a different horizon from the evaluation time limit.

EXPO Candidate Selection
~~~~~~~~~~~~~~~~~~~~~~~~

Use EXPO when the frozen base policy can propose several different action chunks
and you want the critic to choose among them. The implementation follows the
local gr00t ``algorithms/expo`` package: it retains residual SAC's replay,
asynchronous collection, optimizers and bounded editor, and changes action
selection and the actor objective.

At each observation, the frozen base samples eight independent chunks. EXPO
samples one edit of each chunk and scores all sixteen actions with the minimum
of a shared random pair of target critics. It executes the action with the
highest score. The editor uses the target state/visual projection during this
selection. Image/prompt prefix encoding and DINOv3 features are reused across
base proposals; the eight denoising passes are sequential, so collection costs
more than single-proposal residual SAC.
The edit bound applies relative to each proposal. Selecting a different base
proposal can change the action by more than one bounded residual edit.

Replay stores the executed normalized action and actual next-observation
candidates with shape ``[num_candidates, chunk_dim]``. A timeout uses candidates
from its final observation, before reset. The critic repeats selection on these
cached candidates and draws a fresh critic pair for its Bellman backup:

.. math::

   y = r + \gamma (1-d)\min_{j\in J'}Q^{\mathrm{target}}_j(s',a^*).

Success terminates bootstrap; time limits continue bootstrap by default.
``algorithm.backup_entropy`` must be false for EXPO. The actor learns an edit
around the executed replay action, uses detached online features, and optimizes
the mean of all online critics plus the usual SAC entropy term. Alpha remains
learned, with the same edit-scale Jacobian and target-entropy calculation as
residual SAC.

The asynchronous configuration is
``maniskill_expo_openpi_pi05_dinov3_eef_async.yaml``. It uses a frozen SFT
2000-step base, DINOv3 plus EEF proprioception, canonical rot6d inputs, and no
privileged observations. The task starts with a grasped peg 5 cm in front of the
hole, lateral jitter of 3 cm and rotation jitter of 3 degrees. The peg dimensions
are fixed at 210 x 40 x 40 mm and the hole has 1 mm clearance per side. Rewards
are sparse 0/1. The main settings are:

.. list-table:: EXPO Training Defaults
   :header-rows: 1

   * - Setting
     - Value
   * - ``actor.model.num_candidates`` / ``actor.model.num_edits``
     - 8 / 8, with one edit of each of the first eight proposals
   * - ``actor.model.edit_scale`` / initial alpha
     - 0.02 / 1e-5
   * - Batch size / UTD / actor interval
     - 256 / 20 / every 20 critic updates after critic warmup
   * - Warmup
     - 4000 chunk transitions, then 20000 critic-only updates
   * - Critic budget
     - 800000 updates including warmup; 39000 actor updates
   * - Checkpoint and evaluation
     - Every 100000 critic updates; 50 evaluation episodes with videos

Set the base checkpoint, both normalization paths, and the DINOv3 repository and
weights paths before launching. The two normalization paths must refer to the
same SFT statistics. Run from the repository root in the existing OpenPI and
ManiSkill environment:

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

This launches one learner on GPU 0 and frozen-base inference plus simulation on
GPU 1. ``runner.val_critic_interval`` and ``runner.save_critic_interval`` both use
completed critic updates, including warmup. A learner call stops at each
milestone and evaluation waits for the matching weights to reach rollout.
Target projections and target critics belong to the EXPO model state and travel
with weight synchronization and checkpoints; they are excluded from optimizers.
Resume with an EXPO checkpoint through ``runner.resume_dir`` to restore replay,
optimizer state, target weights, update credits and RNG state together.

EXPO samples edits and critic subsets in evaluation too. For paired comparisons,
fix both scene seeds and policy sampling seeds in a standalone evaluator; a
scene seed alone does not fix candidate selection. The built-in 50-episode
evaluation advances its environment seed stream between checkpoints and saves
one tiled video per parallel batch. TensorBoard
retains the shared ``train/sac/critic_updates`` namespace,
``train/actor_q_to_entropy_grad_ratio`` and cumulative
``env/collected_success_rate``. ``train/critic/target_edit_fraction`` measures the
fraction of Bellman targets selecting edited proposals. None of these training
diagnostics replaces independently measured evaluation SR.


Choose the Action Representation and Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the following asynchronous configurations to select the algorithm, action
representation and frozen feature source. Each name is passed to
``examples/embodiment/train_async.py`` through ``--config-name``.

.. list-table:: Residual Learning Configurations
   :header-rows: 1

   * - Algorithm
     - State/action representation
     - Features
     - Config name
   * - Residual SAC
     - Joint
     - π₀.₅ prefix mean
     - ``maniskill_sac_residual_openpi_pi05_async``
   * - Residual SAC
     - Joint
     - DINOv3
     - ``maniskill_sac_residual_openpi_pi05_dinov3_async``
   * - Residual SAC
     - EEF rot6d
     - π₀.₅ prefix mean
     - ``maniskill_sac_residual_openpi_pi05_eef_async``
   * - Residual SAC
     - EEF rot6d
     - DINOv3
     - ``maniskill_sac_residual_openpi_pi05_dinov3_eef_async``
   * - EXPO
     - Joint
     - π₀.₅ prefix mean
     - ``maniskill_expo_openpi_pi05_async``
   * - EXPO
     - Joint
     - DINOv3
     - ``maniskill_expo_openpi_pi05_dinov3_async``
   * - EXPO
     - EEF rot6d
     - π₀.₅ prefix mean
     - ``maniskill_expo_openpi_pi05_eef_async``
   * - EXPO
     - EEF rot6d
     - DINOv3
     - ``maniskill_expo_openpi_pi05_dinov3_eef_async``

Joint configurations use 9-D proprioception and 8-D actions; EEF configurations
use 10-D state and actions with canonical rot6d rotations. Always supply a base
SFT checkpoint and ``norm_stats.json`` trained for the selected representation.
EEF also requires ``actor.model.eef_rot6d_norm_stats_path`` to point to those
same statistics. The existing residual SAC examples default to the full 10 mm
task; the EXPO examples default to the 1 mm held-peg preinsert task. Match task
geometry, episode horizon, reset settings and warmup when comparing algorithms.

Without ``rollout.residual_dinov3``, or when it is null, actor and critic receive
the 2048-D masked mean of all valid π₀.₅ prefix outputs, including image, prompt
and discrete state tokens. Padding tokens are excluded. This is one global
mean, not camera-wise spatial pooling. With DINOv3 enabled, each camera retains
a 2 × 2 grid before concatenation into 3072 features. Both backbones remain
frozen; the critic optimizer trains the shared state and visual projections.
Start a fresh residual learner and replay when switching the representation
or feature source, since input dimensions and stored features change.

The bounded edit is measured in normalized SFT action coordinates. Therefore
``edit_scale: 0.02`` does not imply equal physical authority in joint and EEF
space. With ten-action chunks and automatic entropy targeting, the joint
configuration uses ``80 * (log(0.02) - 0.5) = -352.9618`` and EEF uses
``100 * (log(0.02) - 0.5) = -441.2023``. Neither configuration adds entropy to
the critic backup.
