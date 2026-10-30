"""GPU 驻留 PPO 学习器 (CleanRL 风格自研, SB3 语义逐项对齐)

替换 train_mjx.py 中的 SB3 学习器, 目的: rollout 数据每迭代只搬运一次,
n_epochs×minibatch 梯度步全部在 GPU 完成, 消除 CPU 侧瓶颈。

与 SB3 PPO (v15 现役超参) 对齐清单:
  - 网络: 分离 pi/vf 主干 (Tanh), action_net, 独立 log_std 参数,
    ortho_init (hidden gain=sqrt2, action/value gain=0.01)
  - GAE(lambda=0.95): 所有 done (含 max_steps 截断) 按终止处理不自举
    —— 现役 MJXAvoidVecEnv 不提供 TimeLimit.truncated, SB3 实际行为即如此,
    terminal_observation 不参与 GAE (保持行为零漂移)
  - advantages 逐 minibatch 标准化 (eps 1e-8)
  - loss = -min(ratio*A, clip(ratio)*A) + ent_coef*(-mean(entropy)) + 0.5*MSE
  - Adam(eps=1e-8), grad clip norm 0.5, 无价值裁剪
  - buffer 存未裁剪动作; 喂环境时 clip 到 [-1,1] (与 SB3 collect 一致)
  - 学习率线性退火, 每迭代 (每次 train) 更新一次
  - 奖励归一化: RewardNormalizer 逐字段复刻
    VecNormalize(norm_obs=False, norm_reward=True, clip_reward, gamma)
"""
import numpy as np
import torch
import torch.nn as nn


class RewardNormalizer:
    """复刻 VecNormalize 奖励归一化 (norm_obs=False, norm_reward=True)

    step_wait 语义: returns = returns*gamma + raw_reward;
    ret_rms.update(returns); normed = clip(raw/sqrt(var+1e-8), ±clip);
    returns[dones] = 0。统计量用 float64 (与 SB3 RunningMeanStd 一致)。
    """

    def __init__(self, n_envs, clip_reward=20.0, gamma=0.99, epsilon=1e-8):
        self.n_envs = n_envs
        self.clip_reward = float(clip_reward)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.mean = np.float64(0.0)
        self.var = np.float64(1.0)
        self.count = np.float64(1e-4)      # RunningMeanStd epsilon
        self.returns = np.zeros(n_envs, dtype=np.float64)

    def step(self, rewards, dones):
        """raw rewards (n_envs,) numpy -> normalized rewards (float32)"""
        rewards = np.asarray(rewards, dtype=np.float64)
        self.returns = self.returns * self.gamma + rewards
        # RunningMeanStd.update (平行算法, float64)
        batch_mean = self.returns.mean()
        batch_var = self.returns.var()
        batch_count = self.returns.shape[0]
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta * delta * self.count * batch_count / tot_count
        self.mean = new_mean
        self.var = m_2 / tot_count
        self.count = tot_count
        normed = np.clip(rewards / np.sqrt(self.var + self.epsilon),
                         -self.clip_reward, self.clip_reward)
        self.returns[dones] = 0.0
        return normed.astype(np.float32)

    def state_dict(self):
        return {'mean': float(self.mean), 'var': float(self.var),
                'count': float(self.count), 'returns': self.returns.copy()}

    def load_state_dict(self, st):
        self.mean = np.float64(st['mean'])
        self.var = np.float64(st['var'])
        self.count = np.float64(st['count'])
        self.returns = np.asarray(st['returns'], dtype=np.float64)


def apply_collision_backtrace(rewards, dones, collisions, steps, penalty,
                              decay):
    """Apply causal pre-collision penalties without crossing episode bounds.

    Arrays use (time, env).  The collision transition already receives the
    environment's terminal penalty, so shaping starts from the preceding
    transition.  This only edits trajectories collected by the current policy.
    """
    steps = int(steps)
    penalty = float(penalty)
    decay = float(decay)
    if steps <= 0 or penalty <= 0:
        return 0, 0.0
    events = np.argwhere(np.asarray(collisions, dtype=bool))
    total = 0.0
    for t, env_index in events:
        for offset in range(1, steps + 1):
            prior = int(t) - offset
            if prior < 0 or dones[prior, env_index]:
                break
            amount = penalty * decay ** (offset - 1)
            rewards[prior, env_index] -= amount
            total += amount
    return int(len(events)), float(total)


def _layer_init(layer, gain=np.sqrt(2.0)):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class ActorCriticMLP(nn.Module):
    """与 SB3 ActorCriticPolicy 同构: 分离 pi/vf 主干 + action_net + log_std

    层索引与 SB3 MlpExtractor 的 Sequential(Linear,Tanh,...) 一致
    (policy_net/value_net 的 Linear 位于偶数索引), 保证权重可直接导出。
    """

    def __init__(self, obs_dim, act_dim, arch=(512, 512, 256, 128),
                 ortho_init=True, constrained=False, step_risk_head=False,
                 step_risk_hidden=16, step_action_index=-1,
                 direction_risk_head=False, direction_risk_hidden=32,
                 direction_risk_max_residual=0.15,
                 direction_risk_margin_norm=0.20):
        super().__init__()
        pi, vf = [], []
        in_p, in_v = obs_dim, obs_dim
        for h in arch:
            pi += [_layer_init(nn.Linear(in_p, h)), nn.Tanh()]
            vf += [_layer_init(nn.Linear(in_v, h)), nn.Tanh()]
            in_p, in_v = h, h
        if not ortho_init:
            for m in pi + vf:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0.0)
        self.policy_net = nn.Sequential(*pi)
        self.value_net = nn.Sequential(*vf)
        self.action_net = _layer_init(nn.Linear(in_p, act_dim), gain=0.01)
        self.value_head = _layer_init(nn.Linear(in_v, 1), gain=0.01)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.step_risk_head_enabled = bool(step_risk_head)
        self.step_action_index = int(step_action_index) % act_dim
        if self.step_risk_head_enabled:
            self.step_risk_head = nn.Sequential(
                _layer_init(nn.Linear(2, int(step_risk_hidden))), nn.Tanh(),
                nn.Linear(int(step_risk_hidden), 1))
            nn.init.zeros_(self.step_risk_head[-1].weight)
            nn.init.zeros_(self.step_risk_head[-1].bias)
        self.direction_risk_head_enabled = bool(direction_risk_head)
        self.direction_risk_max_residual = float(direction_risk_max_residual)
        self.direction_risk_margin_norm = float(direction_risk_margin_norm)
        if self.direction_risk_head_enabled:
            self.direction_risk_head = nn.Sequential(
                _layer_init(nn.Linear(obs_dim, int(direction_risk_hidden))),
                nn.Tanh(), nn.Linear(int(direction_risk_hidden), act_dim - 1))
            nn.init.zeros_(self.direction_risk_head[-1].weight)
            nn.init.zeros_(self.direction_risk_head[-1].bias)
        self.constrained = bool(constrained)
        if self.constrained:
            cost, in_c = [], obs_dim
            for h in arch:
                cost += [_layer_init(nn.Linear(in_c, h)), nn.Tanh()]
                in_c = h
            self.cost_net = nn.Sequential(*cost)
            self.cost_head = nn.Linear(in_c, 1)
            # A neutral cost prediction preserves the migrated actor/value
            # behavior exactly at the beginning of v25 fine-tuning.
            nn.init.zeros_(self.cost_head.weight)
            nn.init.zeros_(self.cost_head.bias)

    def forward(self, obs):
        """obs (N, obs_dim) -> (action_mean, value)"""
        feat_pi = self.policy_net(obs)
        feat_vf = self.value_net(obs)
        mean = self.action_net(feat_pi)
        if self.step_risk_head_enabled:
            residual = self.step_risk_head(obs[..., -2:]).squeeze(-1)
            mean = mean.clone()
            mean[..., self.step_action_index] += residual
        if self.direction_risk_head_enabled:
            # obs[-1] is predicted clearance / dist_clip.  The smooth gate is
            # zero outside the risk margin, so free-space behavior is exactly
            # inherited from the champion.
            gate = torch.clamp(
                (self.direction_risk_margin_norm - obs[..., -1])
                / max(self.direction_risk_margin_norm, 1e-6), 0.0, 1.0)
            residual = (torch.tanh(self.direction_risk_head(obs))
                        * self.direction_risk_max_residual * gate[..., None])
            mean = mean.clone()
            mean[..., :self.step_action_index] += residual
        return mean, self.value_head(feat_vf).squeeze(-1)

    def evaluate_actions(self, obs, actions):
        """返回 (values, log_prob, entropy); log_prob/entropy 对动作维求和"""
        mean, values = self.forward(obs)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return values, log_prob, entropy

    def cost_value(self, obs):
        if not self.constrained:
            raise RuntimeError('cost_value requires constrained PPO')
        return self.cost_head(self.cost_net(obs)).squeeze(-1)

    def act(self, obs):
        """采样动作 (训练 rollout 用), 返回 (actions, log_prob, value)"""
        mean, values = self.forward(obs)
        std = torch.exp(self.log_std)
        dist = torch.distributions.Normal(mean, std)
        actions = dist.sample()
        return actions, dist.log_prob(actions).sum(-1), values


class GpuPPO:
    """GPU 驻留 PPO 学习器 (rollout buffer 预分配于 device)"""

    def __init__(self, obs_dim, act_dim, arch=(512, 512, 256, 128),
                 device='cuda', n_steps=64, n_envs=4096, batch_size=1024,
                 n_epochs=5, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                 ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
                 lr_init=2e-4, lr_final=1e-5, seed=0,
                 constrained=False, cost_gamma=0.99,
                 cost_gae_lambda=0.95, cost_vf_coef=0.5,
                 cost_limit_start=0.08, cost_limit_final=0.03,
                 lagrange_init=0.0, lagrange_lr=5.0,
                 lagrange_max=5.0, collision_backtrace_steps=0,
                 collision_backtrace_penalty=0.0,
                 collision_backtrace_decay=0.8,
                 actor_update_mode='full', step_action_index=-1,
                 step_risk_head=False, step_risk_hidden=16,
                 direction_risk_head=False, direction_risk_hidden=32,
                 direction_risk_max_residual=0.15,
                 direction_risk_margin_norm=0.20):
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.device = torch.device(device)
        self.n_steps, self.n_envs = n_steps, n_envs
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma, self.gae_lambda = gamma, gae_lambda
        self.clip_range, self.ent_coef, self.vf_coef = clip_range, ent_coef, vf_coef
        self.max_grad_norm = max_grad_norm
        self.lr_init, self.lr_final = lr_init, lr_final
        self.constrained = bool(constrained)
        self.cost_gamma = float(cost_gamma)
        self.cost_gae_lambda = float(cost_gae_lambda)
        self.cost_vf_coef = float(cost_vf_coef)
        self.cost_limit_start = float(cost_limit_start)
        self.cost_limit_final = float(cost_limit_final)
        self.lagrange_multiplier = float(lagrange_init)
        self.lagrange_lr = float(lagrange_lr)
        self.lagrange_max = float(lagrange_max)
        self._rollout_collision_rate = 0.0
        self._rollout_obstacle_episodes = 0
        self.collision_backtrace_steps = int(collision_backtrace_steps)
        self.collision_backtrace_penalty = float(collision_backtrace_penalty)
        self.collision_backtrace_decay = float(collision_backtrace_decay)
        self.actor_update_mode = str(actor_update_mode).strip().lower()
        self.step_action_index = int(step_action_index)

        torch.manual_seed(seed)
        np.random.seed(seed)
        self.policy = ActorCriticMLP(
            obs_dim, act_dim, arch, constrained=self.constrained,
            step_risk_head=step_risk_head,
            step_risk_hidden=step_risk_hidden,
            step_action_index=step_action_index,
            direction_risk_head=direction_risk_head,
            direction_risk_hidden=direction_risk_hidden,
            direction_risk_max_residual=direction_risk_max_residual,
            direction_risk_margin_norm=direction_risk_margin_norm).to(self.device)
        self._gradient_mask_handles = []
        self._configure_actor_updates()
        # SB3 OnPolicyAlgorithm 默认 optimizer_kwargs=dict(eps=1e-8)
        self.optimizer = torch.optim.Adam(
            (p for p in self.policy.parameters() if p.requires_grad),
                                          lr=lr_init, eps=1e-8)

        # 预分配 GPU buffer (每迭代复用)
        n = n_steps * n_envs
        self.b_obs = torch.zeros((n, obs_dim), dtype=torch.float32,
                                 device=self.device)
        self.b_act = torch.zeros((n, act_dim), dtype=torch.float32,
                                 device=self.device)
        self.b_logp = torch.zeros(n, dtype=torch.float32, device=self.device)
        self.b_val = torch.zeros(n, dtype=torch.float32, device=self.device)
        self.b_rew = torch.zeros(n, dtype=torch.float32, device=self.device)
        self.b_ep_start = torch.zeros(n, dtype=torch.float32,
                                      device=self.device)
        self.b_adv = torch.zeros(n, dtype=torch.float32, device=self.device)
        self.b_ret = torch.zeros(n, dtype=torch.float32, device=self.device)
        if self.constrained:
            self.b_cost = torch.zeros(n, dtype=torch.float32,
                                      device=self.device)
            self.b_cost_val = torch.zeros(n, dtype=torch.float32,
                                          device=self.device)
            self.b_cost_adv = torch.zeros(n, dtype=torch.float32,
                                          device=self.device)
            self.b_cost_ret = torch.zeros(n, dtype=torch.float32,
                                          device=self.device)
        self._rng = np.random.default_rng(seed)

    def _configure_actor_updates(self):
        """Configure which actor parameters PPO may change.

        v28 uses ``step_only``: the migrated policy backbone is immutable and
        gradient masks expose only one row of the action head and one log-std
        element.  The reward value network (and an optional cost critic) stays
        fully trainable.  Keeping the original action-head tensor shapes makes
        exported zips bit-for-bit compatible with every inference backend.
        """
        if self.actor_update_mode == 'full':
            return
        if self.actor_update_mode not in ('step_only', 'risk_step_head',
                                          'risk_direction_head'):
            raise ValueError(
                f'未知 actor_update_mode={self.actor_update_mode!r}; '
                "仅支持 'full'、'step_only'、'risk_step_head' 或 "
                "'risk_direction_head'")
        index = self.step_action_index
        if index < 0:
            index += self.act_dim
        if not 0 <= index < self.act_dim:
            raise ValueError(
                f'step_action_index={self.step_action_index} 超出 '
                f'{self.act_dim} 维动作范围')
        self.step_action_index = index

        for param in self.policy.policy_net.parameters():
            param.requires_grad_(False)

        if self.actor_update_mode == 'risk_step_head':
            if not self.policy.step_risk_head_enabled:
                raise ValueError('risk_step_head 模式要求 step_risk_head=true')
            for param in self.policy.action_net.parameters():
                param.requires_grad_(False)
            std_mask = torch.zeros_like(self.policy.log_std)
            std_mask[index] = 1.0
            self._gradient_mask_handles.append(
                self.policy.log_std.register_hook(
                    lambda grad, mask=std_mask: grad * mask))
            return

        if self.actor_update_mode == 'risk_direction_head':
            if not self.policy.direction_risk_head_enabled:
                raise ValueError('risk_direction_head 模式要求 direction_risk_head=true')
            for param in self.policy.action_net.parameters():
                param.requires_grad_(False)
            if self.policy.step_risk_head_enabled:
                for param in self.policy.step_risk_head.parameters():
                    param.requires_grad_(False)
            std_mask = torch.zeros_like(self.policy.log_std)
            std_mask[:index] = 1.0
            self._gradient_mask_handles.append(
                self.policy.log_std.register_hook(
                    lambda grad, mask=std_mask: grad * mask))
            return

        weight_mask = torch.zeros_like(self.policy.action_net.weight)
        weight_mask[index].fill_(1.0)
        bias_mask = torch.zeros_like(self.policy.action_net.bias)
        bias_mask[index] = 1.0
        std_mask = torch.zeros_like(self.policy.log_std)
        std_mask[index] = 1.0
        self._gradient_mask_handles.extend([
            self.policy.action_net.weight.register_hook(
                lambda grad, mask=weight_mask: grad * mask),
            self.policy.action_net.bias.register_hook(
                lambda grad, mask=bias_mask: grad * mask),
            self.policy.log_std.register_hook(
                lambda grad, mask=std_mask: grad * mask),
        ])

    def trainable_parameter_summary(self):
        """Small diagnostic used by the trainer and v28 regression tests."""
        trainable = sum(p.numel() for p in self.policy.parameters()
                        if p.requires_grad)
        total = sum(p.numel() for p in self.policy.parameters())
        return {
            'actor_update_mode': self.actor_update_mode,
            'step_action_index': self.step_action_index,
            'trainable_parameters': int(trainable),
            'total_parameters': int(total),
        }

    # ---------- 推理接口 (rollout_eval / eval_stats 兼容 SB3 .predict) ----------

    @torch.no_grad()
    def predict(self, observation, deterministic=True):
        """numpy in/out, 与 SB3 model.predict(obs, deterministic=True) 同形

        含 SB3 BasePolicy.predict 的语义: Box 动作空间输出 clip 到 [-1,1]
        (非 squash 高斯策略的防越界裁剪, 现役推理链依赖此行为)。
        """
        obs = np.array(observation, dtype=np.float32)
        single = obs.ndim == 1
        if single:
            obs = obs[None, :]
        obs_t = torch.from_numpy(obs).to(self.device)
        mean, _ = self.policy(obs_t)
        if deterministic:
            actions = mean
        else:
            std = torch.exp(self.policy.log_std)
            actions = torch.distributions.Normal(mean, std).sample()
        actions = np.clip(actions.cpu().numpy(), -1.0, 1.0)
        return (actions[0], None) if single else (actions, None)

    # ---------- rollout 收集 ----------

    @torch.no_grad()
    def collect(self, env, obs, normalizer, ep_starts):
        """收集 n_steps 步; obs/ep_starts 为 numpy (n_envs,), 由调用方维护

        返回 (last_obs, last_ep_starts, metrics)。奖励经 normalizer 归一化。
        与 SB3 一致: buffer 存未裁剪动作, 喂环境前 clip 到动作空间。
        """
        ns, ne = self.n_steps, self.n_envs
        raw_rew_all = np.zeros((ns, ne), dtype=np.float32)
        use_backtrace = (self.collision_backtrace_steps > 0
                         and self.collision_backtrace_penalty > 0)
        done_all = (np.zeros((ns, ne), dtype=bool)
                    if use_backtrace else None)
        collision_all = (np.zeros((ns, ne), dtype=bool)
                         if use_backtrace else None)
        collision_count = 0
        obstacle_episode_count = 0
        safety_cost_sum = 0.0
        failure_resets_before = int(getattr(
            getattr(env, 'mjx', None), 'failure_reset_total', 0))
        for t in range(ns):
            obs_t = torch.from_numpy(np.array(obs, dtype=np.float32))\
                .to(self.device, non_blocking=True)
            actions, logp, values = self.policy.act(obs_t)
            a_np = actions.cpu().numpy()
            sl = slice(t * ne, (t + 1) * ne)
            self.b_obs[sl] = obs_t
            self.b_act[sl] = actions
            self.b_logp[sl] = logp
            self.b_val[sl] = values
            if self.constrained:
                self.b_cost_val[sl] = self.policy.cost_value(obs_t)
            self.b_ep_start[sl] = torch.from_numpy(
                ep_starts.astype(np.float32)).to(self.device)

            clipped = np.clip(a_np, -1.0, 1.0)
            next_obs, rewards, dones, _ = env.step(clipped)
            if use_backtrace:
                done_all[t] = dones
                collision_all[t] = np.asarray(
                    env.last_collision_cost, dtype=np.float32) > 0
            if self.constrained:
                costs = np.array(env.last_safety_cost, dtype=np.float32,
                                 copy=True)
                self.b_cost[sl] = torch.from_numpy(costs).to(self.device)
                collision_count += int(np.asarray(
                    env.last_collision_cost, dtype=np.float32).sum())
                safety_cost_sum += float(costs.sum())
                obstacle_episode_count += int(np.count_nonzero(
                    dones & np.asarray(env.last_has_obstacle, dtype=bool)))
            if not use_backtrace:
                normed = normalizer.step(rewards, dones)
                self.b_rew[sl] = torch.from_numpy(normed).to(self.device)
            raw_rew_all[t] = rewards
            obs, ep_starts = next_obs, dones

        raw_ep_rew_mean = float(raw_rew_all.sum(0).mean())
        backtrace_events = 0
        backtrace_penalty_sum = 0.0
        if use_backtrace:
            backtrace_events, backtrace_penalty_sum = \
                apply_collision_backtrace(
                    raw_rew_all, done_all, collision_all,
                    self.collision_backtrace_steps,
                    self.collision_backtrace_penalty,
                    self.collision_backtrace_decay)
            for t in range(ns):
                sl = slice(t * ne, (t + 1) * ne)
                normed = normalizer.step(raw_rew_all[t], done_all[t])
                self.b_rew[sl] = torch.from_numpy(normed).to(self.device)

        # 末帧自举值 (SB3: last_values; done 世界乘 0 由 GAE 处理)
        last_obs_t = torch.from_numpy(
            np.array(obs, dtype=np.float32)).to(self.device)
        _, last_values = self.policy(last_obs_t)
        self._compute_gae(last_values,
                          torch.from_numpy(ep_starts.astype(np.float32))
                          .to(self.device))
        if self.constrained:
            last_cost_values = self.policy.cost_value(last_obs_t)
            self._compute_cost_gae(
                last_cost_values,
                torch.from_numpy(ep_starts.astype(np.float32)).to(self.device))
            self._rollout_obstacle_episodes = obstacle_episode_count
            self._rollout_collision_rate = (
                collision_count / max(obstacle_episode_count, 1))
        failure_resets_after = int(getattr(
            getattr(env, 'mjx', None), 'failure_reset_total', 0))
        metrics = {
            'ep_rew_mean': raw_ep_rew_mean,
            'failure_resets': failure_resets_after - failure_resets_before,
            'backtrace_collision_events': backtrace_events,
            'backtrace_penalty_sum': backtrace_penalty_sum,
        }
        if self.constrained:
            metrics.update({
                'collision_count': collision_count,
                'obstacle_episodes': obstacle_episode_count,
                'obstacle_collision_rate': self._rollout_collision_rate,
                'safety_cost_mean': safety_cost_sum / (ns * ne),
                'dense_safety_cost_sum': safety_cost_sum - collision_count,
            })
        return obs, ep_starts, metrics

    def _compute_gae(self, last_values, last_ep_starts):
        """对齐 SB3 RolloutBuffer.compute_returns_and_advantage:
        所有 done 均按终止处理 (next_non_terminal=0), 无截断自举。"""
        ns, ne = self.n_steps, self.n_envs
        rew = self.b_rew.view(ns, ne)
        val = self.b_val.view(ns, ne)
        eps = self.b_ep_start.view(ns, ne)
        adv = self.b_adv.view(ns, ne)
        last_gae = torch.zeros(ne, dtype=torch.float32, device=self.device)
        for t in reversed(range(ns)):
            if t == ns - 1:
                next_non_terminal = 1.0 - last_ep_starts
                next_values = last_values
            else:
                next_non_terminal = 1.0 - eps[t + 1]
                next_values = val[t + 1]
            delta = (rew[t] + self.gamma * next_values * next_non_terminal
                     - val[t])
            last_gae = (delta + self.gamma * self.gae_lambda
                        * next_non_terminal * last_gae)
            adv[t] = last_gae
        self.b_ret.copy_(self.b_adv + self.b_val)

    def _compute_cost_gae(self, last_values, last_ep_starts):
        """GAE for the collision-event cost stream."""
        ns, ne = self.n_steps, self.n_envs
        cost = self.b_cost.view(ns, ne)
        val = self.b_cost_val.view(ns, ne)
        eps = self.b_ep_start.view(ns, ne)
        adv = self.b_cost_adv.view(ns, ne)
        last_gae = torch.zeros(ne, dtype=torch.float32, device=self.device)
        for t in reversed(range(ns)):
            if t == ns - 1:
                next_non_terminal = 1.0 - last_ep_starts
                next_values = last_values
            else:
                next_non_terminal = 1.0 - eps[t + 1]
                next_values = val[t + 1]
            delta = (cost[t] + self.cost_gamma * next_values
                     * next_non_terminal - val[t])
            last_gae = (delta + self.cost_gamma * self.cost_gae_lambda
                        * next_non_terminal * last_gae)
            adv[t] = last_gae
        self.b_cost_ret.copy_(self.b_cost_adv + self.b_cost_val)

    # ---------- PPO 更新 (全程 GPU) ----------

    def update(self, progress_remaining):
        """progress_remaining = 1 - 已完成步/总步 (SB3 语义); 返回 loss 指标"""
        n = self.n_steps * self.n_envs
        # 学习率线性退火, 每迭代更新一次 (对齐 linear_schedule)
        lr = self.lr_final + (self.lr_init - self.lr_final) * progress_remaining
        for g in self.optimizer.param_groups:
            g['lr'] = lr

        cost_limit = (self.cost_limit_final
                      + (self.cost_limit_start - self.cost_limit_final)
                      * progress_remaining)
        if self.constrained and self._rollout_obstacle_episodes > 0:
            violation = self._rollout_collision_rate - cost_limit
            self.lagrange_multiplier = float(np.clip(
                self.lagrange_multiplier + self.lagrange_lr * violation,
                0.0, self.lagrange_max))

        bsz = self.batch_size
        assert n % bsz == 0, f'batch_size {bsz} 需整除 rollout {n}'
        pg_losses, vf_losses, cost_vf_losses = [], [], []
        ent_losses, kls, clip_fracs = [], [], []
        for _ in range(self.n_epochs):
            idx = torch.from_numpy(self._rng.permutation(n))\
                .to(self.device)
            for start in range(0, n, bsz):
                mb = idx[start:start + bsz]
                values, logp, entropy = self.policy.evaluate_actions(
                    self.b_obs[mb], self.b_act[mb])
                adv = self.b_adv[mb]
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                cost_vf_loss = torch.zeros((), device=self.device)
                if self.constrained:
                    cost_adv = self.b_cost_adv[mb]
                    cost_adv = ((cost_adv - cost_adv.mean())
                                / (cost_adv.std() + 1e-8))
                    adv = ((adv - self.lagrange_multiplier * cost_adv)
                           / (1.0 + self.lagrange_multiplier))
                ratio = torch.exp(logp - self.b_logp[mb])
                l1 = adv * ratio
                l2 = adv * torch.clamp(ratio, 1 - self.clip_range,
                                       1 + self.clip_range)
                pg_loss = -torch.min(l1, l2).mean()
                vf_loss = nn.functional.mse_loss(self.b_ret[mb], values)
                if self.constrained:
                    cost_values = self.policy.cost_value(self.b_obs[mb])
                    cost_vf_loss = nn.functional.mse_loss(
                        self.b_cost_ret[mb], cost_values)
                ent_loss = -entropy.mean()
                loss = (pg_loss + self.ent_coef * ent_loss
                        + self.vf_coef * vf_loss
                        + self.cost_vf_coef * cost_vf_loss)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()
                with torch.no_grad():
                    log_ratio = logp - self.b_logp[mb]
                    kls.append(float(((log_ratio.exp() - 1)
                                      - log_ratio).mean()))
                    clip_fracs.append(float(
                        ((ratio - 1).abs() > self.clip_range).float().mean()))
                pg_losses.append(float(pg_loss.detach()))
                vf_losses.append(float(vf_loss.detach()))
                cost_vf_losses.append(float(cost_vf_loss.detach()))
                ent_losses.append(float(ent_loss.detach()))
        return {
            'pg_loss': float(np.mean(pg_losses)),
            'vf_loss': float(np.mean(vf_losses)),
            'cost_vf_loss': float(np.mean(cost_vf_losses)),
            'entropy': float(np.mean(ent_losses)),
            'approx_kl': float(np.mean(kls)),
            'clip_fraction': float(np.mean(clip_fracs)),
            'std': float(torch.exp(self.policy.log_std).mean().detach()),
            'lr': float(lr),
            'lagrange': float(self.lagrange_multiplier),
            'cost_limit': float(cost_limit),
            'rollout_collision_rate': float(self._rollout_collision_rate),
        }

    # ---------- 保存 / 恢复 / SB3 导出 ----------

    def save(self, path):
        """自研格式检查点 (.pt): 权重 + 优化器 + 步数由训练侧补齐"""
        torch.save({'policy': self.policy.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'lagrange_multiplier': self.lagrange_multiplier}, path)

    def load(self, path):
        st = torch.load(path, map_location=self.device, weights_only=True)
        self.policy.load_state_dict(st['policy'])
        self.lagrange_multiplier = float(st.get(
            'lagrange_multiplier', self.lagrange_multiplier))
        # 优化器状态不恢复, 与 SB3 PPO.load 续训行为一致
        return st

    def load_policy_zip(self, path, extra_action_init=1.0,
                        extra_action_log_std=-2.0,
                        widen_noise_std=0.0):
        """从部署用 SB3/GpuPPO zip 加载 actor/critic 权重用于微调。

        除历史的观测/动作维扩展外，也支持层数相同、各隐藏层只增不减的
        Net2Wider 迁移。复制的神经元会均分下一层旧权重，因此噪声为 0 时
        扩宽前后网络函数等价；正式大模型实验可给新增神经元施加极小噪声
        打破对称。
        """
        import io
        import zipfile

        if not str(path).lower().endswith('.zip'):
            path = str(path) + '.zip'
        with zipfile.ZipFile(path) as z:
            src = torch.load(io.BytesIO(z.read('policy.pth')),
                             map_location=self.device, weights_only=True)
        target_state = self.policy.state_dict()
        widened = {}

        def _linear_indices(state, prefix):
            out = []
            for key, value in state.items():
                if not key.startswith(prefix) or not key.endswith('.weight'):
                    continue
                middle = key[len(prefix):-len('.weight')]
                if middle.isdigit() and value.ndim == 2:
                    out.append(int(middle))
            return sorted(out)

        def _widen_trunk(dst_prefix, src_prefix):
            """返回最后一层新神经元到旧神经元的映射；不兼容则返回 None。"""
            dst_idx = _linear_indices(target_state, dst_prefix)
            src_idx = _linear_indices(src, src_prefix)
            if not dst_idx or len(dst_idx) != len(src_idx):
                return None
            first_src = src[f'{src_prefix}{src_idx[0]}.weight']
            first_dst = target_state[f'{dst_prefix}{dst_idx[0]}.weight']
            if first_src.shape[1] != first_dst.shape[1]:
                return None

            # prev_map[k] 表示新网络第 k 个输入来自旧网络哪个神经元。
            prev_map = torch.arange(first_src.shape[1], device=first_src.device)
            for di, si in zip(dst_idx, src_idx):
                skw = f'{src_prefix}{si}.weight'
                skb = f'{src_prefix}{si}.bias'
                dkw = f'{dst_prefix}{di}.weight'
                dkb = f'{dst_prefix}{di}.bias'
                sw, sb = src[skw], src[skb]
                tw = target_state[dkw]
                if (tw.shape[0] < sw.shape[0]
                        or tw.shape[1] != prev_map.numel()
                        or sw.shape[1] <= int(prev_map.max())):
                    return None

                out_map = (torch.arange(tw.shape[0], device=sw.device)
                           % sw.shape[0])
                counts = torch.bincount(
                    prev_map, minlength=sw.shape[1]).to(sw.dtype)
                new_w = (sw.index_select(0, out_map)
                         .index_select(1, prev_map)
                         / counts.index_select(0, prev_map)[None, :])
                new_b = sb.index_select(0, out_map)
                if widen_noise_std > 0 and tw.shape[0] > sw.shape[0]:
                    new_w = new_w.clone()
                    new_b = new_b.clone()
                    new_w[sw.shape[0]:].add_(
                        torch.randn_like(new_w[sw.shape[0]:])
                        * float(widen_noise_std))
                    new_b[sw.shape[0]:].add_(
                        torch.randn_like(new_b[sw.shape[0]:])
                        * float(widen_noise_std))
                widened[dkw], widened[dkb] = new_w, new_b
                prev_map = out_map
            return prev_map

        pi_map = _widen_trunk('policy_net.',
                              'mlp_extractor.policy_net.')
        vf_map = _widen_trunk('value_net.',
                              'mlp_extractor.value_net.')
        if self.constrained:
            _widen_trunk('cost_net.', 'mlp_extractor.value_net.')

        def _widen_head(dst_weight, dst_bias, src_weight, src_bias, in_map,
                        allow_extra_output=False):
            if in_map is None or src_weight not in src or src_bias not in src:
                return
            sw, sb = src[src_weight], src[src_bias]
            tw, tb = target_state[dst_weight], target_state[dst_bias]
            extra = tw.shape[0] - sw.shape[0]
            if (tw.shape[1] != in_map.numel() or extra < 0
                    or (extra and not allow_extra_output)):
                return
            counts = torch.bincount(in_map, minlength=sw.shape[1]).to(sw.dtype)
            new_w = tw.clone()
            new_b = tb.clone()
            new_w[:sw.shape[0]] = (sw.index_select(1, in_map)
                                    / counts.index_select(0, in_map)[None, :])
            new_b[:sb.shape[0]] = sb
            if extra and dst_weight == 'action_net.weight':
                new_w[sw.shape[0]:].zero_()
                new_b[sb.shape[0]:].fill_(float(extra_action_init))
            widened[dst_weight], widened[dst_bias] = new_w, new_b

        _widen_head('action_net.weight', 'action_net.bias',
                    'action_net.weight', 'action_net.bias', pi_map,
                    allow_extra_output=True)
        _widen_head('value_head.weight', 'value_head.bias',
                    'value_net.weight', 'value_net.bias', vf_map)

        dst = {}
        for key in target_state:
            if key in widened:
                dst[key] = widened[key]
                continue
            if key.startswith(('policy_net.', 'value_net.')):
                src_key = 'mlp_extractor.' + key
            elif key.startswith('cost_net.'):
                # Deployment zips do not carry a safety critic. Reuse the
                # migrated task-value features as its initial backbone.
                src_key = ('mlp_extractor.value_net.'
                           + key[len('cost_net.'):])
            elif key.startswith('cost_head.'):
                dst[key] = torch.zeros_like(target_state[key])
                continue
            elif key == 'value_head.weight':
                src_key = 'value_net.weight'
            elif key == 'value_head.bias':
                src_key = 'value_net.bias'
            else:
                src_key = key
            if src_key not in src:
                if key.startswith(('step_risk_head.', 'direction_risk_head.')):
                    dst[key] = target_state[key]
                    continue
                raise KeyError(f'zip 缺少权重 {src_key!r}: {path}')
            source = src[src_key]
            target = target_state[key]
            if source.shape == target.shape:
                dst[key] = source
                continue
            expanding_observation = (
                key in ('policy_net.0.weight', 'value_net.0.weight')
                and source.ndim == 2 and target.ndim == 2
                and source.shape[0] == target.shape[0]
                and source.shape[1] < target.shape[1])
            if expanding_observation:
                # v23: observation features are appended at the end. Copy the
                # complete v21b first layer and initially ignore new features,
                # preserving both actor actions and critic values exactly.
                migrated = target.clone()
                migrated[:, :source.shape[1]].copy_(source)
                migrated[:, source.shape[1]:].zero_()
                dst[key] = migrated
                continue
            expanding_action = (
                key in ('action_net.weight', 'action_net.bias', 'log_std')
                and source.shape[0] + 1 == target.shape[0]
                and source.shape[1:] == target.shape[1:])
            if not expanding_action:
                raise ValueError(
                    f'zip 权重形状不兼容 {src_key}: '
                    f'{tuple(source.shape)} -> {tuple(target.shape)}')
            migrated = target.clone()
            migrated[:source.shape[0]].copy_(source)
            if key == 'action_net.weight':
                migrated[-1].zero_()
            elif key == 'action_net.bias':
                migrated[-1] = float(extra_action_init)
            else:
                migrated[-1] = float(extra_action_log_std)
            dst[key] = migrated
        self.policy.load_state_dict(dst)
        return src

    def export_zip(self, path, arch=None):
        """导出模型 zip (历史兼容 SB3 容器格式, 不依赖 stable_baselines3)

        内容: policy.pth (SB3 键布局的 torch state_dict) + data (JSON
        元信息)。读取侧为 ik_avoid/policy_io.load_policy (直读
        policy.pth), 与旧 SB3 save 产物双向兼容。
        """
        import io
        import json
        import zipfile

        if not path.endswith('.zip'):
            path += '.zip'
        src = self.policy.state_dict()
        dst = {}
        for k, v in src.items():
            # The safety critic is training-only; deployment remains the same
            # pure actor MLP used by all existing inference backends.
            if k.startswith(('cost_net.', 'cost_head.')):
                continue
            if k.startswith(('policy_net.', 'value_net.')):
                sb3_k = 'mlp_extractor.' + k
            elif k == 'value_head.weight':
                sb3_k = 'value_net.weight'
            elif k == 'value_head.bias':
                sb3_k = 'value_net.bias'
            else:  # action_net.*, log_std
                sb3_k = k
            dst[sb3_k] = v
        if arch is None:
            arch = [self.policy.policy_net[i].out_features
                    for i in range(0, len(self.policy.policy_net), 2)]
        buf = io.BytesIO()
        torch.save(dst, buf)
        data = json.dumps({
            'obs_dim': int(self.obs_dim), 'act_dim': int(self.act_dim),
            'net_arch': arch, 'use_sde': False,
            'policy_kwargs': {'squash_output': False},
            'step_risk_head': bool(self.policy.step_risk_head_enabled),
            'step_action_index': int(self.policy.step_action_index),
            'direction_risk_head': bool(
                self.policy.direction_risk_head_enabled),
            'direction_risk_max_residual': float(
                self.policy.direction_risk_max_residual),
            'direction_risk_margin_norm': float(
                self.policy.direction_risk_margin_norm),
        }).encode()
        with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as z:
            z.writestr('policy.pth', buf.getvalue())
            z.writestr('data', data)
        # 向后兼容: 旧名 export_sb3 别名保留一段时间
    export_sb3 = export_zip
