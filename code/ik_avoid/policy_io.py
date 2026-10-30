"""策略权重加载 (无 stable_baselines3 依赖)

模型文件为 SB3 格式 zip (历史兼容容器, 由 GpuPPO.export_zip 或旧 SB3
save 产出): 内含 policy.pth (torch state_dict) 与 data (JSON 元信息)。
本模块直接 zipfile + torch.load 解析, 推理/评测全链路不再 import SB3。
支持两个可互换的部署后端:
- numpy: 直接执行 zip 中的 MLP 权重 (默认, 向后兼容)
- onnx_fp32 / onnx_int8: ONNX Runtime CPUExecutionProvider

确定性动作语义 (与 SB3 BasePolicy.predict 对齐):
- 前向: shared_net/policy_net (Linear+Tanh 交替, 含末层激活) → action_net
- squash 判定: 仅 gSDE(use_sde)+squash_output 时对 mu 加 tanh;
  本项目 PPO Box 动作为 clip 路径 (squash=False)
- 末尾 clip 到动作空间边界 (SB3 对 Box 空间的行为)
"""
import os

import numpy as np


def load_policy_from_zip(model_path):
    """直接从 SB3 格式 zip 读策略权重, 返回 (ops, action_lo, action_hi)

    ops 为算子列表 [('lin', W, b) | ('tanh',)]; state_dict 键布局:
    mlp_extractor.(shared_net|policy_net).{偶数下标}.weight|bias +
    action_net.weight|bias (SB3 create_mlp: 每个 Linear 后接 Tanh, 含末层)。
    """
    import base64
    import json
    import pickle
    import zipfile

    import torch

    if not model_path.endswith('.zip'):
        model_path += '.zip'
    data_d = None
    with zipfile.ZipFile(model_path) as z:
        sd = torch.load(z.open('policy.pth'), map_location='cpu',
                        weights_only=True)
        try:
            data_d = json.loads(z.read('data'))
        except Exception:
            pass
    # 动作空间边界: 从 data JSON 的序列化 Box 还原, 失败回退 [-1, 1]
    lo, hi = -1.0, 1.0
    if data_d is not None:
        try:
            box = pickle.loads(base64.b64decode(
                data_d['action_space'][':serialized:']))
            lo, hi = float(box.low.min()), float(box.high.max())
        except Exception:
            pass

    # 骨干层: 优先 shared_net (list net_arch 布局), 其次 policy_net
    prefix = None
    for cand in ('mlp_extractor.shared_net.', 'mlp_extractor.policy_net.'):
        if any(k.startswith(cand) for k in sd):
            prefix = cand
            break
    if prefix is None:
        raise ValueError(f'zip 内未找到策略权重: {model_path}')
    idxs = sorted({int(k.split('.')[2]) for k in sd
                   if k.startswith(prefix) and k.endswith('.weight')})
    assert idxs == list(range(0, 2 * len(idxs), 2)), \
        f'非标准 SB3 MLP 布局 (Linear@偶数下标): {idxs}'
    ops = []
    for i in idxs:
        ops.append(('lin',
                    sd[f'{prefix}{i}.weight'].numpy().astype(np.float32),
                    sd[f'{prefix}{i}.bias'].numpy().astype(np.float32)))
        ops.append(('tanh',))
    ops.append(('lin',
                sd['action_net.weight'].numpy().astype(np.float32),
                sd['action_net.bias'].numpy().astype(np.float32)))
    # squash 判定与 SB3 一致: 仅 SquashedDiagonalGaussianDistribution
    # (use_sde + squash_output) 才对 mu 加 tanh
    squash = False
    if data_d is not None:
        try:
            squash = bool(data_d.get('use_sde')) and bool(
                data_d.get('policy_kwargs', {}).get('squash_output'))
        except Exception:
            pass
    if squash:
        ops.append(('tanh',))
    risk_head = None
    if data_d and data_d.get('step_risk_head'):
        required = (
            'step_risk_head.0.weight', 'step_risk_head.0.bias',
            'step_risk_head.2.weight', 'step_risk_head.2.bias')
        if not all(key in sd for key in required):
            raise ValueError(f'zip 缺少独立步长头权重: {model_path}')
        risk_head = {
            'w1': sd[required[0]].numpy().astype(np.float32),
            'b1': sd[required[1]].numpy().astype(np.float32),
            'w2': sd[required[2]].numpy().astype(np.float32),
            'b2': sd[required[3]].numpy().astype(np.float32),
            'action_index': int(data_d.get('step_action_index', -1)),
        }
    if data_d and data_d.get('direction_risk_head'):
        required = (
            'direction_risk_head.0.weight', 'direction_risk_head.0.bias',
            'direction_risk_head.2.weight', 'direction_risk_head.2.bias')
        if not all(key in sd for key in required):
            raise ValueError(f'zip 缺少风险方向残差头权重: {model_path}')
        if risk_head is None:
            risk_head = {}
        risk_head['direction'] = {
            'w1': sd[required[0]].numpy().astype(np.float32),
            'b1': sd[required[1]].numpy().astype(np.float32),
            'w2': sd[required[2]].numpy().astype(np.float32),
            'b2': sd[required[3]].numpy().astype(np.float32),
            'max_residual': float(data_d.get(
                'direction_risk_max_residual', 0.15)),
            'margin_norm': float(data_d.get(
                'direction_risk_margin_norm', 0.20)),
            'action_index': int(data_d.get('step_action_index', -1)),
        }
    return ops, lo, hi, risk_head


class NumpyPolicy:
    """确定性策略前向 (numpy), 签名兼容 SB3 BasePolicy.predict

    predict(obs, deterministic=True) -> (action, None); 支持单条 (D,)
    与批量 (N,D) 输入; 末尾 clip 到动作空间边界 (对齐 SB3 Box 行为)。
    """

    def __init__(self, ops, action_lo=-1.0, action_hi=1.0,
                 risk_head=None):
        self.ops = ops
        self.action_lo = action_lo
        self.action_hi = action_hi
        self.risk_head = risk_head

    def _forward(self, x):
        obs = x
        for op in self.ops:
            if op[0] == 'lin':
                x = x @ op[1].T + op[2]
            elif op[0] == 'tanh':
                x = np.tanh(x)
            else:
                x = np.maximum(x, 0.0)
        if self.risk_head is not None and 'w1' in self.risk_head:
            h = np.tanh(obs[..., -2:] @ self.risk_head['w1'].T
                        + self.risk_head['b1'])
            residual = h @ self.risk_head['w2'].T + self.risk_head['b2']
            x = x.copy()
            x[..., self.risk_head['action_index']] += residual[..., 0]
        direction = (None if self.risk_head is None
                     else self.risk_head.get('direction'))
        if direction is not None:
            h = np.tanh(obs @ direction['w1'].T + direction['b1'])
            residual = np.tanh(h @ direction['w2'].T + direction['b2'])
            margin = max(direction['margin_norm'], 1e-6)
            gate = np.clip((margin - obs[..., -1]) / margin, 0.0, 1.0)
            x = x.copy()
            index = direction['action_index']
            x[..., :index] += (residual * direction['max_residual']
                               * gate[..., None])
        return np.clip(x, self.action_lo, self.action_hi).astype(np.float32)

    def forward_batch(self, obs):
        """批量前向 (N,D)->(N,A)"""
        return self._forward(np.asarray(obs, np.float32))

    def predict(self, obs, deterministic=True, state=None):
        """SB3 兼容接口: 返回 (action, state=None)"""
        x = np.asarray(obs, np.float32)
        single = x.ndim == 1
        a = self._forward(x[None] if single else x)
        return (a[0], None) if single else (a, None)


class ONNXPolicy:
    """ONNX Runtime 确定性策略前向, 接口兼容 NumpyPolicy/SB3。

    ONNX 图只包含策略 MLP; 动作空间裁剪在图外执行, 与 NumpyPolicy 和
    历史 SB3 ``predict`` 语义一致。线程数固定为 1, 避免单条小 MLP 被
    线程池调度开销淹没。
    """

    def __init__(self, onnx_path, intra_threads=1, inter_threads=1,
                 action_lo=-1.0, action_hi=1.0):
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                'ONNX 后端需要 onnxruntime; 请执行 pip install onnxruntime'
            ) from exc
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(f'ONNX 模型不存在: {onnx_path}')
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = int(intra_threads)
        opts.inter_op_num_threads = int(inter_threads)
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            onnx_path, sess_options=opts, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.onnx_path = onnx_path
        self.action_lo = action_lo
        self.action_hi = action_hi

    def forward_batch(self, obs):
        x = np.asarray(obs, dtype=np.float32)
        if x.ndim == 1:
            x = x[None]
        out = self.session.run(None, {self.input_name: x})[0]
        return np.clip(out, self.action_lo, self.action_hi).astype(np.float32)

    def predict(self, obs, deterministic=True, state=None):
        x = np.asarray(obs, dtype=np.float32)
        single = x.ndim == 1
        out = self.forward_batch(x)
        return (out[0], None) if single else (out, None)


def _model_stem(model_path):
    path = os.fspath(model_path)
    return path[:-4] if path.lower().endswith('.zip') else path


def default_onnx_path(model_path, backend):
    """由模型 zip 路径派生标准 ONNX 文件名。"""
    suffix = {
        'onnx': '_policy_fp32.onnx',
        'onnx_fp32': '_policy_fp32.onnx',
        'onnx_int8': '_policy_int8.onnx',
    }
    try:
        return _model_stem(model_path) + suffix[backend]
    except KeyError as exc:
        raise ValueError(f'未知 ONNX 后端: {backend}') from exc


def load_policy(model_path, backend='numpy', onnx_path=None):
    """加载确定性策略后端。

    Args:
        model_path: 历史兼容 zip 路径 (可省略 ``.zip``)。
        backend: ``numpy`` (默认)、``onnx_fp32`` 或 ``onnx_int8``。
        onnx_path: 显式 ONNX 路径; 省略时从 ``model_path`` 自动派生。
    """
    backend = str(backend).lower()
    if backend == 'numpy':
        ops, lo, hi, risk_head = load_policy_from_zip(os.fspath(model_path))
        return NumpyPolicy(ops, action_lo=lo, action_hi=hi,
                           risk_head=risk_head)
    if backend in ('onnx', 'onnx_fp32', 'onnx_int8'):
        path = os.fspath(onnx_path) if onnx_path else default_onnx_path(
            model_path, backend)
        return ONNXPolicy(path)
    raise ValueError(
        f'未知策略后端 {backend!r}; 可选 numpy/onnx_fp32/onnx_int8')
