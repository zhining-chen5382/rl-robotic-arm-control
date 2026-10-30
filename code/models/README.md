# Included inference checkpoint

Only `panda_ppo_ik_avoid_v30_risk_head_ft_mjx_best/best_model.zip` is included, with the applicant's approval.

- Robot: Franka Panda.
- Observation/action dimensions: 58/8.
- Actor architecture: 512/512/256/128, with the independent risk-step head enabled.
- Default backend: NumPy; the historical ONNX exporter does not support this risk head.
- The ZIP contains `policy.pth` and JSON `data`; it is an inference checkpoint, not a complete optimizer/resume snapshot.
- SHA-256: `d8f9db135291e3dde06d40b02c43f31eeba849c266896e890e794c441b0905b8`.

The file is unchanged from the reviewed research archive. Historical v15/v16 and PyBullet checkpoints are deliberately excluded and cannot be replaced by this model. Load only trusted checkpoints. No general downstream reuse license for the trained policy has been selected; see [provenance and scope](../../THIRD_PARTY_NOTICES.md).
