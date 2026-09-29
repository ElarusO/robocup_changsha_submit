# F2：正式 P103 目录核对（2026-09-29）

## 结论

正式推理包仍是 E009。目录名与 `participant_id` 都为 `P103`，入口为 `entry:build_policy`。入口只从 `artifacts/policy.npz` 读取模型；该文件是 `artifacts/` 内唯一文件，与 `submission.yaml` 的 SHA-256 `6b5c8cc0fc0279a86e3218665e4623eeb10dd4a4b5e75c642933cb87fef6ed51`、92,186 字节一致。NPZ 可以在 `allow_pickle=False` 下读取，权重维度为 104→64→64→2，另含观测均值、方差、epsilon 和裁剪阈值。

`entry.py`、`artifacts/policy.npz`、`submission.yaml`、`requirements-infer.lock` 与 E4 冻结的 E009 ZIP 解出副本逐字节一致，也与此前 Ubuntu 虚拟机检验包里的对应四个文件逐字节一致。因此原有 Linux 公开/v2/v3 检验覆盖了当前正式包的推理关键文件；没有把 E015 的入口或权重接入正式目录。该 Linux 检验仍属于本地检验，不是组织方正式评测。

推理入口只导入 NumPy 和官方 `coverage_bench` 协议/观测模块；`requirements-infer.lock` 只有说明注释，没有要求额外安装 PyTorch、SB3 或训练工具。NumPy 由官方锁定依赖提供。`artifacts/` 没有需要另行登记的文件。

## 官方提交预检

首次预检 `outputs/P103/f2-formal-directory-check-001/` 为 0 拒绝项，同时发现目录中的历史 `__pycache__`。已核实该目录只有 13 个 `.pyc` 缓存文件并清除；随后以 Python `-B` 禁止生成新字节码，在 `outputs/P103/f2-formal-directory-check-002/` 重跑预检，仍为 **0 拒绝项**，目录中文件数由 99 降到 86。

清理后预检有 34 条人工扫描提示，分布在 `train.py` 和实验/检查脚本中的 `subprocess`、`torch` 等符号。正式 `entry.py` 没有这些导入，也不调用相应脚本。提示不是预检拒绝；最终交付前仍需保留人工复核记录。预检的 `working_tree_dirty=true` 反映本地 F1 文件与记录尚未提交，本次按用户要求不创建 Git 提交。

## 交付材料待完善

- `REPORT.md` 已补充 E009 的历史训练版本说明：E009 来自 A2 修正前的 `train.py`；现在同参数运行会经过修正后的训练通路，不能声称会逐字节重现旧权重。`submission.yaml` 的训练命令记录原参数，历史 Git 提交和原训练产物需继续保留。
- 此次核对时 `LICENSE` 仍是官方模板的简短占位文本；随后已按用户确认，补入完整 MIT 授权正文，并保留组织方署名、增加 `ElarusO (P103)` 署名。详见本目录当前 `LICENSE` 与 `THIRD_PARTY.md`。这项材料修改不改变 E009 推理。
- F1/A4 原始输出在被 Git 忽略的 `outputs/P103/`，需要单独备份；正式目录内的 F1 报告只能说明结果，不能代替原始文件。

本轮清理了缓存、补全许可证并补充报告/核对记录；没有修改 `entry.py`、模型、`submission.yaml` 或推理依赖，也没有重新训练或在 A4 上挑选模型。
