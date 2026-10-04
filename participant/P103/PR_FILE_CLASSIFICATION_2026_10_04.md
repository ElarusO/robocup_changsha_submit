# P103 文件分类与最终 PR 范围（2026-10-04）

本表是提交整理清单，不移动原文件。很多历史脚本和报告记录了固定路径、源码摘要、实验编号；直接重排目录会破坏旧复现说明。完整开发状态与原始输出继续保留在本机备份，供需要时核查。

| 类别 | 文件 | 最终用途 |
| --- | --- | --- |
| 正式推理，必须入 PR | `entry.py`、`entry_base.py`、`h3_pursuit_policy.py`、`single_factor_policy.py`、`artifacts/policy.npz`、`submission.yaml`、`requirements-infer.lock` | 组织方加载的 E015＋H3 策略、模型与清单 |
| 训练与导出，入 PR | `train.py`、`mixed_env.py`、`postprocess.py`、`c-protocol-v1.json` | 说明和复现正式 E015 权重的训练配置；不进入正式推理链 |
| 授权与主记录，必须入 PR | `LICENSE`、`THIRD_PARTY.md`、`LOG.md`、`experiments.csv`、`REPORT.md` | 作者授权、第三方来源、真实实验与最终报告 |
| 研究证据，拟入 PR | 本目录的其余 `.md` 与 `linux_eval_setup.sh` | 保存事前协议、对照报告和 Linux 搭建说明；不能当组织方正式成绩 |
| 历史实验程序，保存在本地完整备份 | 除上列正式/训练文件外的 `*.py`，例如 `run_g*`、`diagnose_*`、`train_sampled.py` | 已运行过的研究工具，正式模型不会导入；避免大量无关敏感符号触发人工审计 |
| 历史本地场景，保存在本地完整备份 | 除 `c-protocol-v1.json` 与 `submission.yaml` 外的开发/确认 `.yaml`、`.json` | 这些场景仅供本地比较，正式评分不读取；若评审需要，可连同原始结果单独提供 |
| 本地输出，不入 PR | `outputs/P103/`、训练检查点、临时提交包、评测原始结果、备份 ZIP/Git bundle、虚拟环境和 `__pycache__/` | 本机留存与复核，Git 不跟踪 |

已提交到私有 `P103` 分支的历史 Python 文件仍在私有分支中；**本清单不会通过删除、重写或强推改变该分支**。拟提交的精简副本单独放在 `outputs/P103/pr-review-2026-10-04-003/P103/` 验证。若用它创建最终 PR，必须先从最新官方 `master` 构建干净的 `P103` 来源分支，核对差异只在 `participant/P103/`，再由用户决定是否公开推送。当前私有 `origin` 继续保存完整开发历史。

完整 P103 目录的 2026-10-04 预检为 0 个硬拒绝、96 个仅供人工审查的扫描提示；公开套件 8/8 回合，Windows `local_preview` 200.00。精简副本的独立预检与公开测试结果另记在 `PR_PREPARATION_2026_10_04.md`，未完成前不能称为 PR 就绪。
