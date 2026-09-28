# B2：训练检查点、输出保护与配置记录（2026-09-25）

## 做了什么

一次训练在预定的累计样本数处保存检查点。每个 `checkpoint-实际步数/` 同时包含 `model.zip`（PPO 网络）、`vecnormalize-stats.npz`（这一时刻的观测归一化均值和方差）、`training_meta.json`（种子、参数、版本、Git 提交、训练脚本和协议的 SHA-256 等）以及 `checkpoint_manifest.json`（这三个文件的字节数和 SHA-256）。网络与统计量在同一次 `learn()` 返回后、继续采样前保存，避免把不同时刻的文件配在一起。

训练结束后，`run_manifest.json` 汇总检查点与实际总步数；根目录的模型、统计量和元数据是**最终检查点的逐字节副本**，供现有 `postprocess.py` 使用。`run_status.json` 区分运行中与完整结束。计划步数是请求值，实际步数以模型记录和清单为准：PPO 必须采满整批样本，因此两者可能不同。

`train.py` 只允许把训练写入 `outputs/P103/` 的新子目录。目标目录已经存在时立即报错，不覆盖旧实验；部分失败的目录也会保留以便排查，重试时必须另取新目录。不要手动把旧实验目录清空后复用。

## B1 固定实验的入口

在仓库根目录 `D:\Robocup_Changsha\robocup_changsha` 的 PowerShell 中执行：

```powershell
.\.venv-train\Scripts\python.exe participant/P103/run_b1.py --experiment-id E011 --dry-run
```

将编号换成 `E012` 或 `E013` 可分别核对另两个种子。`--dry-run` 只检查冻结协议摘要、分支、训练软件版本、开发验证文件摘要、参数、预期实际步数以及目标目录是否空闲，**不会训练**。这是 B3 启动前使用的命令；现在 E011–E013 目录均已有正式结果，再运行会因防覆盖保护而拒绝。三次训练均请求 1,000,000 个样本，实际完成 1,044,480；目标检查点为 313,344、626,688、1,044,480。

训练完整结束后，用以下命令核查其中一次的所有检查点。每次核查也使用新的输出文件名：

```powershell
.\.venv-train\Scripts\python.exe participant/P103/verify_b2_checkpoints.py --train-dir outputs/P103/train-E011 --output outputs/P103/b2-verify-E011/result.json
```

核查会实际加载每份 PPO 网络，检查网络步数和结构、归一化数据维度与数值、元数据、文件摘要，并确认根目录副本等于最终检查点。通过核查后，再按 B3 协议导出和评分每个检查点；`postprocess.py export --model-dir` 可以直接接受某个 `checkpoint-实际步数/` 目录。**A4 最终确认集仍保持封存**。

## 本次验证证据

- `outputs/P103/b2-smoke-001/` 与 `b2-smoke-002/` 是 768 样本的短训练，均保存了实际 384 和 768 步的两组网络与归一化文件；核查结果分别位于 `outputs/P103/b2-verify-001/result.json`、`b2-verify-002/result.json`，状态都是 `verified`，最终根目录文件与末次检查点一致。
- 重复使用已有的 `b2-smoke-001` 输出目录时，训练在开始前抛出 `FileExistsError`；原有清单的 SHA-256 未改变。
- E011、E012、E013 的 `--dry-run` 均通过，显示预定的三个检查点和 `training_started=false`。
- 官方 `scripts/check_submission.py` 预检结果位于 `outputs/P103/b2-check-001/audit-report.json`，`rejections` 为空。当前提交模型 E009 未替换。

短训练只证明保存、读取和防覆盖流程能工作，不代表 E011–E013 的得分。三次正式训练与九份检查点评分后来已在 B3 完成，结果见 `B3_B4_REPORT.md`。
