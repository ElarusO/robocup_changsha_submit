# P103 策略与实验报告（2026-09-25）

## 最终方案

提交模型选自 **E009**。使用官方模板的 Stable-Baselines3 PPO 网络架构（104 维局部观测 → 64 → 64 → 二维连续动作），在 basic-0 场景中从头训练；正式推理只用 NumPy，按单个机器人的局部观测独立输出动作。没有改动官方环境、评分程序、观测协议或训练奖励。entry.py 加载 artifacts/policy.npz，submission.yaml 中的 SHA-256 和字节数与该文件一致。

E009 训练种子为 7101，请求 1,000,000 步，因 PPO 按完整 rollout 更新，实际完成 **1,044,480 步**；Windows CPU 训练耗时 **319.0 秒**。关键参数：learning_rate=0.0001、gamma=0.99、gae_lambda=0.95、n_steps=512、batch_size=256、ent_coef=0、4 个向量环境，观测用 VecNormalize 归一化。模型文件大小 92,186 字节，SHA-256：

6b5c8cc0fc0279a86e3218665e4623eeb10dd4a4b5e75c642933cb87fef6ed51

导出时在 256 个真实环境观测上核对 SB3 与 NumPy 的确定性动作，最大绝对差 3.604e-07，低于 1e-5 容差。E009 预检无拒绝项。训练模型、归一化统计量与导出记录保存在本地 outputs/P103/train-E009/、outputs/P103/post-E009/；这些运行产物不进入 Git，正式推理需要的权重已在本人目录的 artifacts/policy.npz 中。

## 对照实验与选择依据

公开分数来自官方公开套件的 Windows 本地预览（8 回合）；本地验证分数来自 validation-suite-v1.yaml 与 validation-seeds-v1.json（40 回合，场景种子与公开套件不同）。两列均不是组织方正式核验成绩。下表是本次实测值，具体运行 ID、配置、模型 SHA-256 和分组指标见 experiments.csv；逐回合原始结果保存在对应的 outputs/P103/eval-EXXX/ 与 holdout-EXXX/。

| 实验 | 训练场景 | 学习率 | 实际步数 | 公开分 | 本地验证分 |
| --- | --- | ---: | ---: | ---: | ---: |
| E001 模板 | 无 | 无 | 无 | 66.67 | 10.00 |
| E002 短训练 | basic-0 | 0.0003 | 6144 | 75.00 | 未测 |
| E003 | basic-0 | 0.0003 | 110592 | 141.67 | 30.67 |
| E004 | basic-0 | 0.0003 | 307200 | 83.33 | 19.67 |
| E005 | basic-0，种子 7102 | 0.0003 | 110592 | 91.67 | 24.67 |
| E006 | coop-0 | 0.0003 | 110592 | 125.00 | 24.67 |
| E007 | basic-0 | 0.0001 | 307200 | 141.67 | 58.33 |
| E008 | basic-0 | 0.0001 | 614400 | **216.67** | 91.33 |
| **E009（提交）** | basic-0 | 0.0001 | 1044480 | 200.00 | **108.33** |
| E010 | basic-0 | 0.0001 | 1536000 | 175.00 | 88.33 |

降低学习率后，E007 到 E009 的本地验证分数随训练步数上升；继续延长到 E010 时，公开与本地验证分数均下降，因此停止这一组参数的迭代。这只说明当前对照中没有继续提升，不能证明算法已达到理论上限。E008 的公开分最高，但 E009 在 40 回合固定本地验证中最高，且本地验证的两个分组均未发生碰撞，因此选择 E009。公开测试仅 8 回合，E008 和 E009 的分差不足以证明其隐藏场景表现有先后顺序。

E009 公开评测 8/8 回合完成，status=ok、provenance=local_preview，总分 **200.00**；基础组平均覆盖率 **0.1167**，协作组 **0.2833**，两组平均碰撞率均为 **0**。本地验证 40/40 回合完成，总分 **108.33**；基础组平均覆盖率 **0.1133**，协作组 **0.1033**，两组平均碰撞率也均为 **0**。模板 E001 的公开分为 66.67，本次选定模型在同一公开套件上的预览分约为其 3 倍；这不代表正式比赛成绩也会提高同样比例。

## 复现方法

在仓库根目录的 Windows PowerShell 中运行。.venv-train 安装训练依赖，.venv 安装官方评测依赖；完整环境配置见仓库 docs/how_to_use.md。复现时使用新的实验编号与输出目录，不覆盖已有结果。以下以 E011 为例：

~~~powershell
.\.venv-train\Scripts\python.exe participant/P103/train.py --total-steps 1000000 --curve-points 10 --seed 7101 --train-case basic-0 --learning-rate 0.0001 --out outputs/P103/train-E011
.\.venv-train\Scripts\python.exe participant/P103/postprocess.py export --model-dir outputs/P103/train-E011 --experiment-id E011
.\.venv-train\Scripts\python.exe participant/P103/postprocess.py activate --experiment-id E011
.\.venv\Scripts\python.exe scripts/check_submission.py --submission participant/P103 --output outputs/P103/check-E011
.\.venv\Scripts\python.exe scripts/evaluate_one.py --submission participant/P103 --suite configs/public-suite-v1.yaml --seeds configs/public-seeds-v1.json --output outputs/P103/eval-E011
.\.venv\Scripts\python.exe scripts/evaluate_one.py --submission participant/P103 --suite participant/P103/validation-suite-v1.yaml --seeds participant/P103/validation-seeds-v1.json --output outputs/P103/holdout-E011
.\.venv-train\Scripts\python.exe participant/P103/postprocess.py restore --experiment-id E011
~~~

activate 会暂时替换提交模型，restore 会恢复本报告选择的 E009。由于框架版本、硬件和随机性，重新训练未必得到逐字节相同的模型。若只想复查当前提交模型，可直接运行预检和两条评测命令，并将 --output 改为全新的目录，例如 outputs/P103/eval-final-001。

## 局限与后续方向

- 训练只使用 basic-0 的 3 机器人、3 目标、10 步场景。虽然在协作组也有收益，当前没有训练多场景混合策略，也没有证明对其他规模或隐藏种子的稳定性。
- 分组得分的区间很宽。E009 公开总分的 95% 区间为 [33.33, 366.67]；40 回合本地验证为 [38.33, 190.04]。这些区间与其他候选重叠，不能把小幅分差当成确定改进。
- 固定本地验证集虽使用不同场景种子，但已被反复用于模型选择，存在选择偏差。它不能替代组织方 Linux 隐藏测试，也不能预测正式名次。Windows local_preview 不执行完整的阶段时限核验。
- 本轮只调了训练场景、随机种子、学习率和总步数；尚未尝试多场景采样、奖励设计、观测历史或其他算法。后续如果继续开发，应新增独立输出目录和实验记录，优先做多种子重复训练及更广的场景验证。

## 追加复测：开发验证 v2（A5，2026-09-25）

在新的固定开发验证 v2 上，以相同的 100 个场景、每场景 2 次重复和相同推理入口复测历史模型：E001 得 **51.67**，E008 得 **114.80**，E009 得 **125.33**，均完成 200/200 回合，均为 Windows `local_preview`。E009 比 E008 高 10.53 分，但同场景配对重采样差值区间为 [-0.27, 22.33]，包括 0，不能据此断言 E009 稳定优于 E008。E009 仍作为当前提交模型；正式模型文件没有被本次复测替换。详细分组结果、模型 SHA-256、运行 ID 和限制见 `A5_REPORT.md`。A4 最终确认集仍未用于评分。

## 后续开发实验附记（2026-09-26）

本报告前面的提交模型说明仍专指 **E009**。后续已在 P103 目录中完成 B3 三种子训练、C 基础/协作 1:1 混合训练，以及 D 阶段的轨迹和参数对照；这些均为**本地开发候选**，没有替换 `artifacts/policy.npz` 或修改 `submission.yaml`。C 的最终三个种子在开发验证 v2 的均分为 **130.71**，详见 `C_REPORT.md`。

目标朝向单因素在独立新建的开发场景 v3 中，相对三份 C 模型平均加分 **7.04**；“目标朝向→队友避让”组合又比目标单项平均加 **0.76**，但在旧开发 v2 上平均低 **0.96**，尚不能认为组合稳定更好。E016 在一个旧场景漏目标的逐步原因、两套开发题上的分组表现见 `D_TRAJECTORY_FOLLOWUP.md`。

D 参数搜索按冻结规则依次比较学习率与探索系数。较高学习率 `0.0002` 在短预算比 C 平均高 5.18 分，但完整预算只高 **1.00** 分，未达到预先设定的 +5 门槛，故按规则停止，继续使用 C 的 `0.0001` / `0` 训练配置作为开发基线。低学习率和增加探索系数的对照结果、每个训练种子、中间检查点、导出校验及本地预览限制见 `D_PARAMETER_REPORT.md`。所有新实验在 `LOG.md` 和 `experiments.csv` 留有记录；A4 最终确认集未用于挑选模型，组织方尚未核验这些本地分数。
