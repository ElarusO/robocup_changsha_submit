# H3D：正式目录已切换为E015追赶方案

日期：2026-10-01；参赛编号P103；分支P103。

## 当前实际生效的方案

**`participant/P103` 现已使用固定E015模型＋H3追赶规则。**正式入口是`entry:build_policy`，由`entry.py`构造`entry_base.py`的NumPy网络，再由`h3_pursuit_policy.py`处理动作。目标较远时更有力地朝最近可见目标推进，接近目标时恢复原组合动作，始终保留队友避让。策略只使用该机器人当前合法观测。

训练来源为E015：种子7102，`mixed-basic0-coop0`，两个基础与两个协作并行场景，请求100万步，实际1,044,480步，学习率0.0001。`submission.yaml`改用C阶段的真实训练命令与`c-protocol-v1.json`，记录模型SHA-256和92,186字节。现存`train.py`、`mixed_env.py`与C协议记载的源码摘要完全一致；训练复现仍可能受依赖、硬件及随机实现差异影响，本轮没有重新训练。

| 推理文件 | 当前SHA-256 | 与H3C冻结候选 |
| --- | --- | --- |
| `entry.py` | `809241193e33233b349d8328c7579bae28c250c1902fc844ca63b44c00227d64` | 逐字节一致 |
| `entry_base.py` | `9c9d8b1ce8ee5cef998025065901bd93cdaffda63c8b251f6ba6da10ef87ad3b` | 逐字节一致 |
| `single_factor_policy.py` | `79177a1ace6a12484093ee58e1742b3f63a32cd7844b25716090970b26b8ee7e` | 逐字节一致 |
| `h3_pursuit_policy.py` | `84467e47244342a4aacdf8d4edf53db2fe0b931f33e5d250bcbad8b7ff0f4b77` | 逐字节一致 |
| `artifacts/policy.npz` | `0bde83fe3983920ee50446f218d54a58e886ee609bc00fcfb5b68efc95c11173` | 逐字节一致 |
| `requirements-infer.lock` | `4e4827ea494097d73c3fbf936e1396cd1bad2a52d58655635aac82f14c4ac9e8` | 逐字节一致 |

训练说明和`submission.yaml`元数据按当前正式版本更新，因此清单文件自身不同于之前标记为“experimental only”的H3C研究副本；上表中实际推理文件保持完全一致。

## 备份与工作区处理

切换前核对E009入口、权重、清单、推理锁与E4冻结回退包相同，并将当时整个P103目录189个文件封装在`outputs/P103/h3d-switch-001/P103_before_H3D_E009_2026-10-01.zip`。ZIP和成员摘要逐一验证，ZIP SHA-256为`ab0bd9afb56c486138a4ded37c70149863f7b4866456b3dc4b303f53a24e3b30`。原E4 E009独立回退ZIP和H3C对照副本仍保留。

旧网络类从`entry.py`移到内容完全相同的`entry_base.py`。13个只用于历史离线诊断的脚本，其导入语句改为`from entry_base import NumpyMLPPolicy`，避免今后导入失败；没有改变这些脚本的分析算法。它们旧版本连同切换前状态均在备份内，旧实验的冻结源码摘要仍对应原版本。

## 对实际正式目录的回归

- `.venv`中运行官方`check_submission.py`：硬拒绝0，另有82条`scan_hits`，分布于31个训练/历史研究脚本。它们是人工审核提示，不是硬拒绝；实际推理链`entry.py`、`entry_base.py`、`h3_pursuit_policy.py`、`single_factor_policy.py`没有命中。原始审计结果在`outputs/P103/h3d-formal-check-001/`，用途说明见[H3D扫描说明](H3D_SCAN_HITS.md)。
- 运行官方公开套件8/8回合：`status=ok`、`errors=[]`、Windows本地预览200.00分，结果`outputs/P103/h3d-formal-public-001/`。
- 八个逐回合结果与H3C冻结候选的Windows和Ubuntu公开测试逐项一致（状态、动作计数、回报、覆盖、碰撞等，时延除外），模型/套件/推理锁摘要相同。
- H3C的固定候选此前已在Ubuntu20.04/Python3.12.13核验入口，并在新v10同场景比E009高18.07分，配对95%区间[11.53,24.93]，两组覆盖提升、碰撞下降。新正式目录的推理文件与那份Linux核验候选逐字节一致；本轮没有再次用Ubuntu运行修改过训练元数据的整个正式目录。

公开测试只有8回合，不能单独证明候选优于E009；换模依据是H3B同模型确认和H3C新v10单模型公平比较。全部成绩均为本地运行，未获组织方评分。

## 接下来的状态

`experiments.csv`新增一条实际正式目录公开评测，记录总数171。新正式版本与E009回退版一并保存于`outputs/P103/h3d-backup-002/`，另有C盘文档副本；ZIP与成员摘要核对。`h3d-backup-001`保留首次封存的历史快照，后来补正了扫描说明，因此以002为本轮交付备份。**本轮未进行Git提交、推送或最终PR。**v10已见；v4—v7未评分。后续迭代需以当前E015追赶入口作为正式基线，修改模型或策略后重新冻结和评测。
