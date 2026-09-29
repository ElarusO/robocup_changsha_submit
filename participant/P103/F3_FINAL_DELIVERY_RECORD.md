# F3：E009 交付核验与备份登记（2026-09-29）

## 当前交付选择

正式 `participant/P103/` 保留 E009。F1 按事前 A4 门槛没有换成 E015 组合版，见 `F1_A4_PRECOMMITTED_PROTOCOL.md` 与 `F1_A4_REPORT.md`。这次没有训练、替换入口或改写模型。

| 项目 | 当前值 |
| --- | --- |
| 编号与入口 | `P103`；`entry:build_policy` |
| 模型 | `artifacts/policy.npz`；92,186 字节；SHA-256 `6b5c8cc0fc0279a86e3218665e4623eeb10dd4a4b5e75c642933cb87fef6ed51` |
| 入口 SHA-256 | `9c9d8b1ce8ee5cef998025065901bd93cdaffda63c8b251f6ba6da10ef87ad3b` |
| 清单 SHA-256 | `634a6d3bad98321fa3427726c8d9cd5d085da8138d8aa48f90cd4d343e4cd36d` |
| 推理锁 SHA-256 | `4e4827ea494097d73c3fbf936e1396cd1bad2a52d58655635aac82f14c4ac9e8` |
| 本记录初稿时的 Git HEAD | `bd47bca076c49ddfab82491a6045336c6e6ae3c9`；本轮材料提交后以最新 Git 历史为准 |

## 已取得的检查证据

- `F2_FORMAL_DIRECTORY_AUDIT.md` 记录正式入口、模型清单、推理依赖和与 E4 冻结包及既有 Ubuntu 检验包的逐字节比对。清理 13 个 `.pyc` 缓存后，`outputs/P103/f2-formal-directory-check-003/audit-report.json` 的拒绝项为 0，人工扫描提示 34 条，均不在正式推理入口。
- 当前正式目录在官方公开套件再次完成 8/8 回合，`status=ok`、无错误、Windows `local_preview` **200.00**，模型摘要为上表 E009；原始文件位于 `outputs/P103/final-public-E009-001/`。这与此前冻结 E009 公开结果相同，不是组织方正式成绩。
- 补全 MIT 许可证后重新运行正式目录预检与公开套件：`outputs/P103/final-check-E009-license-001/audit-report.json` 拒绝项 0；`outputs/P103/final-public-E009-license-001/result.json` 为 `status=ok`、8/8 回合、Windows `local_preview` **200.00**，模型 SHA-256 仍与上表一致。该次复测已登记在 `experiments.csv`。
- 既有 Ubuntu 20.04.6 / Python 3.12.13 检验完成预检和公开/v2/v3 评测；与当前目录相比，入口、模型、清单和推理锁四个文件逐字节相同，见 `LINUX_VM_CHECK.md`。Linux 检验属于本地虚拟机，不代表组织方镜像。
- `REPORT.md` 已把复核命令改成只检查当前 E009，并注明 E009 来自 A2 修正前的历史训练代码；当前训练脚本的同参数运行不能视为逐字节重训旧权重。最新公开复测已登记在 `experiments.csv`。

## 备份与剩余事项

许可证补全前的第一次本地备份保存在 `outputs/P103/final-backup-001/`。补全后的新备份存放于 `outputs/P103/final-backup-licensed-001/`。其中 `P103-delivery-and-evidence.zip` 保存当前完整 P103 目录及 E009 训练/导出、E4 两份冻结包、F1 原始结果、Linux 回传与最终预检/公开测试证据；`P103-history.bundle` 保存已提交的 P103 Git 历史。具体文件列表、逐文件 SHA-256 和备份文件 SHA-256 见同目录 `backup-manifest.json` 与 `SHA256SUMS.txt`。按用户选择，另复制到 C: 文档目录作为第二卷副本；C: 与 D: 是否为不同物理磁盘尚未证实。

本次按用户确认把 `LICENSE` 补为完整 MIT 文本，保留组织方署名并加入 `ElarusO (P103)`；`THIRD_PARTY.md` 已说明模板来源和训练侧依赖。原有 `bd47bca` 已通过 SSH 443 推送到私有 origin；本轮 F1–F3 与许可证材料另行提交。正式公开提交前仍需核实提交窗口及 PR 状态。`outputs/` 被 Git 忽略，不能只依靠 Git 提交保存原始评测结果。
