# H3D正式目录静态扫描说明

2026-10-01。完整`participant/P103`运行官方`check_submission.py`，原始结果：`outputs/P103/h3d-formal-check-001/audit-report.json`。

预检`rejections`为0，`scan_hits`为82，涉及31个Python文件。扫描命中只列为人工审核，按官方`coverage_bench/audit.py`不参与拒绝判断。命中符号分布：`subprocess`31、`subprocess.check_output`29、`subprocess.run`18、`torch`2、`torch.equal`1、`torch.no_grad`1。符号计数不等于独立安全问题数。

正式推理从`entry.py`进入，依次使用`entry_base.py`、`h3_pursuit_policy.py`、`single_factor_policy.py`及NumPy权重。这四个文件在扫描表中**没有命中**，推理调用链也不使用`subprocess`或`torch`。

`train.py`与`postprocess.py`属于训练/导出流程，需要训练环境中的PyTorch及导出一致性检查；比赛推理环境只按`requirements-infer.lock`运行。其他含`subprocess`的文件用于个人开发阶段的预检、批量评测、备份或结果分析，保存历史工作证据，不经`entry:build_policy`调用。源代码及原始命中行号可在对应文件和预检JSON逐项检查。

本说明只解释当前实际命中，不声称已获得组织方人工审核通过。如果最终提交规则要求进一步精简研究脚本，应在不丢失原始源码与实验记录的前提下另行整理提交材料并重跑预检。现阶段正式推理文件、模型与H3C Linux核验包逐字节一致。
