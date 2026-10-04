# 第三方依赖与许可声明

训练阶段使用以下第三方库；这些库只在训练或模型导出时使用，不随模型产物打包：

| 库 | 来源 | 许可证 |
| --- | --- | --- |
| Stable-Baselines3 | https://github.com/DLR-RM/stable-baselines3 | MIT |
| SuperSuit | https://github.com/Farama-Foundation/SuperSuit | Farama 自有部分 MIT；仓库另含 MIT / Apache-2.0 来源代码，见其 LICENSE |
| PyTorch | https://github.com/pytorch/pytorch | BSD-3-Clause |

推理产物为纯 NumPy 前向，无额外第三方源码依赖。NumPy 是推理依赖，版本见 `requirements-infer.lock`；来源：https://github.com/numpy/numpy，许可证：BSD-3-Clause。

参赛目录沿用[组织方官方模板及协议接口](https://github.com/HNU-BigHeroX/robocup_changsha)；官方仓库根目录的 `LICENSE` 为 MIT。
P103 新增代码的授权文本和 `ElarusO (P103)` 署名见本目录 `LICENSE`。
模型权重由本项目训练产生，未引入外部预训练模型。
