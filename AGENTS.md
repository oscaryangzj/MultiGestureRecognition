# 项目协作规则

1. 运行 Python、安装依赖或测试时，始终使用名为 `MultiGestureRecognition` 的虚拟环境；不存在就先创建。
2. 规划阶段只讨论方案和修改文档。只有用户明确要求执行或实现时才改代码。
3. 使用 Git 管理版本；每次 `git commit` 前先说明改动并取得用户同意。
4. 只做必要更改，保持代码简单。不要主动加入与识别无关的哈希、指纹、跨层时间戳传递、积压/断流恢复、异步会话保护等设计。算法计算必需的时间信息可在对应模块内使用。
5. 对建模、标注、训练或解码有疑问时，先查 [Meta 开源项目](https://github.com/facebookresearch/generic-neuromotor-interface)及其[论文](https://www.nature.com/articles/s41586-025-09255-w)，再向用户确认本项目的选择。Meta 使用 sEMG，本项目使用摄像头关键点，模态相关细节不能直接照搬。
6. 跨 session 共用的数据处理、特征、模型、训练及 FSM 后处理超参数统一放在 `config.yaml`，代码中不散落硬编码的参数值。每个 session 的提示顺序、动作时长、提示间隔等采集参数保存在该 session 文件夹的 `session.json`，不放进全局配置。
7. 实现 MediaPipe 关键点提取前，查其[官方 Hand Landmarker Python 文档](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python)、[官方发布记录](https://github.com/google-ai-edge/mediapipe/releases)及 [PyPI 发布页](https://pypi.org/project/mediapipe/)，选兼容当前环境且经过本机验证的较新正式版本并固定依赖；按官方当前 Tasks Vision 接口实现，不沿用旧教程中的接口。
8. 每次修改仓库内容时，在 `CHANGELOG.md` 按日期记录实际改动；方案选择和待确认事项只写在 `DECISIONS.md`。
9. 按 README 的四个阶段分段实现和验收：MediaPipe + 静态 State → 一次性动作模型 → 一次性动作 FSM → 周期性 `waving`。完成当前阶段并经用户验收后，再开始下一阶段的代码实现。

## 文档分工

| 文档 | 只负责 |
| --- | --- |
| [README.md](README.md) | 项目目标、完整 pipeline 和仓库结构 |
| [DECISIONS.md](docs/DECISIONS.md) | 已确定的选择、当前候选方案、仍需确认的问题 |
| [CHANGELOG.md](CHANGELOG.md) | 按日期记录实际文件修改，不记录方案讨论 |

不要在多份文档重复维护同一规则。未确认的内容不得写成已定实现要求。当前不设独立实验计划文档，具体参数用验证数据选择。
