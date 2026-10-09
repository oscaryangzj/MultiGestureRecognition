# 项目协作规则

1. 运行 Python、安装依赖或测试始终使用 `MultiGestureRecognition` 虚拟环境；不存在则先创建。
2. macOS 训练使用可用的 PyTorch MPS GPU，不默认退到 CPU。
3. 提问只回答；修改文档、实现代码均须用户明确指令。要求两者同步时才同时修改。
4. 使用 Git 管理版本；每次 commit 前说明改动并取得用户同意。
5. 最小更改、代码简洁；不主动加入识别无关的哈希、指纹、跨层时间戳、积压/断流恢复或异步会话保护。算法必需时间信息在对应模块内使用。
6. 建模、标注、训练或解码有疑问时，先查 [Meta 项目](https://github.com/facebookresearch/generic-neuromotor-interface)及[论文](https://www.nature.com/articles/s41586-025-09255-w)，再向用户确认。区分 sEMG 与摄像头模态。
7. 共用数据处理、特征、模型、训练及 FSM 超参数集中在 `config.yaml`；每次采集的顺序、时长、间隔等保存在自己的 `session.json`。
8. 实现 MediaPipe 提取前查 [官方 Python 文档](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python)、[发布记录](https://github.com/google-ai-edge/mediapipe/releases)和 [PyPI](https://pypi.org/project/mediapipe/)，选择兼容且本机验证的较新正式版并固定依赖；使用 Tasks Vision 接口。
9. 每次修改在 `docs/CHANGELOG.md` 按日期记录实际文件改动；选择与待确认事项写 `docs/DECISIONS.md`。
10. 内部实现顺序：MediaPipe + 静态 State → 一次性 Action → 一次性 FSM → waving。完整版本实现时完成 README 全流程后统一交付，不逐阶段等待；分别报告代码完成、真实训练和效果验收，不把方案或旧结果当作最终验收。

## 文档分工

| 文件 | 职责 |
| --- | --- |
| [`README.md`](../README.md) | 目标、方法、完整 pipeline、当前结果与结构 |
| [`docs/AGENTS.md`](AGENTS.md) | 协作与修改规则 |
| [`docs/DECISIONS.md`](DECISIONS.md) | 选择依据、待确认事项 |
| [`docs/CHANGELOG.md`](CHANGELOG.md) | 按日期记录实际改动 |

同一规则不重复维护；未确认内容不写成要求。不设独立实验计划文档，参数用验证数据选择。
