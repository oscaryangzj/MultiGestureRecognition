# 修改记录

按日期记录仓库文件的实际改动；设计决定和待确认事项见 [DECISIONS.md](docs/DECISIONS.md)。

## 2026-09-30

- 将 Conda YAML 依赖文件替换为传统的 `requirements.txt`；将 MediaPipe 模型资源从 `data/models/` 移至项目根目录 `models/`，并同步更新配置、忽略规则和文档。
- 代码审核修正：采集拒绝覆盖已有 session 数据；离线提取与录像回放共用逐帧采样时间，并仅在关键点提取完整成功后替换结果文件。
- 标注滑块移动时及时刷新画面，数字键映射读取 State 类别配置；训练前检查两组数据的类别覆盖，并按样本数计算训练和验证损失。
- 明确 State 正类由用户标注，未标正类且连续检测到手的帧自动生成 `NULL` 目标；连续可见帧数放在 `config.yaml` 并保留候选状态。
- 实现第一阶段 MediaPipe + 静态 State 流程：State session 采集、21 点提取、人工区间标注与自动 `NULL`、单帧 MLP 训练、验证指标、录像回放和摄像头输出。
- 新增 `environment.yml` 并创建 `MultiGestureRecognition` 环境；固定 MediaPipe 0.10.35、PyTorch、OpenCV、NumPy 与 PyYAML 版本，加入官方 Hand Landmarker 模型下载入口。MediaPipe 官方模型在本机通过 Tasks Vision `VIDEO` 单帧推理。
- 加入 State 训练/验证 session 隔离检查；修正类别指标计算，并让标注器在退出时保存未手动保存的修改。
- 修正 State 类别配置中 `NULL` 的 YAML 字符串解析，并调整标注界面使缺手帧始终显示为未监督。
- 更新 README，列出第一阶段的安装、session 格式、采集、提取、标注、数据划分、训练和推理命令，并标明后续阶段尚未实现。

## 2026-09-29

- 将实现顺序改为四段验收：MediaPipe + 静态 State、一次性动作模型、一次性动作 FSM、周期性 waving；同步调整 README 的阶段输出和 Action 通道说明。
- 按官方 Hand Landmarker 文档规划 MediaPipe 接口与版本选择；每个 session 的 `session.json` 增加必填 `target_model`，训练按该字段区分 State 和 Action 数据。
- 明确缺手帧不监督、不作为全零模型输入；连续缺手重置 Action LSTM，连续帧数保留为候选参数，FSM 仍按 300 ms 超时。
- 将采集流程改为屏幕按 session 提示表逐项提示、动作间隔 2 秒；每个 session 用独立文件夹和 `session.json` 保存采集参数，并同步调整 README、DECISIONS 与 AGENTS 的配置边界。
- 将采集、标注、训练、在线推理与 FSM 的流程合并到 README.md，并更新仓库结构规划。
- 删除 docs/ARCHITECTURE.md 和 docs/DATAS.md；更新 AGENTS.md 与 DECISIONS.md 的文档分工和链接。
- 新增 CHANGELOG.md，开始按日期记录改动。
- 补充 README.md 的文件格式、PCA 标注、特征与训练目标、FSM 状态、模块接口和验收标准；同步区分 V1 候选与已定规则。
