# 决策与待确认事项

这里只记录选择及其状态；完整流程和系统结构见 [README.md](../README.md)，实际文件修改见 [CHANGELOG.md](../CHANGELOG.md)。

## 已确定

- 基于摄像头做在线识别；静态手势只看当前帧。一次性动作借鉴 Meta 的事件检测；`waving` 由左右方向原子事件交替组成，经 FSM 上报。
- V1 `waving` 用张开的手，手掌大体朝向摄像头。绕手腕或肘关节的左右摆动都算；向一侧运动后，观察到折返才完成该方向的原子事件。
- 静态手型、一次性动作和 `waving` 分开采集。每个 session 的提示顺序、各动作时长和提示间隔放在该 session 文件夹的 `session.json`，不放在 `config.yaml`。采集时屏幕依次提示用户在预定时段完成指定手势，每项之间间隔 2 秒；提示时段不直接作为动作正标签。用户之后亲自标注实际 `waving` 区间。
- 每个 session 的 `session.json` 必须写明 `target_model: state` 或 `target_model: action`，训练时按此字段选择数据。`kind` 仅说明录制内容；静态手型数据用于 State，一次性动作及 `waving` 数据用于 Action，背景 session 也须明确归属。
- 第一阶段按官方 Hand Landmarker Python 文档实现 Tasks Vision `VIDEO` 接口。MediaPipe 官方 0.10.35 提供本机 macOS arm64 wheel；在 `MultiGestureRecognition` 环境中已加载官方模型并完成单帧推理。GitHub 当前最新 1.0.0 发布说明移除 macOS 预编译包，因此依赖固定在本机已验证可用的 0.10.35。模型文件从 Google 官方 Hand Landmarker 地址下载。
- 项目分四段验收并逐段实现：① MediaPipe + 静态手型 State 分类及输出；② 一次性动作事件模型；③ 一次性动作 FSM；④ 周期性 `waving` 的方向事件与 FSM。用户验收当前阶段后再实施下一阶段。
- 用户会录制若干 session，录完后指定训练集和验证集。训练窗口可以高度重叠；希望部分窗口包含两个动作，且窗口起点不在动作中。
- 在已标注的 `waving` 区间内，对手部位置轨迹做 PCA 并投影到主方向，自动生成左右折返点标签，再由用户调整。
- Action 输入包含归一化掌心坐标及腕部到中指的方向角。未标为正事件的帧，若最近若干帧持续检测到手，则作为 Action 背景（四个事件目标均为零）；丢手附近的帧不按此规则标背景。第一版之后再针对错误采集对抗负样本。
- 缺手帧不参与 State 或 Action 监督，不用全零特征作为正常输入；连续缺手后重置 Action LSTM。`waving` FSM 仍按已确定的 300 ms 等待超时。
- State 的正类手型由用户标注；在 State session 中，未标正类且当前与此前连续一段时间都检测到手的帧自动作为 `NULL` 训练目标。未满足连续有手条件的帧不参与 State 监督。
- FSM 首次收到两个方向相反的原子事件（R→L 或 L→R）即触发 `waving`；触发后等待下一个预期方向事件，超过 300 ms 未出现则退出。

## V1 候选，尚非定值

- MediaPipe 目标采样率约 30 FPS；单帧 State MLP 与因果 Action LSTM 对有效手帧运行。类别、特征和 FSM 职责见 README。
- 录像逐帧提取使用 Hand Landmarker 的 `VIDEO` 模式；实时摄像头采用同步逐帧调用，避免异步模式跳帧影响一帧一结果。当前固定 MediaPipe 0.10.35；Google 官方模型文件下载到项目根目录的 `models/hand_landmarker.task`。当前环境使用 CPU delegate；真实摄像头尚未验证。
- 第二段 Action 暂只训练 `close_hand`、`open_hand` 两个事件通道；第四段加入 `left_to_right`、`right_to_left`，成为最终四通道 Action 模型。四段各自的验证指标数值待真实数据确定。
- Action 事件脉冲暂以动作完成后 `e+3` 至 `e+7` 帧为候选。Meta 的[公开配置](https://github.com/facebookresearch/generic-neuromotor-interface/blob/main/config/data_module/discrete_gestures_data_module.yaml)采用更短的事件脉冲；本项目需按摄像头数据验证。
- 训练窗口暂考虑约 8 秒，以容纳两个动作及间隔。先前的 16 秒和固定 3 秒准备加 5 秒动作机会不作为默认要求。
- 为使第一版可实现，暂以单手、原始图像 x 轴定义左右；局部关键点以 wrist 为中心并按 wrist→middle MCP 距离缩放，腕部角度取 wrist→middle MCP 相对图像 x 轴的角度；PCA 先试五个指尖的中心。以上均为候选，需用实际录像检查。
- Action LSTM 暂按“训练窗口零初始 hidden state、在线同长度滚动窗口重算”；事件解码暂按各类阈值上穿及同类去重；`waving` 首次配对间隔暂限 300 ms，每段只上报一次，同向重复事件忽略。均为实现候选，非用户确认规则。
- README 的 `config.yaml` 示例给出可运行的 V1 候选初值（如高重叠裁窗、3 帧有效手历史、PCA 平滑/幅度阈值和轻量模型）；除用户已确定的规则外，均需在实际标注与验证数据上调整。
- State `NULL` 所需的连续有手帧数暂取 3 帧候选，已由 `config.yaml` 驱动第一阶段标签生成。连续缺手 3 帧作为未来 LSTM 重置阈值的候选；短暂缺手跳过该帧并保留已有历史。FSM 不因该候选阈值提前退出，仍按 300 ms 规则计时。
- 一次性开/合手的起止暂候选为“离开原稳定手型”到“首次清楚达到目标手型”；`thumb_up` 与 `closed` 的首版标注边界、单手采集和原始图像左右方向也是候选，仍待用户确认。

## 待确认

**采集与标注前：**

- 用户标注 State 正类时，`opened`、`thumb_up` 与 `closed` 的视觉边界如何保持一致？未标正类且连续有手的帧已确定自动标 `NULL`；缺手已确定不监督。
- PCA 候选的五指尖中心是否足以覆盖两种摆动？如何平滑轨迹、确定折返最小幅度与标注锚点？普通横向移动出现折返是否也标事件？自动生成的标签需要人工检查和修改。
- `close_hand` / `open_hand` 的起止与完成标准是什么？V1 只跟踪一只手还是两只手？镜像后的左右方向按什么坐标定义？
- Action 背景要求连续多少帧可见手？候选的 middle MCP（9）与图像 x 轴角度是否合适？

**在线推理前：**

- LSTM 的滚动窗口候选与实际在线成本是否合适？漏检或换手时如何处理？
- `waving` 已确定由两个相反方向事件触发、等待下个事件超时 300 ms 后退出；持续期间是否重复上报，以及退出后的重触发接口仍待定。

**由数据或交付目标确定：**

- 事件脉冲、窗口长度、特征归一化、模型大小、阈值和冷却时间；验证时的事件容差、误报与延迟目标。
- 各 session 的具体动作时长、提示顺序和训练/验证归属由用户录制前后分别指定；部署目标尚未指定。
