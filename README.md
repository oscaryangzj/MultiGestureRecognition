# MultiGestureRecognition

基于摄像头关键点识别静态手型与动作。当前在线 State 使用 Google 官方 Gesture Recognizer；第二阶段已实现开手/握拳的正负例采集、人工审核、30 Hz 重采样、54 维特征、CNN + 三层 LSTM 训练、录像评估与摄像头在线推理，五个三维手指角模型已训练。下一轮按一次性事件解码/FSM、waving 数据与四通道模型、waving FSM 分步验收，具体范围见文末规划。

一次性动作参考 Meta 的 [generic-neuromotor-interface](https://github.com/facebookresearch/generic-neuromotor-interface) 和[论文](https://www.nature.com/articles/s41586-025-09255-w)：学习逐帧事件脉冲和独立事件通道；摄像头特征与 sEMG 的模态细节分别处理。

协作规则见 [AGENTS.md](AGENTS.md)，选择与待确认项见 [DECISIONS.md](docs/DECISIONS.md)，实际修改见 [CHANGELOG.md](CHANGELOG.md)。

## 安装与 State 在线输出

在项目根目录操作；已有环境时直接激活。

```bash
conda create -n MultiGestureRecognition python=3.11 pip
conda activate MultiGestureRecognition
python -m pip install -r requirements.txt
python -m scripts.download_hand_landmarker
python -m scripts.download_gesture_recognizer
python -m scripts.run_state
```

依赖固定在 `requirements.txt`。MediaPipe 0.10.35 已在本机验证，使用 Tasks Vision 接口。官方模型下载到 `models/`；训练模型与结果写入 `runs/`。MediaPipe 当前用 CPU delegate 推理，模型训练用 PyTorch MPS GPU。

`run_state` 保留摄像头和 `--session-id SESSION_ID` 回放入口，输出 `closed/opened/thumb_up` 三路分数与 21 点 `[21,2]` 图像归一化坐标。官方类别映射由 `gesture_recognizer.category_names` 配置；三个单类别识别器取得各目标的真实分数，不把三类重新归一化。缺手返回缺失。超过 `inference.state_threshold` 的类别显示出来，全低时显示 NONE。

点击 START REC / STOP REC 或按 `r` 控制可选录制，按 `q` 退出；视频包含关键点和分数，保存到 `inference.recordings_dir`。无需先训练本项目的 State MLP。

## Action：采集到回放

### 1. 正负例分别采集

```bash
python -m scripts.collect --plan action
```

终端依次询问采集者姓名、采集正例或负例，以及开合组数或本次负例任务。默认选择正例。录制前显示操作说明，再进入摄像头预览；点击 START 后才创建 session、开始录像和计时。原始画面用于保存，屏幕上的提示只引导被试。录制时按 `q` 可提前结束。

#### 正例：开合一次，随后保持

一组包含握拳一次、张开一次，按两类自然交替，不分别输入类别次数。初始张开手掌，默认流程：

```text
张开手掌 → 点击 START → 保持张开 2 秒
握拳一次，完成后保持 → 保持拳头，随机等待 0.8–2.5 秒
张开一次，完成后保持 → 保持张开，随机等待 0.8–2.5 秒
重复 N 组；最后的保持结束后停止
```

每次动作提示默认 3 秒，被试自然完成一次，剩余时间保持目标手型；HOLD 阶段继续保持刚完成的手型，等待相反动作的提示，不需要额外恢复姿势。随机保持时长避免所有动作形成相同节奏。屏幕显示动作指令、目标手型示意、当前阶段颜色、倒计时和提示进度。

预览窗口可点击 PRACTICE，按同一流程练习默认两组，不保存练习录像或 session；回到预览后再点 START 正式录制。正式默认 10 组、20 次动作，动作前准备保持 2 秒。用另一只手点击按钮。各次保持时长独立随机抽取，开始前按抽取结果打印计划总时长。

#### 负例：按明确任务活动

开始前从以下五项中选择本次任务；每个新 session 只采这一项，准备 3 秒、任务 30 秒。START 预览显示选中任务的中英说明，实际选择和时长保存到本 session。历史多任务录像按自己的 session 计划处理：

| 任务 | 被试指令 |
| --- | --- |
| 张开时移动 | 保持张开，缓慢移动手的位置或转腕 |
| 握拳时移动 | 保持拳头，缓慢移动手的位置或转腕 |
| 局部手指活动 | 小幅活动单根手指，避免完整张开与完整握拳之间的切换 |
| 三指伸出再握拳 | 握拳→伸出中指、无名指、小指→握拳，反复做；拇指、食指保持弯曲 |
| 张开手腕上下摆 | 五指保持伸直，手腕上下摆，反复做 |

每项任务之前有 3 秒 PREPARE，允许调整为该任务需要的姿势。**PREPARE 全部 IGNORE**，不把调整姿势时的开合当作负标签。真正任务范围必须回看并人工确认，才对有效背景监督 `[0,0]`。意外发生完整开合时，在负例标注界面记录真实动作，不能硬标为全零。

训练和标准验证将每个负任务作为独立输入片段：PREPARE 不参与插值、运动特征计算或模型历史；任务边界及人工排除区间切断历史。任务开始后前两帧仍作为输入，达到连续可见帧数后才监督背景。新 session 暂不单独采集握拳静止和张开静止；`prepare_pose` 中的握拳、张开只用于准备姿势。

采集模板统一在 `session_plans/action.json` 的 `positive`、`negative` 两节。选中的模板复制到本 session 的 `session.json`，并记录 `sample_type: positive/negative`、`kind: one_shot/background`、准备/动作/保持时长、组数或任务轮数与展开的实际提示。正例用 `pairs`、`initial_pose`、`practice_pairs`、`prepare_seconds`、`hold_seconds_range` 和 `durations_seconds` 配置；每条实际提示记录 `duration_seconds`、抽取的 `hold_seconds`。负例用 `repetitions`、`prepare_seconds` 和 `tasks` 配置，每项保存指令、准备手型及 `duration_seconds`。录像结束后补充实际提示起止帧。采集参数属于各 session，不放在全局配置中。

### 2. 提取关键点与官方分数

把下列 `SESSION_ID` 替换为采集器打印的 ID；采集结束也会直接打印可复制的命令。

```bash
python -m scripts.extract_action --session-id SESSION_ID
python -m scripts.annotate_action --session-id SESSION_ID
```

`extract_action` 对每个录像帧同步调用官方识别器，同时保存 21 点二维坐标、21 点三维 world 坐标和 State 三路分数，保证数据来自同帧。缺手行的坐标和分数留空。Action 使用其中 `opened/closed` 两路软分数。已有 Action session 在训练前需重新运行提取命令以生成 `world_landmarks.csv`。

使用当前角度特征训练时，对 `data/splits.json` 中每个训练和验证 Action session 都运行一次 `extract_action`；标注文件及数据划分不需要重做。

### 3. 正负例审核与标注

采集正例和负例使用同一个 `annotate_action` 入口，界面按 `sample_type` 提供相应操作。

#### 正例：确认自动候选

打开标注器后，按每次动作提示自动生成起点/完成点候选。类别直接沿用提示。

候选先寻找稳定的原手型，取提示前的局部关键点作为参考；平均关键点形变达到阈值且持续若干帧时提出起点。目标手型分数连续满足阈值时，以这段稳定序列的首帧提出完成点。候选可以延伸到本次提示后的 HOLD。缺少可靠的起始手型、目标手型或发生历史重置时，保留空候选，用户可手动设置。稳定确认可使用后续帧，这是离线标注。

这些候选仅供人工审阅。未确认候选不进入 `action_intervals`，也不作为正标签。

| 操作 | 按钮 / 快捷键 |
| --- | --- |
| 查看候选 | 默认跳到当前候选起点附近；`P` / 空格播放，到当前动作机会末尾暂停 |
| 确认 | `CONFIRM` / Enter，保存为真值并自动跳到下一个待处理动作 |
| 调整起点 | `J` 跳到候选起点，用 `A/D` 逐帧调整，`I` 设置新起点 |
| 调整完成点 | `K` 跳到候选完成点，用 `A/D` 调整，`O` 设置新完成点 |
| 取消整段 | `CANCEL` / `X`，将本次动作机会及后续 HOLD 设为 IGNORE，跳到下一项 |
| 切换动作 | `[` / `]`，或点击整段录像的提示时间轴 |
| 局部定位 | 点击下方局部时间轴，定位当前动作附近的帧 |
| 保存退出 | `S` 保存；`Q` 或关闭窗口退出时自动保存 |

上方提示时间轴显示整段录像；下方放大当前动作附近的确认区间、黄色候选、标签脉冲与监督掩码。调整已确认的边界会使该动作重新待确认。取消后可返回该动作，重新设置边界并确认。

完成所有提示的确认或取消后，`action_reviewed` 自动置为 true。待确认和取消的机会全部忽略，取消范围包含提示前的回看范围及机会末尾之后可能出现的延迟标签。取消区间不补背景，不计为评估误报。每个机会应只有一次指定动作；若出现额外开合或动作不明确，可取消整段。

#### 负例：确认背景，意外动作如实标注

逐个任务播放检查。确认整项任务中没有遗漏的目标事件后，按 Enter 确认该任务；有效有手的背景帧进入 `[0,0]` 监督。按 `X` 可取消整个任务，使其全部 IGNORE，随后自动切换到下一个待处理任务。

如果任务中意外发生完整开合：

1. 用 `1/2` 选择 `open_hand/close_hand`。
2. 逐帧定位，以 `I` 设置开始离开原手型的起点，`O` 设置首次完整达到目标手型的完成点。
3. 按 `E` 保存真实事件。同一任务内可以记录多个事件；Backspace 删除覆盖当前帧的事件后可重标。
4. 标完本任务全部真实事件，再按 Enter 审核整项任务。事件周围的可靠背景仍为全零，事件完成后按同样规则生成正脉冲。

正例类别取提示，负例中的真实事件才需要手动选类。两种模式都按提示逐项确认或取消，不再用整段录像的 REVIEWED 开关。未确认和取消的任务不作为负样本；所有任务处理完后才允许该 session 进入训练。

### 4. 特征与逐帧目标

原始视频、帧时间和人工标注保留不变。模型使用 `data.action_sample_rate_hz`（当前 30 Hz）的统一时间轴：对相邻有效观测的二维/三维 21 点与 State 分数线性插值，再计算下述 54 维特征。在线等右侧观测到达后才生成对应采样点，离线使用相同规则；不跨缺手插值，插值区间接触 IGNORE 时不获得监督。15 FPS 数据插值为 30 Hz 只统一时间步，不增加原始动作细节。

| Action 每帧输入 | 维度 | 定义 |
| --- | ---: | --- |
| 21 点局部 x/y | 42 | 修正宽高比，减去 wrist，按 wrist→middle MCP 缩放并对齐掌轴，再统一左右手局部方向 |
| 掌心速度 | 2 | 相邻有效位置差除以实际观测时间差 |
| 掌心加速度 | 2 | 相邻速度差除以实际观测时间差 |
| 腕部角速度 | 1 | 相邻 wrist→middle MCP 方向角的最短角差除以实际时间差 |
| 官方 State 分数 | 2 | 按 `features.action_state_classes` 取 opened、closed |
| 五个手指弯曲角 | 5 | 四指 PIP 与拇指 MCP，各用 MediaPipe 三维 world landmarks 计算 |
| **合计** | **54** | 保留原 49 维输入和 21 点局部坐标 |

特征依次排列为上述表格顺序。掌心取 wrist 与四个 MCP（0、5、9、13、17）的图像归一化坐标中心，只用于计算速度和加速度，绝对位置不作为模型输入。速度、加速度和腕部角速度保留相机方向；五个手指角使用三维向量夹角，不增加角度变化量。首次有效观测的速度、加速度、角速度为 0；形成两次观测后计算速度，形成两次速度后计算加速度。短暂缺手跳过输入并保留历史，恢复后按实际时间差计算；连续缺手达到 `data.no_hand_reset_frames` 后，运动差分与 LSTM 历史一起清空。缺手的全零数组只作存储占位或批次 padding，不喂作正常样本。

两个事件为 `open_hand`（握拳→完整张开）与 `close_hand`（完整张开→握拳）。人工确认首次完整达到目标手型的完成帧 e，取其原始时间 t；当前在 `[t+0.1s,t+0.25s)` 生成短正脉冲。延迟和持续时间分别由 `labels.action_positive_delay_seconds`、`labels.action_positive_duration_seconds` 配置，不随录制帧率变化。Action 没有独立 NULL 通道。输出顺序为 `[open_hand,close_hand]`。

以一次已确认的 `close_hand` 为例：

| 帧范围 | 目标 | 是否监督 |
| --- | --- | --- |
| 动作前稳定张开、握拳过程中、完成帧 e，以及等待正脉冲的帧 | `[0,0]` | 最近连续若干帧有效有手时监督 |
| 完成后 100 ms 至 250 ms（不含末端） | `[0,1]` | 有手，且动作历史未因连续缺手重置时监督 |
| 正脉冲结束后继续保持拳头 | `[0,0]` | 最近连续若干帧有效有手时监督 |
| 缺手、取消、待确认区间 | 无 | IGNORE，不计算损失 |

`open_hand` 对应的正脉冲是 `[1,0]`。背景当前要求连续 3 帧有效有手，初次观测和缺手恢复后尚未满足条件的背景帧 IGNORE；该参数由 `data.action_background_visible_frames` 配置。每个事件通道独立生成目标；极近事件的脉冲重叠时，分别监督对应通道为 1。

起点用于避开动作中间裁窗并保留动作历史；完成点用于生成正脉冲。保持手型和动作过程都不是持续的正标签。可靠背景与正脉冲共同训练模型，让一次动作产生一个短事件。

#### 三维逐指弯曲角

每根手指增加一个几何角，共 5 个角度，为模型提供手指开合的直接证据。保留掌心速度、加速度、腕部角速度及官方 State 分数；开合与手整体运动仍由同一个时序模型学习。特征提取和在线/离线接口已实现，54 维模型已训练并保存到 `runs/action_finger_angles/`。当前验证集的 40 个动作事件全部命中、无误报，平均延迟约 0.175 秒；验证仅覆盖 5 个 session，其中正样本 session 为 2 个，需通过后续真实在线使用验收泛化效果。

使用 MediaPipe 的 21 个 `hand_world_landmarks` 三维坐标计算角度。官方 Gesture Recognizer 已返回这组数据，可从与当前二维点相同的识别结果、同一只手中取得，无需为此再运行一个 Hand Landmarker。world 坐标的三个轴使用相同单位；图像归一化 x/y 和相对深度 z 的尺度不能直接混用。[官方输出说明](https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer/python#handle_and_display_results)

数据流程：

```text
已有 video.mp4 + frames.csv → 补提取同帧 21 点 world x/y/z
        → 按现有输入片段边界重采样 → 5 个逐指角度
        → 拼接现有 Action 特征 → 原 CNN + LSTM → 原事件评估与在线输出
```

`extract_action` 在原有文件之外保存 `world_landmarks.csv`，每行含 `frame_index,elapsed_seconds,hand_detected,x0,y0,z0,...,x20,y20,z20`。已有录像可重新提取，无需重新采集或标注。三维坐标只用于手指几何，掌心运动继续从图像坐标计算；world 坐标以手中心为原点，不能直接用于掌心绝对运动。

每根手指只取一个代表角，共 5 个角度。首轮关节选择如下，输入按食指、中指、无名指、小指、拇指排列：

| 手指 | 代表关节（首轮候选） | 三维点编号 |
| --- | --- | --- |
| 食指 | PIP，中间关节 | (5,6,7) |
| 中指 | PIP，中间关节 | (9,10,11) |
| 无名指 | PIP，中间关节 | (13,14,15) |
| 小指 | PIP，中间关节 | (17,18,19) |
| 拇指 | MCP，掌指关节 | (1,2,3) |

每组三点 `(a,b,c)` 取 `u=p_b-p_a`、`v=p_c-p_b`，定义 `theta=atan2(||u×v||,u·v)`，单位为弧度。两段同向伸直时接近 0，偏折时增大。一个代表角用于提供该手指开合的线索，不完整描述所有关节或拇指对掌位置；代表关节是否合适通过真实开合和负例片段验证。

刚体平移、旋转、统一缩放及左右镜像不改变理想三维夹角；MediaPipe 的估计仍可能受遮挡、模糊和朝向影响，需用真实片段检查。保留逐指的 5 个角，不先压缩为一个全手开合分数，也不将角度当作 State 概率或动作上报的硬门槛。

在重采样时间轴上直接计算每帧的 5 个角度，不增加手指角度变化量、角速度或角加速度。开合的时间变化由 CNN + LSTM 从角度序列学习。三维重采样沿用现有有效帧与输入片段边界，不跨准备期、缺手或人工排除区间补点。

在原输入末尾追加 5 个角度，共新增 5 维：含加速度为 54 维，不含加速度的对照为 52 维。三维点与现有数据采用相同的有效帧及片段边界，缺失三维点不补成有效的全零角度。角度按现有规则使用训练集统计量归一化；开关与关节组合配置在 `config.yaml`，并随 checkpoint 保存，线上线下共用同一计算流程。角度模型和指标保存在 `runs/action_finger_angles/`，原 49 维模型仍保存在 `runs/action/`。

验收先回看真实开合、三指伸出和张开手腕上下摆片段，确认角度曲线能表达开合且在整体摆动时相对稳定；训练时使用相同 session 划分、设置及监督，对比原 49 维模型与新增角度的 54 维模型。两组使用相同可评估事件及有效时长，同时记录三维缺失造成的数据覆盖变化；报告原方向与镜像方向的逐类召回、误报、每分钟误报和延迟。是否采用及是否调整代表关节由验证结果决定。当前不实现 waving。

### 5. 划分与训练

录完并确认若干 session 后，由用户指定 `data/splits.json` 的 train / val 列表。正例与负例 session 放在相同的列表中，按整个 session 隔离，不能在 train 和 val 重复。State 与 Action 可以共用划分文件，各训练入口按 `target_model` 筛选。

```json
{"train": ["训练SESSION_ID"], "val": ["验证SESSION_ID"]}
```

先按 session 划分并重采样，再以 `data.action_window_seconds` 的窗口和 `data.action_stride_frames` 的步长裁窗。当前为 8 秒、重采样后 5 帧（约 167 ms）；窗口起点避开动作和忽略范围。短缺手帧从输入序列中跳过，连续缺手后划为新历史；短片段可作为较短序列并在批次末尾 padding。若窗口截掉动作起点却保留该动作的延迟脉冲，屏蔽对应通道的脉冲损失。

负例先根据实际任务范围和人工排除区间划分输入片段，再在片段内部重采样和计算特征，最后裁窗。沿用 session 的 30 Hz 时间网格，边界两侧的观测不能共同插值；首次有效观测的速度、加速度和角速度为零。窗口完全落在一个任务片段内，特征在片段内连续计算，重叠窗口不各自重置差分。每个窗口从零 LSTM 状态开始。理想的 15 秒、450 个采样点可裁出 43 个 240 帧窗口；缺手、人工排除和动作起点限制会减少数量。不要求每个窗口包含正事件。

训练与验证使用相同的特征公式。归一化的均值、标准差只从训练集中有监督且有效的重采样特征帧计算，IGNORE 不参与统计；正例权重也只从训练集计算。正负 session 的窗口共同随机打乱，没有额外的固定正负采样比例。

Action 架构参考 Meta 的离散手势小模型：`Conv1d → ReLU → Dropout → LayerNorm → 三层单向 LSTM → LayerNorm → Linear(2)`。当前卷积输出 128 维、kernel=3、stride=1，LSTM hidden=128、3 层，卷积后及 LSTM 层间 dropout=0.1；卷积只在左侧 padding，输出与当前采样点对齐，接口为 `[B,T,54] → [B,T,2]` logits。Meta 的 sEMG 卷积 kernel=21、stride=10 按摄像头时间尺度调整，不沿用其 2 kHz 的降采样参数。模型参数统一配置在 `models`。

本机 PyTorch 2.7.1 的 MPS 多层 LSTM 内置 dropout 存在训练/评估缓存问题，因此使用三个单层 LSTM 并在层间显式调用 Dropout，保持相同的三层结构与正则化语义。见 [PyTorch 问题记录](https://github.com/pytorch/pytorch/issues/180744)。

带逐帧通道掩码的 BCEWithLogitsLoss 在 MPS 上更新。Adam 和梯度裁剪沿用现有设置，学习率参考 Meta 小模型采用 5 轮线性 warmup（1e-6 至 1e-3），每 25 轮乘 0.5；对应参数位于 `training`。模型结构、采样率和监督时间参数随 checkpoint 保存，推理从 checkpoint 恢复。

每个 epoch 除窗口验证损失外，还对完整验证 session 推理并计算事件指标。按两类事件的 macro F1 最大选择 checkpoint；并列时依次选每分钟误报更低、平均绝对延迟更小、未加权 BCE 更低的模型。评价包含负例 session，在负例录像中产生的事件会计为误报。

```bash
python -m scripts.train_action
```

纯负例 session 可以没有动作事件，但 train、val 各自的全部 Action 数据合起来都需要覆盖两个有监督事件通道，裁窗后仍需对应正标签；每个 session 的提示都要确认或取消。54 维角度模型及结果保存到 `inference.action_finger_angles_output_dir`（默认 `runs/action_finger_angles/`）；49 维旧模型保留在 `inference.action_output_dir`（默认 `runs/action/`）。`metrics.json` 记录正负 session 数、有监督秒数、各类真实事件及正/负监督帧数、每轮事件汇总与选中模型的完整验证结果。54 维模型已完成离线验证，真实在线使用与正式事件解码仍需验收。

### 6. 完整 session 评估与回放

```bash
python -m scripts.evaluate_action
python -m scripts.evaluate_action --session-id SESSION_ID
python -m scripts.view_action --session-id SESSION_ID

# 连续录像对照：保留准备阶段的输入历史，监督范围仍按人工审核
python -m scripts.evaluate_action --session-id SESSION_ID --continuous
python -m scripts.view_action --session-id SESSION_ID --continuous
```

默认评估 Action 验证集，也可指定单个完整 session。每个有效帧从零 hidden state 重算最近同长度窗口，连续缺手时清空窗口，与训练的上下文方式保持一致。

负例的标准评估按任务片段清空特征及模型历史。`--continuous` 生成整条录像连续推理的对照，结果单独写入 `runs/action/evaluation/continuous/`，用于检查实际在线使用时前序动作的影响；两种结果分别报告。摄像头及 `run_action --session-id` 继续按真实连续输入推理，不读取任务边界。

评估为每类超过阈值的连续分数区间取一个峰，与原始完成时间按容差一对一匹配，报告 precision、recall、F1、漏检、误报、每分钟误报与延迟。延迟使用采样点实际可用的观测时间，包含插值等待。缺手、人工取消和待确认范围不参加误报率的评估时长；取消和待确认范围也不参加事件匹配。已确认但整个延迟正脉冲都没有可监督帧的事件不计为可评估真值，例如脉冲落入准备期或历史已因缺手重置。负例 session 的有效背景同样评估误报。峰值匹配仅用于第二阶段评估；产品的阈值解码、去重和一次性动作 FSM 在第三阶段实现。

结果位于 `runs/action/evaluation/SESSION_ID/`，包括 `predictions.csv` 与评估 JSON；预测按当时已可用的最新采样点映射回原始视频帧，回放入口保持不变，支持逐帧查看、播放、时间轴定位，显示目标、监督掩码和两条预测分数。

### 7. 在线 Action 推理

```bash
python -m scripts.run_action
python -m scripts.run_action --session-id SESSION_ID
```

默认打开 `data.camera_index` 指定的摄像头；`--session-id` 用同一在线流程逐帧处理已有录像，不读取缓存预测。使用已训练 Action 权重和归一化统计，实时提取官方 21 点二维/三维坐标及 State 分数，按 checkpoint 的采样率生成相同特征，从零 hidden state 重算最近 8 秒有效观测并输出最新概率。每个新采样点参与阈值计数；相机没有新采样点时显示上次分数。短缺手跳过，连续缺手清空特征、模型上下文及阈值检测历史。相机预览按 `display.mirror_preview` 镜像，模型仍使用原始画面。

界面显示两类概率、官方 opened/closed 分数、检测计数和最近一次检测；下方画最近一个模型窗口的概率曲线及阈值线。按 `Q` 退出。事件计数参考 Meta [`detect_gesture_events` / `debounce_events`](https://github.com/facebookresearch/generic-neuromotor-interface/blob/main/generic_neuromotor_interface/cler.py)：独立概率从阈值下方上穿时计一次，持续高分不重复计数，再用短间隔去抖。阈值沿用 `inference.action_threshold`（0.5），去抖初值为 `inference.action_debounce_seconds`（0.05 秒）。这些是在线试用的检测结果；第三阶段再验收一次性动作 FSM 与对外事件接口。

## State 数据与自训练 MLP

保留第一阶段的数据流程，供自训练模型对照使用：

```bash
python -m scripts.collect --plan state
python -m scripts.extract_landmarks --session-id SESSION_ID
python -m scripts.annotate_state --session-id SESSION_ID
python -m scripts.train_state
```

State 仍分别询问 opened、closed、thumb_up 的组数，随机顺序，每次提示 3 秒、REST 2 秒。标注只确认稳定起点，类别与结束帧取 prompt；`I` 设置起点，`X` 取消整段，`A/D` 逐帧，`P` 播放，`S` 保存。REST、缺手、未确认或取消的帧全部 IGNORE。

State MLP 保持 42 维局部关键点输入，三个独立 Sigmoid + BCE 输出；权重保存在 `runs/state/model.pt`。当前 `run_state` 使用官方模型，自训练 MLP 的训练结果作为对照保留。

## 数据文件与配置

```text
data/sessions/SESSION_ID/
  session.json       # 采集者、target_model、sample_type、kind、实际提示、组数/轮数及时长
  video.mp4          # 原始录像
  frames.csv         # frame_index,elapsed_seconds
  landmarks.csv      # 同帧 21 点 x/y、hand_detected
  world_landmarks.csv # Action 提取：同帧 21 点 x/y/z、hand_detected
  state_scores.csv   # Action 提取时生成：frame_index,closed,opened,thumb_up
  annotations.json   # 人工确认、待确认候选与取消范围
data/splits.json     # train / val 的 session ID
```

所有帧号从 0 开始，区间两端都包含。Action 正例标注字段示例：

```json
{
  "session_id": "SESSION_ID",
  "action_review_mode": "prompts",
  "action_reviewed": false,
  "action_intervals": [{"prompt_index": 0, "start_frame": 70, "end_frame": 90, "label": "close_hand"}],
  "action_candidates": [{"prompt_index": 1, "start_frame": 210, "end_frame": 232, "label": "open_hand"}],
  "action_background_prompts": [],
  "action_ignored_intervals": []
}
```

`action_intervals` 是真实事件，`action_candidates` 是正例待确认草稿，边界可以为 null；取消范围单独保存在 `action_ignored_intervals`。重新打开不会覆盖人工调整。负例用 `action_background_prompts` 保存已经审核的任务索引，例如：

```json
{
  "session_id": "SESSION_ID",
  "action_review_mode": "prompts",
  "action_reviewed": false,
  "action_intervals": [{"prompt_index": 1, "start_frame": 900, "end_frame": 913, "label": "close_hand"}],
  "action_candidates": [],
  "action_background_prompts": [0, 1],
  "action_ignored_intervals": []
}
```

上例中任务 0 没有真实事件，任务 1 意外发生了一次完整握拳；两个任务都已审核。任务 1 的正脉冲监督握拳，其余有效背景监督全零。其余尚未审核的任务仍 IGNORE，整个 session 未完成审核，因此 `action_reviewed` 为 false。

`config.yaml` 管理共享的数据处理、候选阈值、特征、模型、训练及评估参数。每个 session 的正负类型、顺序、组数/轮数、练习、初始手型、准备/动作/保持时长来自采集模板并复制进本 session，候选阈值由 `annotation` 节配置。训练统计量随模型保存。

## 下一轮：一次性 FSM 与 waving 的实现、验收规划

```text
一次性正例 / waving 正例 / 动态负例采集
        → 同帧二维与三维关键点、State 分数
        → 人工动作区间 / waving 区间 → PCA 折返草稿 → 人工修订
        → 按 session 划分 → 重采样 / 54 维特征 / 四通道脉冲与掩码
        → Action CNN + LSTM → 因果事件解码
        → 一次性事件 / waving FSM → 最终事件与 waving 活跃状态
```

| 阶段 | 当前状态 / 验收内容 |
| --- | --- |
| 1. MediaPipe + State | 官方三路 State 与 21 点在线输出已接入，用户已实际试用 |
| 2. 一次性 Action | 54 维模型已训练，小验证集 40/40、0 误报；仍需真实在线使用验收 |
| 3. 一次性 FSM | 交付 A：统一因果事件解码、去重、复位及录像/摄像头输出 |
| 4. 周期性 waving | 交付 B：waving 采集、区间与折返标注、四通道训练；交付 C：方向事件组合 FSM 与最终动作验收 |

交付 A/B/C 是原第三、四阶段的验收拆分。当前第二阶段完成在线验收后，按 A → B → C 实现，每个交付由用户验收后继续。以下是文档规划；候选算法、参数初值和验收数值目标集中在 [DECISIONS.md](docs/DECISIONS.md)。

### 交付 A：一次性事件解码与 FSM

目的：一次完整开手/握拳只上报一次；持续高概率、阈值附近抖动、缺手与重新观测的行为能够逐帧解释。

参考 Meta [事件检测代码](https://github.com/facebookresearch/generic-neuromotor-interface/blob/main/generic_neuromotor_interface/cler.py)的阈值上穿及短去抖，再定义本项目的重新允许触发与冲突处理。论文中的 press/release FSM 服务于按住和释放语义，本项目开合手与 waving 的 FSM 按本项目动作定义设计。[论文在线解码说明](https://www.nature.com/articles/s41586-025-09255-w)

同一解码器供录像回放、完整 session 评估和摄像头推理使用，每个新模型采样点处理一次。输入为类名、独立 Sigmoid 分数及当前经过时间；输出为原子事件（类别、触发时间、触发分数）。各通道记录是否可触发及上次分数，具体复位、滞回与去抖规则见 DECISIONS 的候选方案。

正式评估新增在线解码事件的一对一匹配。原离线峰值指标仍显示为模型诊断；最终上报的延迟采用实际触发时间，计入插值等待和解码延迟。连续录像推理遵循真实输入历史，PREPARE 的监督掩码只限制计分范围，不能替代在线输入历史。

验收依次进行：

1. 给定人工分数序列，覆盖一次上穿、长时间高分、短低谷、再次合法动作、同帧冲突、快速开合、短缺手、连续缺手和恢复，核对事件数和时间。
2. 对当前完整验证录像同时保存原始分数、解码候选与最终事件，逐个核对 40 个真值动作和背景误报，比较离线峰值与因果解码结果。
3. 补充独立在线录像，覆盖左右手、不同速度/朝向及已知动态负例。记录逐类漏检、重复上报、误报/分钟、平均及 P95 延迟。

交付时应能在同一时间轴上查看模型分数、人工完成帧和实际触发帧，定位错误来自模型分数还是解码规则。

### 交付 B：waving 采集、标注与四通道模型

waving 使用张开的手，掌心大体朝摄像头，腕部或肘部左右摆动均可。方向原子事件在向一侧运动后，观察到向相反方向运动时完成。类别按折返前的方向命名为 `left_to_right` 或 `right_to_left`；一次折返对应一个事件。

计划为 `collect --plan periodic` 加入与现有采集一致的姓名输入、预览、START 和简洁中英提示。录制时提示在一小段时间内连续 waving，再停止并休息；每段可含多个周期。单段时长、段数、速度/手侧/腕部或肘部提示、准备和休息时长由采集模板生成并写入本 session 的 `session.json`。正负例分开录制，训练/验证归属仍由用户指定。

先标真实 waving 区间，再自动生成方向草稿：

```text
真实 waving 区间 → 宽高比修正后的代表点轨迹 → PCA 主方向
        → 主方向投影与因果平滑 → 位移/反向运动确认
        → 方向事件草稿 → 人工移动、删除、补充、确认
```

PCA 使用整段已标区间确定离线标注主轴；线上只使用训练模型输出方向事件。主轴符号按原始图像 x 轴正向固定，镜像预览在显示层转换左右说明。物理极值帧和观察到反向运动的确认帧分别展示，事件完成帧取后者，避免将尚未观察到的折返提前标成完成。

上下摆动也能产生 PCA 主轴和极值，因此自动候选必须检查主轴是否接近左右方向；幅度不足、方向不明确和缺手边界交给人工修订。连续缺手两侧不能拼成一次折返。起点保留该次移动的历史，用于避免从动作中间裁窗；结束时仅停住而没有观察到折返的最后一段不生成方向正事件。

标注工具复用 Action 的逐帧、播放和时间轴操作，在同一视图增加 waving 区间、投影曲线、折返候选及方向标签。`annotations.json` 计划增加 `waving_intervals` 表示真实 waving 段，并用独立字段保存已审核的方向监督范围，字段建议见 DECISIONS；确认后的方向动作写入原 `action_intervals`，沿用 `start_frame/end_frame/label`，其中 `end_frame` 是折返确认帧。草稿经人工确认后才参加训练；未确认和取消范围 IGNORE。

模型输入首轮沿用 54 维特征与现有 CNN + 三层 LSTM，输出扩至四个独立通道：`open_hand/close_hand/left_to_right/right_to_left`。方向事件沿用完成点后短正脉冲，时间参数由配置控制。waving 活跃区间是组合动作真值，不能把整段 waving 作为持续为 1 的方向标签。

镜像增强时，相机方向的运动特征随镜像变换，两个方向通道的目标、监督掩码和验证真值同步交换；开合通道保持原语义。左右手局部坐标归一化与图像镜像增强分别处理，方向标签始终以原始相机坐标定义。

**扩展数据时按通道审核监督。** 旧 session 原有开合标签和掩码保留；未经 waving 审核的旧数据，其两个新方向通道应为 IGNORE。尤其旧“张开手移动/转腕”负例可能含真正 waving，不能直接给新增通道填零。人工审核后的方向背景才监督全零；新 waving 录像中意外发生的真实开合动作同样如实标注。

训练集和验证集均覆盖左右手、腕部与肘部摆动、不同速度及动态负例。单段 waving 提示可以短于模型窗口；训练仍从完整 session 的有效历史裁窗，不能把每段短提示当成一条必须填满 8 秒的独立样本。负例任务边界、缺手和人工取消继续遵循已有的输入切段规则。四通道模型单独保存，复用原开合验证录像检查是否退化。先检查标注投影和方向事件，再做训练及完整 session 回放；详细首轮数据量建议见 DECISIONS。

交付 B 验收：方向草稿可逐帧调整，真值及监督掩码正确；train/val 按 session 隔离；报告四类原子事件的 precision/recall、误报/分钟与延迟，同时展示 waving 上下摆干扰、左右手及开合回归结果。

### 交付 C：waving FSM 与最终动作

FSM 消费已解码的两个方向事件；首次收到两个相反方向事件就进入活跃并触发 waving。继续收到预期方向事件时交换预期方向并续期；300 ms 未收到下一预期事件退出。这里“两次”指两个相反的原子事件，开始可以是任意方向。

| waving 状态 | 输入与转换 |
| --- | --- |
| 空闲 | 收到第一个方向事件，记住方向并等待相反方向 |
| 等待第二个事件 | 收到相反方向事件后触发；首次配对期限与同方向输入规则见 DECISIONS 候选 |
| 活跃 | 收到预期方向事件，交换预期方向并重新计时；续期不再触发新的开始事件 |
| 活跃 | 300 ms 没有预期事件，退出并回到空闲；缺手或无新分数时计时继续 |

先用人工事件与确定时间线核对两种起始方向、交替续期、同方向重复、超时、停止后重触发、缺手及超时边界；再接模型事件逐帧回放。每帧都推进 FSM 的时间，避免仅在新事件到达时才发现超时。

300 ms 限制的是相邻折返事件的到达间隔：稳定节奏下相当于完整周期不超过约 600 ms，即约 1.67 Hz 以上。慢 waving 可能不断退出；因此单独展示真实半周期和模型事件间隔的分布，核对当前规则能覆盖的速度，并由用户决定是否调整。首次配对超时单独配置。

最终验收按三层分别报告：模型方向脉冲 → 解码原子事件 → waving 组合输出。waving 每段只匹配一个首次触发，额外开始事件计重复误报；另报告活跃覆盖率、段内错误退出/重新进入次数、停止后的退出延迟及背景误报/分钟。首次触发延迟从人工确认的第二个相反方向事件计起，同时记录从 waving 区间起点到触发的总时间。

最终演示包含左右手、腕部/肘部、快慢 waving、突然停止、单次侧移、上下摆、三指活动、手进入/离开画面及开合动作与 waving 连续切换。普通张开手的往返平移是否计为 waving 仍需确认，定义确定后才能作为正例或负例。

### 共同接口、产物及配置范围

录像、评估与摄像头共用 `scores → 原子事件 → 最终动作/活跃状态`。输出事件字段、每段上报策略及同帧冲突处理的建议见 DECISIONS。界面展示动态数量的概率通道、最近原子/最终事件、waving 活跃状态及预期方向。

验收保存三类产物：`predictions.csv`（逐帧分数、真值/掩码及 FSM 状态）、`events.csv`（候选、最终事件与触发原因）、`evaluation.json`（三层指标及逐条错误）。可以从每个漏报、误报和错误退出定位到原录像帧。

实现时将每类触发/复位阈值、去抖、PCA 代表点/平滑/最小幅度/反向确认/主轴倾角、方向脉冲、首次配对期限、活跃超时及模型输出目录放入 `config.yaml`。采集段数、提示时长、休息及顺序保存到各 session。验收参数选定后冻结，再跑独立录像；配置和候选初值在实施时落地。

## 仓库结构

```text
AGENTS.md / README.md / CHANGELOG.md / docs/DECISIONS.md
config.yaml / requirements.txt
session_plans/state.json / action.json
gesture/
  config.py / collection.py / prompts.py    # 配置、采集阶段与提示范围
  hand_landmarker.py / state_recognizer.py  # 官方 21 点及静态分数
  features.py / data.py / labels.py         # State 特征、数据、监督
  action_features.py / action_resampling.py / action_data.py  # 54 维、重采样、裁窗与掩码
  action_candidates.py / action_annotations.py / action_ui.py
  models.py / training.py / action_training.py
  action_inference.py / action_evaluation.py
scripts/
  collect.py / download_hand_landmarker.py / download_gesture_recognizer.py
  extract_landmarks.py / annotate_state.py / train_state.py / run_state.py
  extract_action.py / annotate_action.py / train_action.py
  evaluate_action.py / view_action.py / run_action.py
tests/                                     # 时序、候选、取消、掩码等回归测试
models/ / data/ / runs/                     # 下载资源、真实数据与运行产物
```

`gesture/` 保存复用逻辑，`scripts/` 提供入口。运行测试使用指定环境：`python -m unittest discover -s tests -v`。

下一轮计划新增 `session_plans/periodic.json`（采集模板）、`gesture/action_decoding.py`（公共事件解码与 FSM）和 `gesture/waving_annotations.py`（PCA 折返草稿）。采集、标注、训练、评估、回放和在线入口扩展已有脚本；四通道模型计划保存到独立的 `runs/action_waving/`。
