# MultiGestureRecognition

基于摄像头和手部关键点的在线手势识别项目。当前实现第一阶段：MediaPipe Hand Landmarker、静态 State 数据采集与标注、State MLP 训练、录像回放和摄像头输出。一次性动作与周期性 `waving` 仍按下方流程规划，尚未实现。

一次性动作参考 Meta 的 [generic-neuromotor-interface](https://github.com/facebookresearch/generic-neuromotor-interface) 和[论文](https://www.nature.com/articles/s41586-025-09255-w)。Meta 使用 sEMG，本项目使用摄像头，输入特征需重新设计。

项目规则见 [AGENTS.md](AGENTS.md)；已定与待定事项见 [DECISIONS.md](docs/DECISIONS.md)；实际修改见 [CHANGELOG.md](CHANGELOG.md)。

本文是 V1 的实施说明。**已定**表示用户明确的规则；**候选**表示为实现第一版提出的默认方案，可经标注和验证调整。候选尚未成为用户确认的决策，集中列在 [DECISIONS.md](docs/DECISIONS.md)。第一阶段代码已经就绪；目前没有录制数据、训练权重或识别效果结果。后续阶段须等当前阶段经用户验收后再实现。

## 第一阶段：安装与运行

在项目根目录创建并进入指定的 Conda 环境，下载 Google 官方 Hand Landmarker 模型：

```bash
conda create -n MultiGestureRecognition python=3.11 pip
conda activate MultiGestureRecognition
python -m pip install -r requirements.txt
python -m scripts.download_hand_landmarker
```

依赖版本固定在传统的 `requirements.txt` 中。当前选用 MediaPipe 官方 `0.10.35` 正式发布版：它提供本机架构的 macOS arm64 wheel，且通过了本机官方模型加载和单帧推理检查。GitHub 当前最新的 `1.0.0` 发布记录说明已移除 macOS 预编译包，因此本项目固定在本机可运行的 `0.10.35`。MediaPipe 的 Hand Landmarker 模型资源下载到项目根目录的 `models/`；训练所得模型与运行结果写入 `runs/`。采集前先为每个 session 建目录和 `session.json`，例如 `data/sessions/static_01/session.json`：

```json
{
  "session_id": "static_01",
  "kind": "static",
  "target_model": "state",
  "gap_seconds": 2,
  "durations_seconds": {"closed": 5, "opened": 5, "thumb_up": 5},
  "prompts": [
    {"gesture": "opened"},
    {"gesture": "closed"},
    {"gesture": "thumb_up"}
  ]
}
```

提示顺序和动作时长按 session 设置。采集脚本显示当前手型、剩余时间和休息倒计时，保存视频及逐帧 `frames.csv`；结束后把提示覆盖的帧范围补进 `session.json`，并创建空的 `annotations.json`。执行：

```bash
python -m scripts.collect --session-id static_01
python -m scripts.extract_landmarks --session-id static_01
python -m scripts.annotate_state --session-id static_01
```

已录制过的 session ID 不会被采集脚本覆盖；重新录制时先建立新的 session 文件夹和 ID，避免旧标注与新视频混用。

标注窗口用滑块定位帧。按 `[` 设区间起点、`]` 设终点，再按 `1` 标 `closed`、`2` 标 `opened`、`3` 标 `thumb_up`；`x` 删除当前帧所在的标注区间，`s` 保存，`q` 退出。退出时也会自动保存本次修改。区间两端都包含在标签内。界面显示未标且连续检测到手的帧为自动 `NULL`；缺手帧及连续可见帧数未达到 `config.yaml` 要求的帧不参与监督。

每个录制 session 完成后，由用户在 `data/splits.json` 指定训练集和验证集。两组必须使用不同的 session，示例：

```json
{"train": ["static_01", "static_02"], "val": ["static_03"]}
```

训练集和验证集各自都需覆盖 `config.yaml` 中的全部 State 类别（包括自动生成的 `NULL`），然后训练并查看验证集指标：

```bash
python -m scripts.train_state
```

模型和训练指标分别写到 `runs/state/model.pt` 与 `runs/state/metrics.json`。训练后可回放验证录像，或打开摄像头查看逐帧类别概率；缺手时画面显示 `NO HAND`，不调用分类器：

```bash
python -m scripts.run_state --session-id static_03
python -m scripts.run_state
```

`q` 退出采集、标注、回放或实时画面。新仓库没有训练数据，因此需要先录制、标注并填写 session 划分，才能训练或运行分类器。

## 端到端流程

```text
摄像头录制 session → 提取手部关键点 → 标注 State、原子事件和 waving
                   → 按 session 划分数据 → 训练 State 与 Action 模型
实时画面 → 同一套关键点与特征处理 → State 概率 → Action 事件分数
         → 事件解码与统一 FSM → close_hand / open_hand / waving
```

### 1. 采集

采集工具持续显示摄像头预览，并按预先写好的 session 提示表逐项引导用户：屏幕显示当前要做的手势及剩余时间，用户在该时段内完成动作；时段结束后显示休息提示，**间隔 2 秒**再显示下一项。视频从第一项开始前持续录到最后一项结束，保留提示间隔中的静止、自然挪手等背景。静态手型、一次性动作和 `waving` 分开录制；一次性动作可在同一提示时段内重复出现。`waving` 使用张开的手、手掌大体朝向摄像头，覆盖绕手腕和绕肘关节的摆动。提示时段只是动作机会，不直接作为正标签；实际动作区间随后由用户标注。

第一版**候选**只处理画面中的一只手，左右按未镜像的原始图像 x 轴定义；显示预览是否镜像不得改变保存的数据与标签方向。

每次录制对应一个 `session_id`，使用独立文件夹。`session.json` 在录制前定义本次提示顺序、每种动作的时长、提示间隔及 `target_model`，录制后补记各次提示实际覆盖的帧范围。`target_model` 必填，只能是 `state` 或 `action`，明确这段数据用于训练哪个模型；`kind` 则说明录制内容，不能代替该字段。静态手型 session 标 `state`，一次性动作和 `waving` session 标 `action`；背景 session 也必须明确指定用途。各动作时长预先定好，可按 session 调整；提示间隔为用户指定的 2 秒。`frame_index` 从 0 开始；`elapsed_seconds` 是该帧在本次录制中的采样时间，供标注对齐和速度计算使用。

```text
data/sessions/<session_id>/session.json
data/sessions/<session_id>/video.mp4
data/sessions/<session_id>/frames.csv       # frame_index,elapsed_seconds
data/sessions/<session_id>/landmarks.csv    # frame_index,elapsed_seconds,hand_detected,x0,y0 ... x20,y20
data/sessions/<session_id>/annotations.json
data/splits.json                # 用户录制后指定 train / val 的 session_id 列表
```

例如一次性动作的提示表可以是下列形状；下面的时长只是示例，不是已定数值。静态和 `waving` session 使用相同结构，替换 `kind`、动作名和各动作时长。`prompts` 的每一项是一次屏幕提示；录完后在该项记录实际 `start_frame`、`end_frame`，便于标注时定位。

```json
{
  "session_id": "one_shot_01",
  "kind": "one_shot",
  "target_model": "action",
  "gap_seconds": 2,
  "durations_seconds": {"close_hand": 6, "open_hand": 5},
  "prompts": [
    {"gesture": "close_hand"},
    {"gesture": "open_hand"},
    {"gesture": "close_hand"}
  ]
}
```

采集脚本读取该文件，按提示表显示动作名、倒计时与休息倒计时；录制完成后补齐实际帧范围并创建空的 `annotations.json`。视频和时间表必须一帧对应一行；标注始终使用同一视频的帧序号。提示时间不自动生成 State、Action 或 `waving` 正标签，训练/验证归属也不写在 session 配置里。

### 2. 标注与生成训练目标

关键点提取以 Google 官方 [MediaPipe Tasks Vision Hand Landmarker Python 指南](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python)为准，使用其 21 个手部关键点接口。实施时对照[官方发布记录](https://github.com/google-ai-edge/mediapipe/releases)和 [PyPI](https://pypi.org/project/mediapipe/)选较新、兼容本机且已验证的正式版本，并固定在环境依赖中；模型资源使用官方推荐的 Hand Landmarker 模型。离线录像逐帧提取用 `VIDEO` 模式；实时摄像头也先以同步逐帧方式处理，保证一帧对应一次结果。官方 `LIVE_STREAM` 模式可能在忙时跳过输入帧，若以后使用，需要重新核对帧与标注的对应关系。MediaPipe 要求的逐帧时间参数由关键点提取模块内部处理。

State 的 `closed`、`opened`、`thumb_up` 正类区间由用户亲自标注。生成 State 训练目标时，人工正类优先；在 `target_model: state` 的 session 中，未被任何正类区间覆盖的帧，若**当前帧及此前连续若干帧均检测到手**，自动标为 `NULL`。这里“检测到手”指 Hand Landmarker 为该帧返回手部关键点；连续帧数 `K` 由 `config.yaml` 指定，检查当前帧和此前 `K-1` 帧。未满足条件的帧不参与 State 监督。**缺手帧不参与 State 或 Action 监督**，也不标为 State `NULL`。`thumb_up` 候选要求拇指伸出、其余手指弯曲；`closed` 候选要求全部手指弯曲。一次性动作的**候选**起点是离开原稳定手型的首帧，完成点是首次清楚达到目标手型的帧；用户可修订。

用户先标出实际 `waving` 区间。在区间内，对选定的手部位置轨迹做 PCA，将位置投影到主方向，自动生成左右折返点候选；用户可移动或删除候选。左右原子事件只有在向一侧运动后观察到折返才算完成。PCA 代表位置、平滑和最小幅度见 [待确认事项](docs/DECISIONS.md)。

标注文件采用一份可编辑的 JSON，区间两端均包含对应帧；`direction_events` 记录折返确认帧，而非整段 `waving` 的类别。以下仅展示字段形状，帧号是示例：

```json
{
  "session_id": "example_01",
  "state_intervals": [{"start_frame": 61, "end_frame": 180, "label": "opened"}],
  "action_intervals": [{"start_frame": 45, "end_frame": 60, "label": "open_hand"}],
  "waving_intervals": [{"start_frame": 70, "end_frame": 180}],
  "direction_events": [{"frame": 95, "label": "left_to_right"}]
}
```

`state_intervals` 只保存用户标注的 State 正类区间；满足连续有手条件的未标注帧在生成训练目标时补为 `NULL`，不写回人工标注。这也包括未被标为正类的手型过渡帧。`action_intervals` 用于人工标注一次性动作，完成点是 `end_frame`；`direction_events` 由 PCA 产生初稿并由用户修订。该 session 的 `kind` 和 `target_model` 以 `session.json` 为准，不在标注文件中重复维护。相同动作出现在不同采集类型中也遵守同一事件定义。

从完成点生成 Action 的短事件目标：对类别 `c`、完成帧 `e`，**候选**在 `[e+3,e+7]` 的每帧令 `target[t,c]=1`（含两端，其余类别可独立为 1）；区间以外的有效背景帧所有已启用事件通道为 0。检查目标事件是否标全后，未标正事件且最近若干帧持续有手的帧作为 Action 背景；丢手附近不按背景训练。Action 不设置单独的 `NULL` 输出通道。

PCA 候选生成的具体顺序是：取已标注 `waving` 区间内的手部代表位置 → 平滑轨迹 → 求二维 PCA 主轴 → 投影成一维轨迹 → 找到有效极值与折返确认帧 → 根据折返前的运动方向命名 `left_to_right` 或 `right_to_left`。**候选**用五个指尖的像素坐标中心作为代表位置，以覆盖掌心几乎不动的腕部摆动；PCA 主轴的符号固定为与图像 x 轴正向点积非负。极值后连续观察到配置帧数的反向运动时，以确认帧而非物理极值帧作为事件时间。平滑、有效幅度和确认帧数放进 `config.yaml`，用户可修订所有候选。

### 3. 划分与裁窗

用户录制后指定各 session 的训练或验证归属，先划分 session 再裁窗，避免重叠窗口跨集合。训练 State 时只读取 `target_model: state` 的 session；训练 Action 时只读取 `target_model: action` 的 session，同时使用已训练 State 模型产生的概率特征。训练窗口允许高重叠，起点不在动作中，并希望部分窗口包含两个动作。约 8 秒只是候选长度。验证按完整 session 计数，不把同一录像的重叠窗口当作独立样本。

`splits.json` 的键为 `train`、`val`，值为互不重叠的 `session_id` 列表。所有窗口、速度统计量和模型选择都以这个划分为准。裁窗前先从标注生成逐帧目标；若一个窗口截掉动作历史却保留了该动作的延迟正标签，应跳过该窗口或屏蔽对应帧的损失。缺手帧既不作正标签，也不作背景负样本；达到连续缺手阈值时，训练序列与在线推理一样从新的 LSTM 历史开始。

### 4. 模型训练

先训练 State MLP：它读取单帧局部归一化的 21×(x,y) 关键点（42 维），输出 `NULL`、`closed`、`opened`、`thumb_up` 的概率。再训练 Action 因果 LSTM：它每帧读取局部关键点 42 维、归一化掌心坐标/速度/加速度各 2 维、腕部到中指的方向角 1 维，以及当前帧 State 概率 4 维，当前合计 53 维；第二段输出 `close_hand`、`open_hand`，第四段再加入 `left_to_right`、`right_to_left`。两模型对每个有效手帧运行，不设静态/动态路由。

特征处理先产生 42 维 State 输入和 49 维 Action 基础特征；State 模型训练完成后，才把其四维软概率拼到基础特征上形成 53 维 Action 输入。在线推理也按这个顺序处理当前帧。

V1 特征契约如下；关键点编号采用 MediaPipe Hand Landmarker 的 21 点顺序。局部关键点以 wrist（0）为原点，**候选**用 wrist（0）到 middle MCP（9）的距离作手部尺度，不把手旋转到统一朝向，以保留摆动角度。掌心位置取 wrist（0）和四个 MCP（5、9、13、17）的均值，并保留其画面归一化坐标。腕部到中指角度的**候选定义**是 wrist（0）指向 middle MCP（9）的向量相对图像 x 轴的角度，取 `atan2(Δy, Δx)/π`。这些候选定义仍见 [待确认事项](docs/DECISIONS.md)。

| 模型 | 每帧输入 | 输出 | 训练目标 |
| --- | --- | --- | --- |
| State MLP | `[42]` 局部 x/y | `[4]` 类别概率 | 有效且已标注手型的单帧分类；无手和无法判断的帧不计算损失 |
| Action LSTM | `[53]`：42 局部 x/y + 2 掌心位置 + 2 速度 + 2 加速度 + 1 角度 + 4 State 概率 | 第二段 `[2]`，第四段 `[4]` 事件分数 | 每类独立的短事件脉冲；有效背景帧所有已启用通道全零 |

模型结构**候选**为 `State: Linear(42, hidden) → ReLU → Linear(hidden, 4)`；`Action: 单向 LSTM(53, hidden, layers) → Linear(hidden, 已启用事件类数)`。`hidden`、`layers` 和事件类别列表由 `config.yaml` 提供，不加入 Meta 针对原始 sEMG 的卷积前端。

训练顺序：先完成 State 训练，再用该模型产生训练与验证录像的 State 软概率，作为 Action 的输入；不以人工 State 标签的 one-hot 代替模型概率。模型内部输出 logits：State 用交叉熵训练、经 softmax 得到四类概率；Action 对已启用事件通道逐帧用二元交叉熵训练、经 sigmoid 得到各事件概率，损失只计算有效帧。这沿用 Meta 的“独立事件概率 + 多标签 BCE”思路，输入网络仍针对摄像头关键点设计。[Meta 论文与模型](https://www.nature.com/articles/s41586-025-09255-w)。

训练与在线推理共用特征公式。速度和加速度按实际观测间隔计算；归一化统计量只从训练集拟合，并复用于验证和在线推理。跨 session 共用的处理、模型、训练和 FSM 超参数放在 `config.yaml`；单个 session 的采集参数留在该 session 的 `session.json`。

速度定义为相邻有效掌心位置之差除以采样时间差；加速度是相邻速度之差除以采样时间差。画面 x/y 已在 `[0,1]`，速度与加速度用训练集统计量归一化，验证和实时推理不重新拟合。局部手型、角度及短暂缺手后重新观测的特征处理细节属于 V1 候选，实现时须与标注和推理保持一致。

Action LSTM 的上下文方式**候选**为训练时每个窗口从零 hidden state 开始，在线时每帧重算最近一个同长度的滚动窗口；这样训练和推理看到的历史一致。后续若改为持续 hidden state，必须同时修改训练和实时更新方式，不能只改推理端。

### 5. 在线推理与后处理

每个有效手帧依次运行关键点与特征处理、State MLP、Action LSTM。事件分数经过阈值触发与去重，再交给统一 FSM；模型不直接上报动态动作。FSM 从两个相反方向的原子事件（R→L 或 L→R）首次触发 `waving`，之后每收到预期方向事件重新计时；超过 300 ms 未收到则退出。一次性动作也通过 FSM 上报。阈值、连续缺手帧数和重复上报规则见 [待确认事项](docs/DECISIONS.md)。

缺手帧不喂全零特征，不运行 State 或 Action 推理，也不产生事件；连续缺手达到阈值后重置 Action LSTM。当前滚动窗口重算的**候选**实现中，“重置”即清空窗口历史，恢复可见手后重新积累。阈值暂以 `no_hand_reset_frames: 3` 为候选；短暂缺手时先跳过这些帧、不更新 LSTM 历史。FSM 仍按时间等待预期事件，`waving` 的 300 ms 超时规则照常生效。若批量训练需要补齐序列长度，全零只作带掩码的 padding，不作为缺手样本。

事件解码的 V1 候选是每类分数**从阈值下方上穿**时生成一个候选事件，并对同类短间隔重复峰去重；不同方向事件不做全局互斥去重，以免抹掉快速折返。这借鉴 Meta 的[事件解码实现](https://github.com/facebookresearch/generic-neuromotor-interface/blob/main/generic_neuromotor_interface/cler.py)。阈值和去重间隔由验证数据选定，放在 `config.yaml`。

`waving` FSM 的最小状态转换：

| 当前状态 | 输入 | 下一状态与输出 |
| --- | --- | --- |
| 空闲 | 首个 L→R 或 R→L | 记住方向，等待相反方向 |
| 等待第二个 | 相反方向事件 | 进入活跃状态，首次上报 `waving` |
| 等待第二个 | 超过候选的首次配对间隔 | 丢弃首个候选，回到空闲 |
| 活跃 | 下一个预期方向事件 | 保持活跃，更新预期方向并重新计时 |
| 活跃 | 300 ms 内没有预期事件 | 退出，回到空闲 |

首次两个事件的最大间隔、活跃期间的重复上报及非预期事件如何处理尚待确认。为保持实现简单，第一版**候选规则**是首次配对也限 300 ms、每段只上报一次、同向重复事件忽略；这些数值和行为写入 `config.yaml`，在 [DECISIONS.md](docs/DECISIONS.md) 保留候选状态。`close_hand`、`open_hand` 各自从事件候选经过阈值和去重后由同一 FSM 上报，不绕过它。

第一版在完整验证 session 上检查原子事件和 `waving` 的漏检、误报与延迟，再针对实际错误补采负样本。

## 分段实现与验收

以下顺序是实施约束。每段交付可运行、可回放的结果，经用户验收后才进入下一段；后面的完整 pipeline 和配置示例描述最终形态，不要求第一段一次建完。第一段的 State 指静态手型分类及输出，不是 `waving` 的 FSM。

| 阶段 | 本段实现 | 本段验收 |
| --- | --- | --- |
| 1. MediaPipe + State | 官方 Hand Landmarker、State session 的提示采集与标注、单帧特征、State MLP 训练，以及录像回放和摄像头输出 `NULL`、`closed`、`opened`、`thumb_up` 概率/类别 | 视频、帧表、标注和关键点能对齐；按 session 分开的验证数据可回放并查看逐帧预测；实时画面能显示静态手型，缺手时无模型输出 |
| 2. 一次性动作 | `close_hand`、`open_hand` 的 Action session 采集与事件标注、因果 LSTM 训练、逐帧事件分数及回放评估；Action 输入接第一段 State 模型的软概率 | 在完整验证 session 上查看两个原子事件的漏检、误报和延迟；能逐帧检查标签脉冲、有效帧掩码与预测分数 |
| 3. 一次性动作 FSM | 将第二段事件分数接入阈值上穿、同类去重和统一 FSM，对外上报 `close_hand`、`open_hand`；回放与实时运行共用解码逻辑 | 回放和摄像头都能输出最终一次性动作；按完整 session 检查上报次数、误报及延迟 |
| 4. 周期性 `waving` | 采集 `waving` session、人工标区间、PCA 生成并修订折返事件；将 `left_to_right`、`right_to_left` 加入 Action 模型，并扩展第三段的 FSM | 人工事件接 FSM 与模型事件接 FSM 均可回放；两个相反方向事件触发 `waving`，后续预期事件续期、300 ms 超时退出；按完整 session 检查漏检、误报和触发延迟 |

第二段的 Action 输出暂只覆盖 `close_hand`、`open_hand`；第四段加入两个方向事件，成为最终四通道输出。事件类别列表随阶段由配置管理，数据仍按 session 划分。阶段验收使用已有验证录像，未录制所需 session 时不声称该阶段识别效果已经验收。

## 模块接口与实现状态

所有训练和在线入口读取同一个 `config.yaml`。`gesture/` 保存可复用逻辑，`scripts/` 只负责入口和画面；训练与在线 State 分类共用特征处理。下表中第一阶段为当前实现，后续阶段仍是规划。

| 阶段 | 模块入口 | 输入 → 输出 | 状态 |
| --- | --- | --- | --- |
| 采集 | `scripts/collect.py` | 摄像头、State session 提示表 → 视频、逐帧时间表、提示帧范围、空标注文件 | 已实现；当前只接受 `kind: static` |
| 关键点提取 | `scripts/extract_landmarks.py`、`gesture/hand_landmarker.py` | 视频、帧时间 → 每帧 21 点或缺手标记 | 已实现；离线及实时均使用官方 Tasks Vision `VIDEO` 模式 |
| 标注与目标 | `scripts/annotate_state.py`、`gesture/labels.py` | 视频与关键点 → State 正类区间及自动 `NULL`/忽略目标 | 已实现；Action 与 PCA 标注尚未实现 |
| 特征与数据 | `gesture/features.py`、`gesture/data.py` | 关键点、标注及 session 划分 → `[T,42]` State 输入、标签和有效帧 | 已实现；Action 的 `[T,49]` 特征尚未实现 |
| 训练与评估 | `gesture/models.py`、`gesture/training.py`、`scripts/train_state.py` | 按 session 划分的 State 数据 → State MLP、验证指标 | 已实现；训练依赖用户录制并标注的数据 |
| 回放与实时运行 | `scripts/run_state.py` | 已训练 State 模型、录像或摄像头 → 每帧类别概率；缺手时不推理 | 已实现；动态事件和 FSM 尚未实现 |
| Action、解码、FSM | 后续新增模块 | Action 事件分数 → `close_hand`、`open_hand` 与 `waving` | 阶段 2–4，当前未实现 |

State 接口为 `[B,42] → [B,4]`；Action 输入为 `[B,T,53]`，第二段先输出两个事件通道，第四段扩展为最终的 `[B,T,4]`。`labels.py` 负责把人工或修订后的事件转换为逐帧目标及损失掩码；`fsm.py` 接收逐帧分数、State 概率和帧时间，返回零个或多个最终动作事件。第四段先让人工方向事件经过 FSM 得到合理的 `waving`，再用模型预测的事件替换，便于区分标注/FSM 问题和模型问题。

验证至少按完整 session 报告每类原子事件的精确率、召回率和检测延迟，以及 `waving` 的按段召回、每分钟误报和首次触发延迟。匹配容差与目标数值仍见 [DECISIONS.md](docs/DECISIONS.md)；不得只用训练损失判断完成。

## 配置边界

`config.yaml` 至少分为 `data`（采集帧率、session 根路径、裁窗与有效手历史）、`hand_landmarker`（官方模型资源路径与检测阈值）、`labels`（PCA 平滑/幅度/折返确认、事件脉冲）、`features`（局部尺度、掌心和角度定义、归一化）、`models`、`training`、`decoder`（阈值和去重）及 `fsm`（首次配对、300 ms 退出、上报规则）。跨 session 共用的数值在这里维护；各 session 的提示顺序、动作时长和 2 秒间隔只在本 session 的 `session.json` 中维护。训练得到的归一化统计量作为运行产物保存。

下面是完整 V1 后续阶段配置项的**候选初值草案**，不是当前 `config.yaml` 的内容，也不是用户确认的最优参数。各阶段实施时再把对应参数加入 `config.yaml`，代码读取该配置；用用户修订的训练/验证 session 调整。Stage 1 当前实际运行配置以仓库根目录的 `config.yaml` 为准。

```yaml
data:
  sessions_dir: data/sessions
  splits_file: data/splits.json
  fps_target: 30
  train_window_seconds: 8.0
  train_stride_seconds: 1.0
  action_background_visible_frames: 3
  no_hand_reset_frames: 3
hand_landmarker:
  model_asset_path: models/hand_landmarker.task
  delegate: CPU
  num_hands: 1
  min_hand_detection_confidence: 0.5
  min_hand_presence_confidence: 0.5
  min_tracking_confidence: 0.5
labels:
  state_null_visible_frames: 3
  positive_start_offset_frames: 3
  positive_end_offset_frames: 7
  pca_smooth_frames: 5
  pca_reversal_confirm_frames: 2
  pca_min_amplitude_hand_scales: 0.5
features:
  local_scale: wrist_to_middle_mcp
  pca_position: fingertips_center
models:
  state_hidden_size: 128
  action_hidden_size: 128
  action_num_layers: 1
  action_classes: [close_hand, open_hand, left_to_right, right_to_left] # 最终阶段；第二段先用前两类
training:
  batch_size: 32
  learning_rate: 0.001
  max_epochs: 30
decoder:
  threshold: 0.5
  same_class_debounce_ms: 100
fsm:
  first_pair_max_gap_ms: 300
  wave_timeout_ms: 300
  wave_report_once: true
```

实现第一版所需而用户尚未确认的少量语义在 [DECISIONS.md](docs/DECISIONS.md) 标为候选，不能写成既定产品规则：单手与坐标方向、State `NULL` 的连续可见帧数、一次性动作完成点、连续缺手阈值与短暂缺手处理、PCA 候选阈值和输出接口。先把这些选择集中在配置或清晰的单处实现，再用用户修订的标签与独立验证 session 检查。

## 当前仓库结构

Stage 1 已有的代码与后续阶段规划模块如下。`data/` 与 `runs/` 在运行时创建，实际录制数据和模型产物不纳入 Git。

```text
.
├── AGENTS.md
├── README.md
├── CHANGELOG.md            # 按日期记录实际修改
├── config.yaml              # 跨 session 共用的数据处理、特征、模型、训练、FSM 超参数
├── requirements.txt         # 固定 Python 依赖版本
├── .gitignore               # 本地数据、模型产物和虚拟环境
├── docs/
│   └── DECISIONS.md
├── gesture/
│   ├── __init__.py
│   ├── config.py            # 读取统一配置与解析项目路径
│   ├── data.py              # Stage 1 session、划分与 State 数据读取
│   ├── features.py          # State 单帧 42 维关键点特征
│   ├── hand_landmarker.py   # MediaPipe Tasks Vision 关键点提取
│   ├── labels.py            # State 正类与自动 NULL 目标
│   ├── models.py            # State MLP
│   └── training.py          # State 训练、验证指标及 checkpoint
├── scripts/
│   ├── collect.py           # State 摄像头采集
│   ├── download_hand_landmarker.py
│   ├── extract_landmarks.py # 每帧提取关键点
│   ├── annotate_state.py    # State 区间标注
│   ├── train_state.py       # State MLP 训练
│   └── run_state.py         # 录像回放或摄像头 State 推理
├── models/                  # 下载的 MediaPipe Hand Landmarker 资源；不入库
├── data/                    # 每个 session 的视频、提示表、标注及总体划分；不入库
└── runs/                    # 训练模型与运行结果；不入库
```

`scripts/` 只提供命令行入口，数据处理与分类逻辑放在 `gesture/`。真实 session 的训练/验证归属由用户录制后指定。Action 模型、PCA 方向事件标注、事件回放评估、解码及 FSM 会在阶段 2–4 按验收顺序增加。没有真实 session 时，可以运行采集和标注流程，但不能声称模型已训练或识别效果已验收。
