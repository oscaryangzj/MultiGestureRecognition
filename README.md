# MultiGestureRecognition

基于摄像头关键点的在线手势识别研究原型：识别静态手型、一次性开合动作，并将左右折返事件组合为周期性挥手（`waving` / BYEBYE）。提供数据采集、人工标注、模型训练、事件评估和摄像头推理的完整流程。

## 演示

[观看在线手势识别 demo](assets/demo.mp4)（38 秒）

**当前已跑通端到端链路，在现有小规模验证集上达到 100% 原子事件召回率（112/112）；现阶段实测使用效果很好。** 目前只采集了少量正样本，已有负样本覆盖也有限；后续需要针对在线误报定向采集更多负样本，并完成独立验收。

一次性动作建模参考 Meta 的 [generic-neuromotor-interface](https://github.com/facebookresearch/generic-neuromotor-interface) 及论文 [A generic non-invasive neuromotor interface for human-computer interaction](https://www.nature.com/articles/s41586-025-09255-w)。本项目使用视觉关键点，特征、采样率和 waving FSM 按摄像头任务设计。

## 方法

```text
摄像头 / 录像
  → 官方 MediaPipe：同帧 21 点二维/三维坐标 + 静态手型分数
  → 固定 30 Hz 重采样、因果特征
  → 因果 Conv1d + 三层单向 LSTM + 独立 Sigmoid
  → 原子事件解码 → waving FSM → 单一 Action / NULL
```

本仓库统一以 30 fps 为目标帧率：采集录像按 30 fps 写入，在线 Action 摄像头按 30 fps 配置，训练与推理输入按帧时间戳重采样至 30 Hz。实际摄像头输出帧率可能随设备和处理负载波动。

| 层级 | 输出 | 含义 |
| --- | --- | --- |
| State | `opened`、`closed`、`thumb_up` | 当前静态手型，直接使用官方模型；不训练本项目 State MLP |
| Action 原子事件 | `open_hand`、`close_hand` | 握拳→完整张开、完整张开→握拳；保持手型不重复报事件 |
| 方向原子事件 | `left_to_right`、`right_to_left` | 张开手向一侧移动，到端点后观察到连续回移才完成 |
| 最终 Action | OPEN HAND、CLOSE HAND、BYEBYE、NULL | 开合短暂显示；waving 活跃时显示 BYEBYE |

**Action 输入共 54 维：** 21 点局部二维坐标（42）+ 掌心速度（2）+ 加速度（2）+ 腕部角速度（1）+ opened/closed 分数（2）+ 五指三维弯曲角（5）。局部坐标以 wrist 为原点、按 wrist→middle MCP 距离缩放、对齐掌轴并统一左右手；运动特征保留相机方向，绝对位置不输入模型。三维角采用四指 PIP 和拇指 MCP，不增加角度变化量。

四个事件通道独立监督，没有额外 NULL 类。人工完成帧起连续 4 帧为正脉冲；重采样后从不早于完成时刻的首个采样点开始。其余已审核且连续有手的帧监督为零。缺手、未审核和取消范围 IGNORE。短缺手跳过输入，连续缺手清空运动与模型历史。

因果解码在分数达到触发阈值时输出一次，降到复位阈值以下后才允许再次触发；开合与方向各自作为冲突组，同组同帧择高，完全同分不输出。

waving FSM 收到两个相反方向事件后启动，之后只由预期方向续期；当前配对窗 500 ms、活跃超时 600 ms。PCA 仅用于离线标注候选，在线由 Action 模型识别方向事件。

## 安装与快速运行

当前配置在 **macOS Apple Silicon、Python 3.11** 上验证；训练使用 PyTorch MPS，在线推理使用 CPU。依赖版本固定在 `requirements.txt`。以下命令均在项目根目录执行。

```bash
conda create -n MultiGestureRecognition python=3.11 pip
conda activate MultiGestureRecognition
python -m pip install -r requirements.txt
python -m scripts.download_gesture_recognizer
```

已有同名环境时直接激活。首次打开摄像头需给运行程序摄像头权限；相机编号和界面字体路径在 `config.yaml` 中设置。

**无需自训练即可查看官方 State：**

```bash
python -m scripts.run_state
```

**已有 Action 权重时运行完整识别：**

```bash
python -m scripts.run_action
```

仓库提供最终 Action 权重 `models/action/model.pt`，默认配置可直接加载。官方 `.task` 模型由下载脚本获取。训练配置、原 Action 数据划分和指标摘要一并保存在 `models/action/`；原始训练录像不随源码发布，复现训练结果还需对应数据。

界面同时显示 State 和最终 Action。`R` 开始/停止录制，`D` 展开分数和 FSM 细节，`Q` 退出。摄像头预览为镜像，保存的原始录像和模型输入使用原始相机方向。方向标签按原始相机 x 轴定义，与手侧无关；采集时手侧填写实际使用的手。

## 完整 pipeline

### 1. 采集开合、waving 和负例

```bash
python -m scripts.collect --plan action
python -m scripts.collect --plan periodic
```

每条命令采一个 session。输入采集者姓名及正/负类型，选择组数或负任务；在摄像头预览点击 START 后才录制，PRACTICE 不保存。session ID 自动由时间和姓名组成，采集结束打印 ID 及后续命令。

| 计划 | 正例交互 | 负例交互 |
| --- | --- | --- |
| `action` | 初始张开；每个提示完成一次开合，随后保持目标手型。默认 10 组/20 次，动作提示 3 s，随机 HOLD 0.8–2.5 s | START 前选一个任务；PREPARE 3 s 后活动 30 s |
| `periodic` | 选择手侧、腕部/肘部和段数；默认 5 段，每段 PREPARE 2 s、WAVE 随机 4–6 s、HOLD 0.5 s、REST 2 s | 选一个任务；覆盖单向移动、单次折返、进出画面、端点停顿和已有开合干扰任务 |

waving 使用张开手、掌心大体朝摄像头，沿腕部或肘部左右摆动均可。端点长时间停住再回移不算连续折返。负任务中意外发生真实目标动作时如实标注。

采集模板在 `session_plans/action.json`、`session_plans/periodic.json`；本次实际顺序、时长、手侧和提示帧范围保存到各自 `session.json`。历史 session 按自己的记录处理。

### 2. 提取关键点与 State 分数

将 `SESSION_ID` 替换为采集器打印的 ID，对每个 session 执行：

```bash
python -m scripts.extract_action --session-id SESSION_ID
python -m scripts.annotate_action --session-id SESSION_ID
```

提取器共享一次手部关键点检测，再用官方分类权重取得三类分数；保存二维 21 点、三维 world landmarks 和 State 分数。启用五指角度需要 `world_landmarks.csv`；旧数据缺该文件时重新提取即可，人工标注保留。

### 3. 人工审核与标注

界面顶部显示当前段、审核进度和保存状态；下方提示下一步。默认时间轴显示动作区间，`H` 才展开训练标签和 PCA 等细节。

| 操作 | 按键 |
| --- | --- |
| 前/后一帧、播放/暂停 | `A` / `D`、`P` 或空格 |
| 设置起点、完成帧；跳到已标边界 | `I` / `O`；`J` / `K` |
| 上一段、下一段 | `[` / `]` |
| 确认本段，自动保存并前进 | `Enter` |
| 忽略本段 | `X` |
| 只保存进度、退出 | `S`、`Q`（退出也保存） |

- **开合正例：** 检查自动候选，类别沿用提示。`I` 标开始离开原手型的帧，`O` 标首次完整达到目标手型的帧，`Enter` 确认。
- **waving 正例：** 每段分别用 `I/O` 手选开始/结束，`G` 保存区间并生成已分配方向的候选；点击事件或上一/下一候选，检查并用 `O` 调整到端点后已开始回移的完成帧。全部检查后 `Enter` 完成本段。`W` 重选 waving 区间，`Del` 删除错误候选。
- **负例：** 看完整段；无目标动作直接 `Enter` 审核背景。有真实开合时用 `1/2` 选类别、`I/O` 标边界、`E` 添加。periodic 用 `B` 展开补标，`3/4` 对应两个方向；一次有效折返应标原子事件，即使不足以组成 waving。

所有区间为两端包含的 `[开始帧,结束帧]`。动作区间用于保留完整历史，正脉冲只从完成帧开始；相邻方向事件的历史允许重叠。草稿和仅按 `S` 保存的未审核段不参与监督。训练前每段都须确认或忽略。

### 4. 按完整 session 划分数据

创建或修改 `data/splits.json`，填实际 session ID；正负例共同放入各组，组间不重复：

```json
{
  "train": ["训练开合正例ID", "训练waving正例ID", "训练负例ID"],
  "val": ["验证开合正例ID", "验证waving正例ID", "验证负例ID"],
  "acceptance": []
}
```

combined 模式下，train 和 val 都必须覆盖四类有效正事件。重叠窗口只在划分后生成，不能把同一 session 的窗口分给不同组。独立验收数据放入可选 `acceptance`，不用于选模和调阈值。

发布模型的原始划分见 `models/action/splits.json`；重新采集的数据填写自己的 `data/splits.json`。

### 5. 训练

当前 `config.yaml` 已设 `action_model_mode: combined`、`training.device: mps`。仅训练开合可改为 `one_shot`，模型分别写入对应的配置目录。

```bash
python -m scripts.train_action
```

默认重采样 30 Hz、8 s 窗口、stride=5，使用训练集统计归一化、镜像增强和带通道掩码的 BCE。PREPARE/REST 不进入负例与 periodic 标准训练历史；每个任务独立计算运动特征。裁窗避开动作起点中间，截掉动作历史的正脉冲不监督。

每轮按完整验证 session 的因果事件 macro F1 选模型，镜像验证也参与选模；并列时比较误报率、延迟和 BCE。镜像结果不算新增真实样本。

当前默认输出目录为 `models/action/`，训练写入 `model.pt`、`metrics.json` 和验证预测，并替换该目录已有的模型和指标。留存本次完整配置及划分：

```bash
cp config.yaml models/action/train_config.yaml
cp data/splits.json models/action/splits.json
```

改输出目录时相应调整复制路径。`train_action --config PATH` 可使用另存的配置；发布的 `models/action/train_config.yaml` 保留原训练设置，复训输出到 `runs/action_reproduced/`，最终在线 FSM 设置以根目录 `config.yaml` 为准。checkpoint 自带特征定义、归一化、采样率、标签规则和模型结构。

### 6. 评估与录像回放

```bash
python -m scripts.evaluate_action --split val
python -m scripts.evaluate_action --split val --continuous
python -m scripts.run_action --session-id SESSION_ID
```

标准评估按采集任务隔离历史；`--continuous` 保留整条录像历史，用于检查在线场景中的前序动作影响。`run_action --session-id` 使用同一在线链路播放录像，不按提示重置。需要用保存的训练配置评估时添加 `evaluate_action --config PATH`。

正式指标取 `decoded_events`：逐类 precision/recall/F1、漏报、误报/分钟和延迟；waving 另报启动、重复启动、活跃覆盖和退出延迟。只在已审核有效范围计分。结果写入模型目录的 `evaluation/`，包括 `predictions.csv`、`events.csv`、`final_events.csv`、`metrics.json` 和 `failure_cases.json`；连续结果单独存于 `evaluation/continuous/`。

冻结模型和阈值后，可运行 `python -m scripts.evaluate_action --split acceptance --continuous` 做独立验收。在线按 `R` 录下的连续 session 也保存在 `data/sessions/`，可按步骤 2 提取与人工标注后加入验收划分。

连续录像标注使用 `M/R/W` 切换开合、方向、waving：用类别键和 `I/O` 选区、`E` 添加草稿、`Enter` 逐个确认。候选全部处理后，还需分别在 `M` 和 `R` 模式用 `I/O` 选择审核范围、`Enter` 确认背景；范围须覆盖整条录像，无法判断的范围显式忽略。它与 periodic 的“完成本段”操作不同。

### 7. 摄像头推理

```bash
python -m scripts.run_action
```

当前在线设置为单次关键点检测、640 宽输入、CPU 两线程和默认收起详细曲线；训练设备与在线设备分别配置。State 的 `thumb_up` 可显示，Action 输入仍只取 opened/closed。

## 数据规模与链路验证

截至 2026-10-09，当前 combined 模型使用的数据规模如下；事件数为实际参与监督的事件，不是提示次数：

| 数据 | 训练 | 验证 |
| --- | ---: | ---: |
| 开合正例 session | 6 | 2 |
| waving 正例 session / 已标区间 | 4 / 19 | 1 / 5 |
| 动态负例 session | 8 | 4 |
| 有监督开合事件 | 119 | 40 |
| 有监督方向事件 | 294 | 72 |

合计 18 条训练、7 条验证 session，有效监督时间约 17.3/6.3 分钟。裁出 4,158/1,594 个重叠窗口；窗口数不代表独立样本量。以下为当前 54 维 checkpoint（最佳 epoch=23）的小规模链路验证结果：

| 原子事件 | 检出 / 真值 | 误报 | F1 |
| --- | ---: | ---: | ---: |
| open_hand | 20 / 20 | 1 | 0.976 |
| close_hand | 20 / 20 | 1 | 0.976 |
| left_to_right | 35 / 35 | 0 | 1.000 |
| right_to_left | 37 / 37 | 1 | 0.987 |

原子事件召回率 **100%（112/112）**，误报 3 次，macro F1=0.9845，来自当前 checkpoint 的验证记录。500/600 ms FSM 在同组验证录像的连续回放中得到 waving 5/5、0 次额外启动、匹配段活跃覆盖 100%；实际停止到退出中位约 500 ms、最长约 833 ms。摘要见 `models/action/metrics.json`。现阶段摄像头实测反馈：实际使用效果很好。

这些是单一采集者的小样本验证结果，periodic 正例来自右手；镜像检查不能代替真实左手数据。跨采集者、真实左手、慢 waving 和端点长停顿的泛化仍待独立验收。

**下一步重点补采针对性负样本：** 从在线误报录像中整理局部手指活动、保持手型时的上下摆/转腕/平移、手进出画面、单次侧移以及端点停顿后回移等场景，扩大动态背景覆盖。意外发生真实目标动作时仍如实标注；补采后重新训练，并用独立 session 检查误报率。

## 文件与配置

```text
docs/                    # 协作规则、决策和修改记录
  AGENTS.md
  DECISIONS.md
  CHANGELOG.md
config.yaml              # 数据处理、特征、训练、模型和 FSM 的共用参数
requirements.txt         # 固定依赖
session_plans/           # 开合与 periodic 采集模板
scripts/                 # 采集 → 提取 → 标注 → 训练 → 评估 → 在线推理
gesture/                 # 上述流程的共用实现
models/                  # 下载的官方模型（不纳入 Git）
  action/                # 发布权重、训练配置、划分和指标摘要
data/sessions/<ID>/       # 每次采集的数据
  session.json           # 采集者、类型、手侧、实际提示和阶段
  video.mp4, frames.csv   # 原始录像和对应帧时间
  landmarks.csv          # 21 点二维坐标
  world_landmarks.csv    # 21 点三维坐标
  state_scores.csv       # 官方静态分数
  annotations.json       # 人工事件、waving 区间及审核范围
data/splits.json         # train / val / acceptance
runs/                    # 本地复训、实验及诊断产物
```

源码中保留的旧 State MLP 采集/训练工具不属于当前 pipeline，发布时留在本地。协作规则见 [AGENTS.md](docs/AGENTS.md)，选择依据与待确认项见 [DECISIONS.md](docs/DECISIONS.md)，版本变化见 [CHANGELOG.md](docs/CHANGELOG.md)。

## 参考

- Meta：[代码与数据](https://github.com/facebookresearch/generic-neuromotor-interface)、[Nature 论文](https://www.nature.com/articles/s41586-025-09255-w)。
- MediaPipe：[Hand Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python)、[Gesture Recognizer](https://developers.google.com/edge/mediapipe/solutions/vision/gesture_recognizer/python)。
