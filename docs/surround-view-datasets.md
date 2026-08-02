# 车载环视与泊车场景公开数据集调研

调研日期：2026-08-02

## 结论

ParkSight-VLM 的目标输入不应理解为普通道路照片，而应优先采用下列三种汽车近场视觉表示之一：

1. 车身前、后、左、右的原始鱼眼单帧；
2. 同一时刻四路鱼眼图组成的 `2 x 2` mosaic；
3. 四路鱼眼经标定和拼接得到的 AVM 鸟瞰图。

现有公开数据集没有直接提供本项目的 `risk_level`、`events`、`evidence`、`driver_advice` 标签。公开标注主要是 2D/3D 框、语义/实例分割、深度、运动、停车位顶点或占用状态。它们可以帮助筛选候选图片和核对图中事实，但最终仍需按 ParkSight 的严格 JSON 契约进行二次人工标注。

当前最可执行的组合是：

- **WoodScape**：获取真实、车载、四向鱼眼近场图片，作为主要人工标注来源；
- **AVM-SLAM Dataset**：补充地下车库、弱光、重复纹理下的四路鱼眼与 AVM 连续序列；
- **Tongji ps2.0 + VPS-Net occupancy annotations**：补充停车位形状、可用/占用和车位冲突样本，但其输入是已拼接 AVM 图，不是原始鱼眼图。

**FPD（Fisheye Parking Dataset）在场景和标签上最匹配，但原始论文及论文落地页没有给出公开下载入口或数据许可证，因此目前不能把它当作可直接获得的数据源。** SynWoodScape 可用于合成角落案例和标签工具验证，但不应进入最终真实场景测试集。

## 首先需要冻结的输入定义

数据集下载之前，应先决定项目正式输入到底是哪一种表示。三种表示不能混成同一质量基线：

| 表示 | 优点 | 主要限制 | 对现有单图接口的影响 |
| --- | --- | --- | --- |
| 单路原始鱼眼 | 与量产环视相机最接近；保留畸变和细节 | 看不到完整 360°；风险可能出现在另一相机 | 无需改接口，但 manifest 应记录 `camera_position` |
| 四路 `2 x 2` mosaic | 一张图保留四向信息；不需要先做几何拼接 | 单路分辨率下降；模型需理解四个视角的空间关系 | 仍可走单图接口，需冻结拼图顺序与尺寸 |
| AVM 鸟瞰图 | 车辆附近空间关系直观，适合车位和狭窄通道 | 拼接盲区、拉伸和车辆遮罩会改变图像事实 | 仍可走单图接口，但结论仅适用于 AVM 输入 |

还有一个比相机格式更重要的边界：`vru_near_maneuver_path` 和 `vehicle_near_maneuver_path` 都依赖“车辆将沿哪条轨迹运动”。如果输入只有静态图片，且没有倒车方向、转向意图、规划轨迹或可见引导线，标注员无法唯一判断对象是否位于“maneuver path”。正式数据契约应至少增加以下一种上下文：

- `maneuver_direction`，例如 `forward`、`reverse`、`turn_left`、`turn_right`；
- 图像中的规划轨迹/倒车引导线；
- 车辆坐标系下的轨迹多边形或轨迹点；
- 若坚持纯单图输入，则将该事件解释冻结为“位于车辆紧邻的潜在运动走廊”，并在标注指南中明确这是代理定义，不是已知规划轨迹。

## 数据集总览

| 数据集 | 相机与布局 | 是否目标视角 | 规模与现有标注 | 获取与限制 | 对 ParkSight 的判断 |
| --- | --- | --- | --- | --- | --- |
| WoodScape | 车载前/后/左/右 4 路，约 190° 鱼眼，覆盖 360° | 是，真实车载原始鱼眼；但不专注停车场 | 原始论文设计 9 类任务；当前官方仓库明确发布 10K 图/7 类内容，其中 8.2K 图及其 previous frame 可下载、1.8K 留作 benchmark | 官方仓库链接 Google Drive；代码 MIT，**数据为 proprietary license**，二者不可混淆 | **首选真实图片源**；适合人、车、固定障碍和遮挡，停车位冲突覆盖不足 |
| FPD | 车载前/后/左/右 4 路 1920×1280 鱼眼，20 Hz；128 线 LiDAR 10 Hz | 是，且专门采集代客泊车场景 | 论文表中列出 420,000 张/条数据，3 城、100+ 停车场、400+ 视频、200+ 小时；8 类 2D/3D 框和深度 | 论文和论文落地页未给出数据下载入口或数据许可证 | **语义上最匹配，但当前不可执行**；可以联系作者申请访问 |
| AVM-SLAM | 4 路 1280×960 鱼眼 10 Hz；同步轮速计与 IMU；另有 1354×1632 AVM BEV 10 Hz | 是，地下车库真实连续环视 | 220 m × 110 m、430+ 车位的地下车库序列；主要面向 SLAM，没有对象或风险语义标签 | 官方页给出百度网盘；仅学术使用，页面声明 GPLv3，商业用途需联系作者 | **首选车库补充源**；适合弱光、窄通道、遮挡和连续帧，但所有风险标签要自建 |
| Tongji ps2.0 | 车载 4 路低成本鱼眼拼接成 600×600 AVM，覆盖约 10 m × 10 m | 是 AVM，但不是原始鱼眼 | 12,165 张：9,827 train、2,338 test；停车位标记点、入口线、方向和类型；VPS-Net 另发布空闲/非空闲标注 | 官方项目页/论文入口经百度网盘；未在公开落地页看到明确数据许可证，使用前需确认 | **车位冲突的最佳补充**；不适合评估原始鱼眼理解，也缺少通用障碍和驾驶建议标签 |
| SynWoodScape | CARLA 中复刻 WoodScape 的 4×190° 鱼眼，四向同步并带 BEV | 是，但为合成数据 | 论文计划发布 80,000 张、10+ 任务；公开研究常见版本为 2,000 张；提供语义/实例/运动分割、2D/3D 框、深度、光流、BEV、LiDAR 等 | 论文指向 WoodScape 网站；下载时需要接受当时的数据条款 | **仅作训练增强与角落案例**；不进入真实测试集，避免把仿真纹理表现当成实车质量 |
| nuScenes | 6 路校正后的车载相机 + 1 LiDAR + 5 radar，覆盖 360° | 车载环视，但不是近场鱼眼/AVM | 1,000 个 20 秒场景、约 140 万相机图；40k keyframes 上有 3D 框，另有 map、tracking、lidarseg/panoptic 等 | 注册账号并接受官方条款；非商业研究可免费，商业使用需另购许可证 | 适合车辆、VRU、交通锥和遮挡的辅助筛选；视角和行驶场景与泊车域有明显差距 |
| DDAD | 6 路 1936×1216 相机按 60° 间隔布置，10 Hz 360°；同步 LiDAR | 车载环视，但为长距针孔视觉 | train 150 scenes/12,650 frames/75,900 RGB；val 50/3,950/23,700；稠密深度为主，train+val 257 GB | 官方 S3 直接下载；CC BY-NC-SA 4.0 | 适合深度和 360° 几何辅助，不适合泊车域主数据，也没有风险语义标签 |
| FishEye8K | 18 台固定路侧鱼眼相机，不安装在车辆上 | 否 | 8,000 张、22 段视频、157k 个框；Bus/Bike/Car/Pedestrian/Truck 五类 | AI City Challenge 申请表和 Track 4 数据协议 | 可做鱼眼目标识别预训练；**不能作为项目质量测试集**，因为视点、尺度和任务都不同 |

## 逐项核对

### WoodScape

WoodScape 原始论文明确说明：四路车载鱼眼覆盖 360°，分辨率为 1 MP，视频为 30 FPS；论文设计包括语义/实例分割、2D/3D 目标框、深度、运动分割、相机污损、视觉里程计/SLAM 和端到端控制等任务。[原始论文](https://arxiv.org/abs/1905.01489)

但“论文描述的完整数据”不能等同于“当前公开下载内容”。Valeo 官方仓库当前明确列出 10K 图和 7 类内容：RGB、semantic segmentation、2D boxes、instance segmentation、motion segmentation、previous images、CAN、soiling 和 calibration；其中 8.2K 图及对应 previous frame 已发布，1.8K test samples 留作 benchmark。[WoodScape 官方仓库](https://github.com/valeoai/WoodScape)

Valeo 官方发布页将其定位为用于自动驾驶和停车研究的 surround-view fisheye 数据集，并给出 WoodScape 下载入口。[Valeo 官方发布页](https://www.valeo.com/en/valeo-releases-the-first-multitask-fisheye-camera-open-source-dataset-aiming-to-take-automated-driving-to-the-next-level/) [官方数据入口](https://woodscape.valeo.com/dataset)

适配性：

- `vru_near_maneuver_path`：有 pedestrian/cyclist 和运动标签，可筛选候选，但没有车辆规划轨迹；
- `vehicle_near_maneuver_path`：有 vehicle 2D/3D 框和深度，适合候选筛选；
- `fixed_obstacle_near_path`：分割类别和深度可辅助，但常见停车场柱、墙、路沿需要人工复核；
- `visibility_occlusion`：原始鱼眼、污损和遮挡场景有价值；
- `narrow_passage`、`parking_space_conflict`：没有直接标签，且并非专门停车数据，需人工筛选和标注。

不足：SynWoodScape 原始论文说明，为了增加帧多样性，WoodScape 并非所有时刻都同步标注四路相机；因此不能默认每个公开标注样本都能组成完整四相机同步输入。官方仓库明确区分“代码 MIT”和“数据 proprietary”，不能仅因官方称为“open-source dataset”就推断图片可自由再分发。

### FPD（Fisheye Parking Dataset）

FPD 原始论文的采集系统是前、后、左、右四路 1920×1280 鱼眼相机（20 Hz）和 128 线 LiDAR（10 Hz）。数据来自 3 个城市、100 多个停车场、400 多段视频，并人为布置 car meeting、car crossing、person circling 等泊车关键场景。标注覆盖 15 米范围内的 car、truck、pedestrian、rider、baby carriage、traffic cone、motorbike、no-stop sign，包含 2D 框、3D 框、朝向与深度。[FPD 原始论文](https://arxiv.org/pdf/2212.04111)

论文表 II 报告 train/validation/test 为 210,000/126,000/84,000，总计 420,000；同一论文又说明“一条 data 包含四张鱼眼图和一个点云”。由于作者在 `data` 与 `images` 用词上存在歧义，在拿到数据前不应进一步推断实际图片文件数。

适配性：六类事件中，VRU、车辆、固定障碍、狭窄空间和遮挡都能找到强候选，且停车域最接近项目。主要缺口仍然是无规划轨迹、无风险级别、无证据句和无驾驶建议。

获取边界：论文声称贡献 FPD，但原始论文、arXiv 关联资源和论文落地页均未给出数据下载链接或数据许可证。本项目应先联系论文作者或 ZongMu Technology，获得书面访问和使用范围，再考虑依赖该数据集。

### AVM-SLAM Dataset

官方项目页说明，数据由地下车库中的测试车采集：四路鱼眼为 10 Hz、1280×960，同时提供由 AVM 子系统生成的 10 Hz、1354×1632 BEV 序列、四路轮速和 IMU。车库约 220 m × 110 m，包含 430 多个停车位。[官方数据页](https://yale-cv.github.io/avm-slam_dataset/) [原始论文](https://arxiv.org/abs/2309.08180)

它的价值不在现有目标标注，而在于非常接近真实自动代客泊车：地下弱光、重复纹理、GPS 缺失、动态遮挡和连续行驶。可从视频中按序列抽帧，自建 `narrow_passage`、`visibility_occlusion`、固定障碍和负样本。

官方页面只承诺 SLAM 相关传感器数据，没有声明 2D/3D 对象框或风险标签。页面写明“only for academic use under GPLv3”，商业用途需联系作者；下载前仍应保存当时的页面和许可证副本。

### Tongji ps2.0 与 VPS-Net occupancy 扩展

ps2.0 原始工作使用安装在 SAIC Roewe E50 上的四路低成本鱼眼 AVM 系统，发布 12,165 张 600×600 的环视图，覆盖室内、白天、雨天、阴影、路灯和斜列车位；每张图约对应车辆周围 10 m × 10 m。训练/测试划分为 9,827/2,338。[DeepPS 原始论文](https://cslinzhang.github.io/home/files/parkingslot.pdf) [Tongji 官方项目页](https://cslinzhang.github.io/ps/)

原始 ps2.0 主要标注停车位标记点、入口线、方向和 perpendicular/parallel/slanted 等类型，并不包含车位占用。VPS-Net 作者随后公开了 ps2.0 和 PSV 的 vacant/non-vacant 扩展标注。[VPS-Net 官方仓库](https://github.com/weili1457355863/VPS-Net) [VPS-Net 原始论文](https://www.mdpi.com/1424-8220/20/7/2138)

适配性：`parking_space_conflict` 最强，也能挑选窄通道、阴影和雨天图片；但拼接 BEV 已经消除了大部分原始鱼眼外观，且停车位占用不等于“当前机动路径与车位冲突”。对象风险和驾驶建议仍需人工标注。

许可证边界：官方项目页提供下载，但页面没有展示明确的数据许可证；VPS-Net 仓库也不应被视为自动授予原始图片再分发权。适合作为内部学术研究输入，是否允许发布衍生标注或图片子集需另行确认。

### SynWoodScape

SynWoodScape 在 CARLA 中复刻四路 190° WoodScape 鱼眼布局，四路同步，提供 BEV。论文计划发布 80,000 张合成图片和 10 多类标签，包括语义/实例/运动分割、2D/3D 框、深度、光流、event signals、LiDAR、雷达、IMU/GNSS；每个样本还包含前一帧，采样率 10 FPS。[原始论文](https://arxiv.org/abs/2203.05056)

它可以定向生成行人绕车、车辆交汇、狭窄通道、低照度等稀缺案例，并自动获得几何真值。限制是论文场景主要来自 CARLA Town10 的城市道路，不等同于真实车库；域差异会让图像质量结论过度乐观。适合 LoRA 训练增强和标注工具测试，不适合作为冻结真实测试集。

### nuScenes

nuScenes 有 6 路车载相机、5 路 radar 和 1 路 LiDAR，传感器覆盖 360°；包含 1,000 个 20 秒场景、约 140 万相机图片、40,000 个关键帧上的 3D 框，并提供 23 类和 8 种属性。[官方数据说明](https://www.nuscenes.org/nuscenes) [原始论文](https://arxiv.org/abs/1903.11027) [官方 devkit 与下载说明](https://github.com/nutonomy/nuscenes-devkit)

nuScenes 相机图已校正，不是量产泊车鱼眼，也不是以停车场低速近场为主。它适合通过 3D 框、visibility、场景描述和地图自动筛选人、车、交通锥、遮挡候选，但最后必须人工确认是否满足泊车语义。

下载需要注册并同意官方条款。官方条款明确要求商业用途另购许可证；非商业研究可免费访问，具体许可文字应以下载时的当前条款为准。[官方使用条款](https://www.nuscenes.org/terms-of-use-commercial)

### DDAD

DDAD 配置 6 路 1936×1216 全局快门相机，按 60° 间隔布置并以 10 Hz 同步，从而覆盖 360°；主要任务是长距离稠密深度。训练集 150 scenes、12,650 帧、75,900 张 RGB，验证集 50 scenes、3,950 帧、23,700 张 RGB；train+val 下载约 257 GB。[DDAD 官方仓库与下载](https://github.com/TRI-ML/DDAD)

它不是鱼眼，也不是泊车数据，因此只适合深度、360° 视觉或普通城市障碍的辅助研究。官方仓库将数据置于 CC BY-NC-SA 4.0 下，不能把它作为不受限制的商业数据。

### FishEye8K

FishEye8K 有 8,000 张图片，来自台湾新竹 18 台固定路侧鱼眼相机的 22 段视频，分辨率为 1080×1080 或 1280×1280；包含约 157,000 个 Bus、Bike、Car、Pedestrian、Truck 框。[原始论文](https://openaccess.thecvf.com/content/CVPR2023W/AICity/html/Gochoo_FishEye8K_A_Benchmark_and_Dataset_for_Fisheye_Camera_Object_Detection_CVPRW_2023_paper.html)

AI City 官方页面给出的划分为 5,288 train 和 2,712 validation，并提供 VOC、COCO、YOLO 三种标签格式；获取需要填写申请表并接受 Track 4 数据协议。[AI City 官方数据页](https://www.aicitychallenge.org/2024-data-and-evaluation/) [Track 4 数据协议](https://www.aicitychallenge.org/wp-content/uploads/2024/03/DATASET_LICENSE_AGREEMENT_2024_Track4_v2.pdf)

虽然它是高质量鱼眼目标检测数据，但相机固定在道路基础设施上，物体尺度和空间关系与车身近场摄像头不同。因此只能用于辅助鱼眼目标识别，不应进入 ParkSight 质量基线。

## 六类事件的数据覆盖

下表表示“是否容易从该数据集中筛出可供人工标注的候选图片”，不表示数据集已经提供对应事件标签。

| 数据集 | VRU near path | Vehicle near path | Fixed obstacle | Narrow passage | Visibility occlusion | Parking-space conflict |
| --- | --- | --- | --- | --- | --- | --- |
| WoodScape | 强 | 强 | 中 | 中 | 强 | 弱 |
| FPD | 强 | 强 | 强 | 强 | 强 | 中 |
| AVM-SLAM | 中 | 中 | 强 | 强 | 强 | 中 |
| ps2.0/VPS | 弱 | 中 | 弱 | 中 | 中 | 强 |
| SynWoodScape | 强（合成） | 强（合成） | 中（合成） | 中（合成） | 中（合成） | 弱（合成） |
| nuScenes | 强 | 强 | 中 | 弱 | 中 | 弱 |
| DDAD | 中 | 强 | 中 | 弱 | 中 | 弱 |
| FishEye8K | 有对象但视点错误 | 有对象但视点错误 | 弱 | 弱 | 弱 | 无 |

这里的 `near path` 只有在附带机动方向或轨迹时才能形成可靠真值；否则最多说明“对象靠近车身/潜在运动区域”。

## 建议的高质量标注集构建方式

### 第一阶段：建立小型冻结测试集

先做 60 至 120 张真实图片，不立即追求大规模 LoRA 数据。建议结构：

- 每个事件至少 15 个正例；一个样本允许多个事件；
- 至少 20 个明确的低风险负例，防止模型“见物就报风险”；
- 室外白天、室外夜间、地下车库、雨天/反光、遮挡各有覆盖；
- 原始单路鱼眼、mosaic 或 AVM 只能选择一种作为第一版正式输入；
- 连续视频只稀疏抽帧，同一视频/行程全部使用同一个 `source_group_id`，不得跨 train/validation/test。

第一版建议来源比例：

| 来源 | 建议用途 | 建议数量 |
| --- | --- | --- |
| WoodScape | 人、车、近场障碍、遮挡、负例 | 40-70 |
| AVM-SLAM | 地下车库、弱光、窄通道、柱体/墙体 | 20-40 |
| ps2.0/VPS | 车位形状、占用、车位冲突 | 15-30 |

同一图片可以覆盖多个事件，因此总数不必等于各类别数量之和。

### 第二阶段：人工标注与复核

每个样本至少由两人独立标注，然后对分歧进行仲裁。标注时先记录可观察事实，再给风险结论：

1. 观察对象：人、非机动车、车辆、墙柱、路沿、车位边界；
2. 观察空间关系：相对车身方向、是否遮挡、通道余量、是否侵入已知轨迹；
3. 依据冻结规则选择 `events` 和 `risk_level`；
4. `evidence` 只描述图像可见事实，不猜测速度、意图或不可见区域；
5. `driver_advice` 必须从项目枚举中选择，且与风险事实匹配。

高质量标准不是“原数据集标签多”，而是：

- 输入视角和部署摄像头一致；
- 路径/机动上下文足以支持事件定义；
- 图片来源和许可可追溯；
- 同源序列不泄漏；
- 双人标注与仲裁有记录；
- 困难负例和遮挡样本充足；
- 每条 `evidence` 可以被图片直接核验。

### 第三阶段：再扩充 LoRA 数据

冻结测试集之后，再从上述来源和自采数据中扩大训练集。优先补充基线的真实错误，而不是按数据集原始类别平均抽样。合成数据只进入训练集，不进入验证/测试集；许可证不允许再分发的原图保留在 `data/raw/` 本地目录，仓库只提交 manifest、允许公开的衍生标注和数据来源说明。

## 推荐执行顺序

1. 决定第一版输入是“单路鱼眼”还是“AVM/mosaic”，并补充机动方向/轨迹定义；
2. 注册 WoodScape，先检查可下载子集、文件结构、同步相机信息和当前许可证；
3. 下载 AVM-SLAM 的一个短序列，验证四路鱼眼与 BEV 的时间对应关系；
4. 获取 ps2.0 与 VPS occupancy annotations，确认是否允许保存和发布衍生 JSON 标注；
5. 每个来源各选 5 至 10 张，先制作一个约 20 张的 pilot casebook；
6. 用当前 Transformers FP16 runtime 跑 pilot，检查 schema 有效率和主要语义错误，再决定正式 60 至 120 张冻结集的类别配额。
