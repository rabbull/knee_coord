# knee_coord

Python 工具库，用于分析膝关节运动学和接触行为，基于股骨和胫骨的 3D 模型以及运动数据生成动画、接触深度图、自由度曲线等结果。

## 功能特性

- 解析 CSV/JSON 运动学数据并计算骨骼坐标系与平移旋转
- 生成骨骼运动动画、接触区域和深度热图
- 输出自由度、接触面积、软骨厚度等曲线
- 支持自定义相机视角、深度方向及各类绘图参数

## 快速开始

### 1. 下载源码并进入工程
```bash
git clone https://github.com/rabbull/knee_coord.git
cd knee_coord
```

### 2. （可选）创建虚拟环境

使用 Virtual Env：
```bash
python -m virtualenv .venv
source .venv/bin/activate
```

或者使用 Conda
```bash
conda create -n knee
conda activate knee
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 准备配置文件

复制示例配置并根据需要修改模型路径、运动数据以及任务开关：

```bash
cp config.py.example config.py
```

### 5. 运行

```bash
python main.py
```

所有输出默认保存到 `output/` 目录，可在 `config.py` 中修改 `OUTPUT_DIRECTORY`。

## 配置说明

### 枚举类型说明

#### 1. `AnimationCameraDirection`
动画相机视角设置：
- `AUTO`：自动选择视角（未实现）
- `FIX_TIBIA_FRONT`：固定胫骨前视图
- `FIX_TIBIA_L2M`：固定胫骨从外侧向内看
- `FIX_TIBIA_M2L`：固定胫骨从内侧向外看

#### 2. `DepthDirection`
深度图深度计算方式：
- `Z_AXIS`：沿 z 轴方向
- `CONTACT_PLANE`：拟合接触面法线方向
- `VERTEX_NORMAL`：顶点法线方向（未实现）

#### 3. `MomentDataFormat`
运动数据文件格式：
- `CSV`：以逗号分隔的 CSV 文件
- `JSON`：JSON 格式文件

#### 4. `DofRotationMethod`
自由度（DOF）旋转计算方法：
- `EULER_XYZ` ~ `EULER_YXZ`：欧拉角旋转方式，不同顺序
- `PROJECTION`：使用投影法计算旋转角
- `JCS`：使用联合坐标系（Joint Coordinate System）

#### 5. `InterpolateMethod`
运动数据插值方法：
- `CubicSpline`：三次样条插值
- `Akima`：Akima 插值法
- `Pchip`：保形三次插值

#### 6. `BaseBone`
参考骨骼：
- `FEMUR`：股骨
- `TIBIA`：胫骨

#### 7. `KneeSide`
膝盖侧别：
- `LEFT`：左膝
- `RIGHT`：右膝

### 路径配置
```python
OUTPUT_DIRECTORY = 'output'
```
- `OUTPUT_DIRECTORY`：所有输出文件根目录。

### 各任务开关
```python
GENERATE_ANIMATION = True
GENERATE_DEPTH_CURVE = True
GENERATE_DEPTH_MAP = True
GENERATE_DOF_CURVES = True
```
- `GENERATE_ANIMATION`：是否生成动画。
- `GENERATE_DEPTH_CURVE`：是否生成接触深度曲线。
- `GENERATE_DEPTH_MAP`：是否生成深度热图。
- `GENERATE_DOF_CURVES`：是否生成自由度曲线。

### 膝关节模型配置

```python
KNEE_SIDE = KneeSide.LEFT
FEMUR_MODEL_FILE = 'archive/acc_task/Femur.stl'
FEMUR_CARTILAGE_MODEL_FILE = 'archive/acc_task/Femur_Cart_Smooth.stl'
TIBIA_MODEL_FILE = 'archive/acc_task/Tibia.stl'
TIBIA_CARTILAGE_MODEL_FILE = 'archive/acc_task/Tibia_Cart_Smooth.stl'
FEATURE_POINT_FILE = 'archive/acc_task/Coordination_Pt.txt'
IGNORE_CARTILAGE = False
```
- `KNEE_SIDE`：设置分析的是左膝还是右膝
- `FEMUR_MODEL_FILE`：股骨的 3D 模型文件路径
- `FEMUR_CARTILAGE_MODEL_FILE`：股骨软骨模型文件路径
- `TIBIA_MODEL_FILE`：胫骨的 3D 模型文件路径
- `TIBIA_CARTILAGE_MODEL_FILE`：胫骨软骨模型文件路径
- `FEATURE_POINT_FILE`：特征点数据文件路径
- `IGNORE_CARTILAGE`：是否忽略软骨数据

### 运动数据配置

```python
MOVEMENT_DATA_FORMAT = MomentDataFormat.CSV
MOVEMENT_DATA_FILE = 'archive/model_0313/First_Profile.csv'
MOVEMENT_PICK_FRAMES = None
MOVEMENT_SMOOTH = True
MOVEMENT_INTERPOLATE_METHOD = InterpolateMethod.CubicSpline
```
- `MOVEMENT_DATA_FORMAT`：运动数据的文件格式
- `MOVEMENT_DATA_FILE`：运动数据文件路径
- `MOVEMENT_PICK_FRAMES`：选择的帧索引列表（如`[1, 3, 7]`表示只取第 1、3、7 帧原始数据）；`None` 表示使用全部帧
- `MOVEMENT_SMOOTH`：是否对运动数据进行平滑处理
- `MOVEMENT_INTERPOLATE_METHOD`：平滑处理使用的插值方法

### 动画参数设置
```python
ANIMATION_BONE_COLOR_FEMUR = '#ffffff'
ANIMATION_BONE_COLOR_TIBIA = '#ffffff'
ANIMATION_CARTILAGE_COLOR_FEMUR = '#00e5ff'
ANIMATION_CARTILAGE_COLOR_TIBIA = '#8800ff'
ANIMATION_RESOLUTION = (1000, 1000)
ANIMATION_DIRECTION = AnimationCameraDirection.FIX_TIBIA_M2L
ANIMATION_LIGHT_INTENSITY = 3.0
ANIMATION_SHOW_BONE_COORDINATE = True
```
- `ANIMATION_BONE_COLOR_FEMUR`：股骨在动画中的颜色
- `ANIMATION_BONE_COLOR_TIBIA`：胫骨在动画中的颜色
- `ANIMATION_CARTILAGE_COLOR_FEMUR`：股骨软骨颜色
- `ANIMATION_CARTILAGE_COLOR_TIBIA`：胫骨软骨颜色
- `ANIMATION_RESOLUTION`：动画分辨率
- `ANIMATION_DIRECTION`：动画中相机的拍摄方向
- `ANIMATION_LIGHT_INTENSITY`：光照强度
- `ANIMATION_SHOW_BONE_COORDINATE`：是否显示骨骼坐标轴

### 深度图参数设置
```python
DEPTH_MAP_BONE_COLOR_FEMUR = '#ffffff'
DEPTH_MAP_BONE_COLOR_TIBIA = '#ffffff'
DEPTH_MAP_CARTILAGE_COLOR_FEMUR = '#1d16a1'
DEPTH_MAP_CARTILAGE_COLOR_TIBIA = '#1d16a1'
DEPTH_MAP_RESOLUTION = (1000, 1000)
DEPTH_MAP_LIGHT_INTENSITY = 3.0
DEPTH_DIRECTION = DepthDirection.Z_AXIS
DEPTH_BASE_BONE = BaseBone.TIBIA
DEPTH_MAP_MARK_MAX = True
DEPTH_MAP_DEPTH_THRESHOLD = 10
```
- `DEPTH_MAP_BONE_COLOR_FEMUR`：股骨在深度图中的颜色
- `DEPTH_MAP_BONE_COLOR_TIBIA`：胫骨在深度图中的颜色
- `DEPTH_MAP_CARTILAGE_COLOR_FEMUR`：股骨软骨在深度图中的颜色
- `DEPTH_MAP_CARTILAGE_COLOR_TIBIA`：胫骨软骨在深度图中的颜色
- `DEPTH_MAP_RESOLUTION`：深度图分辨率
- `DEPTH_MAP_LIGHT_INTENSITY`：深度图光照强度
- `DEPTH_DIRECTION`：计算深度的方向
- `DEPTH_BASE_BONE`：深度计算参考的骨骼
- `DEPTH_MAP_MARK_MAX`：是否在热图中标注最大深度点
- `DEPTH_MAP_DEPTH_THRESHOLD`：用于绘图的深度阈值，超过该值不绘制

### 自由度曲线参数设置
```python
DOF_ROTATION_METHOD = DofRotationMethod.JCS
DOF_BASE_BONE = BaseBone.FEMUR
```
- `DOF_ROTATION_METHOD`：计算自由度角度的方法
- `DOF_BASE_BONE`：自由度角度计算时的参考骨骼
