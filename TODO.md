# ComfyUI 自定义节点开发需求文档：Mask轮廓提取与可视化

## 1. 项目概述 (Project Overview)

**目标**：开发一个包含两个自定义节点的 ComfyUI 扩展包 (`ComfyUI_Mask_Utils`)。
**核心功能**：

1. **MaskToContourJSON**: 将输入的二值 Mask 图像转换为基于多边形轮廓点的 JSON 数据，支持多实例分离。
2. **ContourVisualizer**: 读取上述 JSON 数据和原始图像，将轮廓绘制在图像上进行预览，支持样式自定义。

**技术栈**：Python 3.10+, OpenCV (`cv2`), NumPy, PyTorch (ComfyUI 标准环境)。

---

## 2. 数据结构定义 (Data Schema)

### 2.1 输出 JSON 格式

`MaskToContourJSON` 节点生成的 JSON 必须遵循以下 Schema：

```json
{
  "instance_0": [[x1, y1], [x2, y2], ...],
  "instance_1": [[x1, y1], [x2, y2], ...],
  ...
}

```

* **Key**: 字符串，代表实例 ID (用户可定义前缀)。
* **Value**: 二维整数列表 `List[List[int]]`，代表轮廓顶点的像素坐标 `[x, y]`。
* **坐标系**: 绝对像素坐标 (非归一化)，原点在图像左上角。

---

## 3. 节点 1：Mask 转轮廓点 (MaskToContourJSON)

### 3.1 功能描述

接收 ComfyUI 的 `MASK` 类型输入，使用 OpenCV 提取所有独立的连通区域（实例），通过 Douglas-Peucker 算法进行轮廓近似（简化），最终输出 JSON 字符串并保存为文件。

### 3.2 输入参数配置 (INPUT_TYPES)

| 参数名 | 类型 | 默认值 | 范围/选项 | 说明 |
| --- | --- | --- | --- | --- |
| **mask** | `MASK` | - | - | 输入的二值遮罩 (Tensor: [B, H, W]) |
| **min_area** | `INT` | 100 | 0 ~ 10000 | 过滤噪点，面积小于此值的轮廓将被忽略 |
| **epsilon_factor** | `FLOAT` | 0.005 | 0.001 ~ 0.1 | 轮廓近似系数。值越小轮廓越精细，值越大轮廓越简略 |
| **instance_id_prefix** | `STRING` | "segment" | - | JSON key 的前缀 (如 "A" -> "A_0", "A_1") |
| **save_path** | `STRING` | "output/contours.json" | - | JSON 文件保存的绝对或相对路径 |

### 3.3 输出 (RETURN_TYPES)

1. `json_str` (`STRING`): JSON 数据的字符串表示，用于传递给下游节点。

### 3.4 核心逻辑 (Implementation Logic)

1. **Tensor 处理**:
* 检查输入 `mask` 的维度。如果是 Batch，默认处理第一张图 (`mask[0]`)。
* 将 Tensor 转换为 NumPy 数组，并缩放至 0-255，转换为 `np.uint8` 类型。


2. **轮廓提取**:
* 使用 `cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)`。
* `RETR_EXTERNAL` 确保只获取最外层轮廓，实现实例分离。


3. **遍历与处理**:
* 遍历所有 `contours`。
* 计算 `cv2.contourArea`，若小于 `min_area` 则跳过。
* 计算 `epsilon = epsilon_factor * cv2.arcLength`。
* 使用 `cv2.approxPolyDP` 获取近似轮廓。
* 将 numpy 坐标点转换为 Python list `[[x, y], ...]`。


4. **序列化**:
* 构建字典 `data = {f"{prefix}_{i}": points_list}`。
* 将字典 dump 为 JSON 字符串。
* 写入文件到 `save_path`。



---

## 4. 节点 2：轮廓可视化 (ContourVisualizer)

### 4.1 功能描述

接收原始图像和 JSON 字符串，将多边形轮廓绘制在图像上。

### 4.2 输入参数配置 (INPUT_TYPES)

| 参数名 | 类型 | 默认值 | 范围/选项 | 说明 |
| --- | --- | --- | --- | --- |
| **image** | `IMAGE` | - | - | 原始图像 (Tensor: [B, H, W, C]) |
| **json_str** | `STRING` | - | - | 上游节点输出的 JSON 字符串 |
| **line_thickness** | `INT` | 2 | 1 ~ 10 | 轮廓线宽度 (px) |
| **line_color_hex** | `STRING` | "#00FF00" | - | 轮廓颜色 (Hex 格式) |
| **draw_points** | `BOOLEAN` | False | - | 是否绘制顶点圆点 |
| **point_radius** | `INT` | 4 | 1 ~ 10 | 顶点圆点半径 (仅当 draw_points=True 时有效) |

### 4.3 输出 (RETURN_TYPES)

1. `image` (`IMAGE`): 绘制了轮廓的图像 Tensor。

### 4.4 核心逻辑 (Implementation Logic)

1. **图像转换**:
* ComfyUI 的 `IMAGE` 是 `[B, H, W, C]` 的 RGB float32 Tensor。
* 转换为 NumPy `uint8` 格式 (H, W, 3) 供 OpenCV 绘图。注意：OpenCV 使用 BGR，需用 `cv2.cvtColor` 转换颜色空间。


2. **JSON 解析**:
* 使用 `json.loads(json_str)` 解析数据。
* 异常处理：如果 JSON 格式错误，打印 Error 并返回原图。


3. **颜色解析**:
* 将 Hex 字符串 (如 `#FF0000`) 解析为 RGB 元组，再转换为 OpenCV 需要的 BGR 元组。


4. **绘图**:
* 遍历 JSON 中的所有实例。
* 将点集转换为 `np.array(points, dtype=np.int32)`。
* 使用 `cv2.polylines(img, [pts], isClosed=True, ...)` 绘制线段。
* (可选) 如果 `draw_points` 为真，遍历每个点使用 `cv2.circle` 绘制顶点。


5. **输出转换**:
* 将 OpenCV 的 BGR 图像转回 RGB。
* 归一化为 0.0-1.0 float32。
* 转回 Torch Tensor `[1, H, W, C]`。



---

## 5. 边缘情况与异常处理 (Edge Cases)

1. **空 Mask**: 如果 Mask 全黑，生成的 JSON 应为空对象 `{}`，代码不应崩溃。
2. **图像尺寸不匹配**: 如果 JSON 中的坐标超出了 `ContourVisualizer` 输入图像的尺寸，OpenCV 会自动裁剪，无需报错，但应确保代码逻辑允许这种情况。
3. **文件路径**: 确保 `save_path` 所在的目录如果不存在，代码会自动创建 (`os.makedirs`)。

---

## 6. 代码结构模版 (Code Structure Template)

请基于以下类结构生成 Python 代码：

```python
import torch
import numpy as np
import cv2
import json
import os

class MaskToContourJSON:
    @classmethod
    def INPUT_TYPES(s):
        # Implementation here
        pass
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("json_str",)
    FUNCTION = "process"
    CATEGORY = "Custom/MaskUtils"

    def process(self, mask, min_area, epsilon_factor, instance_id_prefix, save_path):
        # Implementation here
        pass

class ContourVisualizer:
    @classmethod
    def INPUT_TYPES(s):
        # Implementation here
        pass
        
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "visualize"
    CATEGORY = "Custom/MaskUtils"

    def visualize(self, image, json_str, line_thickness, line_color_hex, draw_points, point_radius):
        # Implementation here
        pass

# 节点映射
NODE_CLASS_MAPPINGS = { ... }
NODE_DISPLAY_NAME_MAPPINGS = { ... }

```