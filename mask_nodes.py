"""
ComfyUI 自定义节点：Mask 轮廓提取与 JSON 转换
将 Mask 图像转换为 JSON 轮廓数据，并支持可视化预览

作者：ComfyUI-Mask2JSON
依赖：torch, numpy, opencv-python
"""

import torch
import numpy as np
import cv2
import json
import os
from typing import List, Tuple, Dict


# class MaskToContourJSON:
#     """
#     Mask 转轮廓点 JSON 节点

#     功能：将输入的二值 Mask 图像转换为基于多边形轮廓点的 JSON 数据
#     特性：
#       - 支持多实例分离（自动识别独立连通区域）
#       - 使用 Douglas-Peucker 算法进行轮廓近似（简化顶点数量）
#       - 支持按面积过滤噪点
#       - 自动保存 JSON 文件到指定路径
#     """

#     @classmethod
#     def INPUT_TYPES(cls):
#         """
#         定义节点的输入参数配置

#         Returns:
#             dict: 包含 required（必填）和 optional（可选）参数的字典
#         """
#         return {
#             "required": {
#                 # ========== 必填参数 ==========

#                 # mask: 输入的二值遮罩图像
#                 # 类型：ComfyUI MASK 类型（torch.Tensor）
#                 # 形状：[Batch, Height, Width] 或 [Height, Width]
#                 # 值范围：0.0（黑色/背景）到 1.0（白色/前景）
#                 # 说明：节点会自动提取其中所有独立的白色连通区域作为轮廓
#                 "mask": ("MASK",),

#                 # min_area: 最小轮廓面积阈值
#                 # 类型：整数
#                 # 默认值：100（像素）
#                 # 范围：0 ~ 10000
#                 # 用法：小于此面积的轮廓会被视为噪点而忽略
#                 # 建议：
#                 #   - 对于精细分割：设置为 50-200
#                 #   - 对于粗略分割：设置为 500-2000
#                 #   - 设置为 0 则保留所有轮廓
#                 "min_area": ("INT", {
#                     "default": 100,
#                     "min": 0,
#                     "max": 10000,
#                     "step": 10
#                 }),

#                 # simplify_contours: 是否简化轮廓（Douglas-Peucker 算法）
#                 # 类型：布尔值
#                 # 默认值：True（启用简化）
#                 # 说明：
#                 #   - True: 使用 Douglas-Peucker 算法简化轮廓，减少顶点数量
#                 #          JSON 文件更小，但轮廓可能与 Mask 边缘略有偏差
#                 #   - False: 使用原始轮廓点，完全贴合 Mask 边缘
#                 #           顶点数量更多，JSON 文件更大，但精度最高
#                 # 建议：需要精确贴合 Mask 时设置为 False
#                 "simplify_contours": ("BOOLEAN", {
#                     "default": True
#                 }),

#                 # epsilon_factor: 轮廓近似系数（Douglas-Peucker 算法参数）
#                 # 类型：浮点数
#                 # 默认值：0.005
#                 # 范围：0.001 ~ 0.1
#                 # 说明：该值乘以轮廓周长得到近似精度 epsilon
#                 #       仅在 simplify_contours=True 时生效
#                 # 用法：
#                 #   - 值越小（如 0.001）：轮廓越精细，保留更多顶点，JSON 文件更大
#                 #   - 值越大（如 0.05）：轮廓越简略，顶点更少，JSON 文件更小
#                 #   - 推荐值：0.005（平衡精度与文件大小）
#                 "epsilon_factor": ("FLOAT", {
#                     "default": 0.005,
#                     "min": 0.001,
#                     "max": 0.1,
#                     "step": 0.001
#                 }),

#                 # contour_offset: 轮廓偏移量（收缩/扩张）
#                 # 类型：浮点数
#                 # 默认值：0.0（不偏移）
#                 # 范围：-20.0 ~ 20.0（像素）
#                 # 说明：
#                 #   - 负值：轮廓向中心收缩（如 -5.0 表示向内收缩 5 像素）
#                 #   - 正值：轮廓向外扩张（如 5.0 表示向外扩张 5 像素）
#                 #   - 0：保持原始轮廓
#                 "contour_offset": ("FLOAT", {
#                     "default": 0.0,
#                     "min": -20.0,
#                     "max": 20.0,
#                     "step": 0.5
#                 }),

#                 # instance_id_prefix: 实例 ID 前缀
#                 # 类型：字符串
#                 # 默认值："segment"
#                 # 说明：用于生成 JSON 数据中的键名（key）
#                 # 示例：
#                 #   - 前缀 "segment" -> 生成 "segment_0", "segment_1", ...
#                 #   - 前缀 "person" -> 生成 "person_0", "person_1", ...
#                 #   - 前缀 "" -> 生成 "0", "1", ...
#                 "instance_id_prefix": ("STRING", {
#                     "default": "segment",
#                     "multiline": False
#                 }),

#                 # save_path: JSON 文件保存路径
#                 # 类型：字符串
#                 # 默认值："output/contours.json"
#                 # 说明：
#                 #   - 支持绝对路径（如 "D:/data/contours.json"）
#                 #   - 支持相对路径（相对于 ComfyUI 启动目录）
#                 #   - 如果目录不存在，会自动创建
#                 "save_path": ("STRING", {
#                     "default": "output/contours.json",
#                     "multiline": False
#                 }),
#             }
#         }

#     # ========== 节点输出配置 ==========
#     RETURN_TYPES = ("STRING",)           # 输出类型：字符串（JSON 格式）
#     RETURN_NAMES = ("json_str",)          # 输出端口显示名称
#     FUNCTION = "process"                  # 执行函数名称
#     CATEGORY = "Mask2JSON/Conversion"     # 节点在菜单中的分类路径

#     def process(
#         self,
#         mask: torch.Tensor,
#         min_area: int,
#         simplify_contours: bool,
#         epsilon_factor: float,
#         contour_offset: float,
#         instance_id_prefix: str,
#         save_path: str
#     ) -> Tuple[str]:
#         """
#         节点执行函数：处理 Mask 并提取轮廓

#         Args:
#             mask: 输入的 Mask Tensor
#             min_area: 最小面积阈值（过滤噪点）
#             simplify_contours: 是否简化轮廓（False 则完全贴合 Mask 边缘）
#             epsilon_factor: 轮廓近似系数（仅在 simplify_contours=True 时生效）
#             contour_offset: 轮廓偏移量（负值收缩，正值扩张）
#             instance_id_prefix: JSON 键名前缀
#             save_path: JSON 文件保存路径

#         Returns:
#             Tuple[str]: 包含 JSON 字符串的元组（ComfyUI 要求返回元组）
#         """
#         # ==================== 第 1 步：Tensor 维度处理 ====================
#         # ComfyUI 的 MASK 可能是 3D [B, H, W] 或 2D [H, W]
#         if mask.dim() == 3:
#             # Batch 处理：默认取第一张图
#             # TODO: 未来可扩展为支持 batch 输出多个 JSON
#             mask_tensor = mask[0]
#         elif mask.dim() == 2:
#             # 单张 Mask，直接使用
#             mask_tensor = mask
#         else:
#             raise ValueError(f"不支持的 Mask 维度: {mask.dim()}，期望 2 或 3")

#         # 确保 Tensor 在 CPU 上（NumPy 转换需要）
#         mask_tensor = mask_tensor.cpu()

#         # 转换数据格式：torch.Tensor (0.0-1.0 float32) -> numpy.ndarray (0-255 uint8)
#         # OpenCV 的 findContours 函数要求输入为 uint8 类型
#         mask_np = (mask_tensor.numpy() * 255).astype(np.uint8)

#         # ==================== 第 2 步：轮廓提取 ====================
#         # cv2.findContours: 从二值图像中提取轮廓
#         # 参数说明：
#         #   - image: 输入的二值图像（会被函数修改，如需保留原图需先复制）
#         #   - mode: RETR_EXTERNAL 只获取最外层轮廓（实现实例分离）
#         #           其他选项：RETR_TREE（获取层级关系）、RETR_LIST（获取所有轮廓）
#         #   - method: CHAIN_APPROX_SIMPLE 压缩水平、垂直、对角线段，只保留端点
#         #             其他选项：CHAIN_APPROX_NONE（保留所有轮廓点）
#         contours, _ = cv2.findContours(
#             mask_np,
#             cv2.RETR_EXTERNAL,     # 只获取最外层轮廓，实现实例分离
#             cv2.CHAIN_APPROX_SIMPLE  # 简化轮廓，节省内存
#         )

#         # ==================== 第 3 步：遍历并处理每个轮廓 ====================
#         result_dict = {}  # 存储最终结果：{"prefix_0": [[x,y],...], "prefix_1": [...]}

#         for i, contour in enumerate(contours):
#             # --- 计算轮廓面积，过滤噪点 ---
#             area = cv2.contourArea(contour)
#             if area < min_area:
#                 # 面积太小，跳过此轮廓
#                 continue

#             # --- 根据参数选择轮廓处理方式 ---
#             if simplify_contours:
#                 # 使用 Douglas-Peucker 算法简化轮廓，减少顶点数量
#                 # arcLength: 计算轮廓周长
#                 perimeter = cv2.arcLength(contour, closed=True)
#                 # epsilon: 近似精度，值越大简化程度越高
#                 epsilon = epsilon_factor * perimeter
#                 # approxPolyDP: 执行轮廓近似
#                 processed_contour = cv2.approxPolyDP(contour, epsilon, closed=True)
#             else:
#                 # 使用原始轮廓点，完全贴合 Mask 边缘
#                 # CHAIN_APPROX_SIMPLE 已经去除了共线点，但保留了所有关键拐点
#                 processed_contour = contour

#             # --- 轮廓收缩/扩张处理（如果 contour_offset != 0） ---
#             if abs(contour_offset) > 0.001:
#                 # 计算轮廓质心（使用图像矩）
#                 M = cv2.moments(processed_contour)
#                 if M["m00"] != 0:  # 确保质心计算成功
#                     cx = int(M["m10"] / M["m00"])
#                     cy = int(M["m01"] / M["m00"])

#                     # 对每个点应用偏移
#                     adjusted_contour = []
#                     for pt in processed_contour.reshape(-1, 2):
#                         x, y = int(pt[0]), int(pt[1])

#                         # 计算从质心到点的向量
#                         dx, dy = x - cx, y - cy
#                         dist = (dx * dx + dy * dy) ** 0.5

#                         if dist > 0:
#                             # 归一化方向向量并应用偏移
#                             nx, ny = dx / dist, dy / dist
#                             new_x = int(round(x + nx * contour_offset))
#                             new_y = int(round(y + ny * contour_offset))
#                         else:
#                             # 点与质心重合，保持不变
#                             new_x, new_y = x, y

#                         adjusted_contour.append([[new_x, new_y]])

#                     processed_contour = np.array(adjusted_contour, dtype=np.int32)

#             # --- 转换为 Python 列表格式 ---
#             # contour 的形状是 (N, 1, 2)，需要 reshape 为 (N, 2)
#             # 然后转换为标准的 Python list: [[x, y], [x, y], ...]
#             points_list = processed_contour.reshape(-1, 2).tolist()
#             # 确保坐标为整数（像素坐标必须是整数）
#             points_list = [[int(x), int(y)] for x, y in points_list]

#             # --- 构建 JSON 的 key-value ---
#             # key 格式："{前缀}_{索引}"，例如 "segment_0"
#             key = f"{instance_id_prefix}_{i}"
#             result_dict[key] = points_list

#         # ==================== 第 4 步：序列化为 JSON ====================
#         # indent=2: 美化输出，便于人类阅读
#         # ensure_ascii=False: 支持中文字符（虽然当前 key 是英文）
#         json_str = json.dumps(result_dict, indent=2, ensure_ascii=False)

#         # ==================== 第 5 步：保存到文件 ====================
#         # 提取目录路径
#         save_dir = os.path.dirname(save_path)
#         # 如果目录不存在，自动创建（exist_ok=True 避免已存在时报错）
#         if save_dir and not os.path.exists(save_dir):
#             os.makedirs(save_dir, exist_ok=True)

#         # 写入文件（使用 UTF-8 编码支持中文）
#         with open(save_path, 'w', encoding='utf-8') as f:
#             f.write(json_str)

#         # 打印处理结果（便于调试）
#         print(f"[MaskToContourJSON] 提取到 {len(result_dict)} 个轮廓，已保存至 {save_path}")

#         # 返回元组（ComfyUI 要求）
#         return (json_str,)

class MaskToContourJSON:
    """
    Mask 转轮廓点 JSON 节点 (已优化小物体点密度问题)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),

                "min_area": ("INT", {
                    "default": 100,
                    "min": 0,
                    "max": 10000,
                    "step": 10
                }),

                "max_area": ("INT", {
                    "default": 2000,
                    "min": 0,
                    "max": 10000,
                    "step": 10
                }),

                "simplify_contours": ("BOOLEAN", {
                    "default": True
                }),

                "epsilon_factor": ("FLOAT", {
                    "default": 0.005,
                    "min": 0.001,
                    "max": 0.1,
                    "step": 0.001
                }),

                # ========== 【优化新增参数】 ==========
                # min_epsilon_abs: Epsilon 的绝对最小值（像素）
                # 作用：防止小物体因周长短导致 epsilon 过小，从而保留了像素锯齿。
                # 建议：设置为 2.0 ~ 5.0，能有效让小物体轮廓变平滑，减少点数。
                "min_epsilon_abs": ("FLOAT", {
                    "default": 3.0,   
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.5
                }),
                # ====================================

                "contour_offset": ("FLOAT", {
                    "default": 0.0,
                    "min": -20.0,
                    "max": 20.0,
                    "step": 0.5
                }),

                "instance_id_prefix": ("STRING", {
                    "default": "segment",
                    "multiline": False
                }),

                "save_path": ("STRING", {
                    "default": "output/contours.json",
                    "multiline": False
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("json_str",)
    FUNCTION = "process"
    CATEGORY = "Mask2JSON/Conversion"

    def process(
        self,
        mask,
        min_area,
        max_area,
        simplify_contours,
        epsilon_factor,
        min_epsilon_abs,  # 新增接收参数
        contour_offset,
        instance_id_prefix,
        save_path
    ):
        # 1. Tensor 处理
        if mask.dim() == 3:
            mask_tensor = mask[0]
        elif mask.dim() == 2:
            mask_tensor = mask
        else:
            raise ValueError(f"不支持的 Mask 维度: {mask.dim()}")

        mask_tensor = mask_tensor.cpu()
        mask_np = (mask_tensor.numpy() * 255).astype(np.uint8)

        # 2. 轮廓提取
        contours, _ = cv2.findContours(
            mask_np,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        result_dict = {}

        for i, contour in enumerate(contours):
            # 过滤噪点
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue

            processed_contour = contour

            # 3. 轮廓简化 (重点优化区域)
            if simplify_contours:
                perimeter = cv2.arcLength(contour, closed=True)
                
                # --- 【核心修改逻辑】 ---
                # 计算基于周长的动态 epsilon
                dynamic_epsilon = epsilon_factor * perimeter
                
                # 取 动态值 与 绝对最小值 中的较大者
                # 对于大物体：dynamic_epsilon 通常较大，起主导作用
                # 对于小物体：dynamic_epsilon 很小（如0.1），min_epsilon_abs (3.0) 强制生效
                # 这样就忽略了小物体边缘 < 3px 的像素阶梯
                final_epsilon = max(dynamic_epsilon, min_epsilon_abs)
                # ---------------------
                
                processed_contour = cv2.approxPolyDP(contour, final_epsilon, closed=True)
            else:
                processed_contour = cv2.approxPolyDP(contour, min_epsilon_abs, closed=True)

            # 4. 轮廓偏移 (保持你原有的逻辑)
            if abs(contour_offset) > 0.001:
                M = cv2.moments(processed_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    adjusted_contour = []
                    # 注意：这里 reshape 确保兼容性
                    pts = processed_contour.reshape(-1, 2)
                    
                    for pt in pts:
                        x, y = int(pt[0]), int(pt[1])
                        dx, dy = x - cx, y - cy
                        dist = (dx * dx + dy * dy) ** 0.5

                        if dist > 0:
                            nx, ny = dx / dist, dy / dist
                            new_x = int(round(x + nx * contour_offset))
                            new_y = int(round(y + ny * contour_offset))
                        else:
                            new_x, new_y = x, y
                        
                        adjusted_contour.append([[new_x, new_y]])
                    
                    if len(adjusted_contour) > 0:
                         processed_contour = np.array(adjusted_contour, dtype=np.int32)

            # 5. 格式转换
            points_list = processed_contour.reshape(-1, 2).tolist()
            
            # 再次检查点数，防止简化过度导致点数不足3个（无法构成多边形）
            if len(points_list) < 3:
                continue
                
            points_list = [[int(x), int(y)] for x, y in points_list]

            key = f"{instance_id_prefix}_{i}"
            result_dict[key] = points_list

        # 6. 保存与输出
        json_str = json.dumps(result_dict, indent=2, ensure_ascii=False)
        
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(json_str)

        print(f"[MaskToContourJSON] Saved {len(result_dict)} instances.")
        return (json_str,)

class ContourVisualizer:
    """
    轮廓可视化节点

    功能：读取 JSON 轮廓数据和原始图像，将轮廓绘制在图像上进行预览
    特性：
      - 支持自定义轮廓颜色（Hex 格式）
      - 支持调整线宽
      - 可选择绘制顶点圆点
    """

    @classmethod
    def INPUT_TYPES(cls):
        """
        定义节点的输入参数配置
        """
        return {
            "required": {
                # ========== 必填参数 ==========

                # image: 输入的原始图像
                # 类型：ComfyUI IMAGE 类型（torch.Tensor）
                # 形状：[Batch, Height, Width, Channels]
                # 颜色：RGB 格式，值范围 0.0-1.0
                # 说明：轮廓将被绘制在此图像上
                "image": ("IMAGE",),

                # line_thickness: 轮廓线宽度
                # 类型：整数
                # 默认值：2
                # 范围：1 ~ 10
                # 单位：像素（px）
                # 说明：线条越粗越明显，但可能遮挡更多图像内容
                "line_thickness": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 10,
                    "step": 1
                }),
            },
            "optional": {
                # ========== 可选参数 ==========

                # line_color_hex: 轮廓颜色（Hex 格式）
                # 类型：字符串
                # 默认值："#00FF00"（绿色）
                # 格式支持：
                #   - 6 位 Hex：#RRGGBB（如 #FF0000 红色）
                #   - 3 位简写：#RGB（如 #F00 红色）
                # 常用颜色：
                #   - #FF0000: 红色
                #   - #00FF00: 绿色（默认）
                #   - #0000FF: 蓝色
                #   - #FFFF00: 黄色
                #   - #FF00FF: 品红
                #   - #00FFFF: 青色
                "line_color_hex": ("STRING", {
                    "default": "#00FF00"
                }),

                # point_color_hex: 轮廓顶点圆点颜色（Hex 格式）
                # 类型：字符串
                # 默认值："#FF0000"（红色）
                # 格式支持：#RRGGBB 或 #RGB
                # 说明：仅在 draw_points=True 时生效，与轮廓线颜色区分
                "point_color_hex": ("STRING", {
                    "default": "#FF0000"
                }),

                # json_str: 上游节点输出的 JSON 字符串
                # 类型：字符串
                # 默认值："{}"（空 JSON，用于测试）
                # 格式：{"instance_0": [[x1,y1], [x2,y2], ...], ...}
                # 说明：通常来自 MaskToContourJSON 节点的输出
                "json_str": ("STRING", {
                    "default": "{}",
                    "multiline": True  # 允许多行输入，便于粘贴 JSON
                }),

                # draw_points: 是否绘制顶点圆点
                # 类型：布尔值
                # 默认值：False（不绘制）
                # 说明：启用后会在每个轮廓顶点位置绘制实心圆点
                # 用途：便于观察轮廓简化的效果，对比顶点数量
                "draw_points": ("BOOLEAN", {
                    "default": False
                }),

                # point_radius: 顶点圆点半径
                # 类型：整数
                # 默认值：4
                # 范围：1 ~ 10
                # 单位：像素（px）
                # 说明：仅在 draw_points=True 时生效
                "point_radius": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 10,
                    "step": 1
                }),
            }
        }

    # ========== 节点输出配置 ==========
    RETURN_TYPES = ("IMAGE",)             # 输出类型：图像 Tensor
    RETURN_NAMES = ("image",)              # 输出端口显示名称
    FUNCTION = "visualize"                 # 执行函数名称
    CATEGORY = "Mask2JSON/Visualization"   # 节点在菜单中的分类路径

    def visualize(
        self,
        image: torch.Tensor,
        line_thickness: int,
        line_color_hex: str = "#00FF00",
        point_color_hex: str = "#FF0000",
        json_str: str = "{}",
        draw_points: bool = False,
        point_radius: int = 4
    ) -> Tuple[torch.Tensor]:
        """
        节点执行函数：将 JSON 中的轮廓绘制到图像上

        Args:
            image: 输入的图像 Tensor
            line_thickness: 轮廓线宽度
            line_color_hex: 轮廓线颜色（Hex，默认绿色）
            point_color_hex: 轮廓点颜色（Hex，默认红色）
            json_str: JSON 格式的轮廓数据（默认空 JSON）
            draw_points: 是否绘制顶点圆点
            point_radius: 顶点圆点半径

        Returns:
            Tuple[torch.Tensor]: 包含绘制后图像的元组
        """
        # ==================== 第 1 步：图像格式转换 ====================
        # ComfyUI IMAGE: [B, H, W, C] RGB float32 (0.0-1.0)
        # OpenCV 需要: [H, W, C] BGR uint8 (0-255)

        # 处理维度：如果 Batch > 1，取第一张图
        if image.dim() == 4:
            image_tensor = image[0]
        else:
            image_tensor = image

        # 确保 Tensor 在 CPU 上
        image_tensor = image_tensor.cpu()

        # 转换数据格式：0.0-1.0 float32 -> 0-255 uint8
        img_np = (image_tensor.numpy() * 255).astype(np.uint8)

        # 颜色空间转换：RGB -> BGR（OpenCV 默认使用 BGR）
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # ==================== 第 2 步：JSON 解析 ====================
        try:
            contour_data = json.loads(json_str)
        except json.JSONDecodeError as e:
            # JSON 格式错误，打印错误信息并返回原图
            print(f"[ContourVisualizer] JSON 解析错误: {e}，返回原图")
            return (image,)

        # 检查是否为空数据
        if not contour_data:
            print("[ContourVisualizer] 轮廓数据为空，返回原图")
            return (image,)

        # ==================== 第 3 步：颜色解析 ====================
        # 解析轮廓线颜色
        try:
            line_color_rgb = self._hex_to_rgb(line_color_hex)
            # RGB -> BGR：交换 R 和 B 的位置
            line_color_bgr = (line_color_rgb[2], line_color_rgb[1], line_color_rgb[0])
        except ValueError as e:
            # 颜色格式错误，使用默认绿色
            print(f"[ContourVisualizer] 无效的轮廓线颜色: {e}，使用默认绿色")
            line_color_bgr = (0, 255, 0)  # BGR 格式的绿色

        # 解析轮廓点颜色
        try:
            point_color_rgb = self._hex_to_rgb(point_color_hex)
            # RGB -> BGR：交换 R 和 B 的位置
            point_color_bgr = (point_color_rgb[2], point_color_rgb[1], point_color_rgb[0])
        except ValueError as e:
            # 颜色格式错误，使用默认红色
            print(f"[ContourVisualizer] 无效的轮廓点颜色: {e}，使用默认红色")
            point_color_bgr = (0, 0, 255)  # BGR 格式的红色

        # ==================== 第 4 步：绘制轮廓 ====================
        for instance_name, points in contour_data.items():
            # 跳过空轮廓
            if not points:
                continue

            # 将点列表转换为 NumPy 数组，dtype 必须为 int32
            pts = np.array(points, dtype=np.int32)

            # --- 绘制多边形轮廓线 ---
            # cv2.polylines: 绘制一条或多条多边形曲线
            # 参数说明：
            #   - img: 目标图像
            #   - [pts]: 轮廓点列表（注意是列表的列表）
            #   - isClosed: True 表示闭合多边形（首尾相连）
            #   - color: BGR 颜色元组
            #   - thickness: 线宽
            cv2.polylines(
                img_bgr,
                [pts],
                isClosed=True,
                color=line_color_bgr,
                thickness=line_thickness
            )

            # --- 可选：绘制顶点圆点 ---
            if draw_points:
                for pt in pts:
                    # cv2.circle: 绘制实心圆
                    # thickness=-1 表示填充圆
                    cv2.circle(
                        img_bgr,
                        tuple(pt),            # 圆心坐标（需转为元组）
                        radius=point_radius,   # 半径
                        color=point_color_bgr, # 使用独立的点颜色
                        thickness=-1           # -1 表示填充
                    )

        # ==================== 第 5 步：输出转换 ====================
        # BGR -> RGB（转回 ComfyUI 的颜色格式）
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # 归一化：0-255 uint8 -> 0.0-1.0 float32
        img_normalized = img_rgb.astype(np.float32) / 255.0

        # 转回 Torch Tensor: [H, W, C] -> [1, H, W, C]
        img_tensor = torch.from_numpy(img_normalized).unsqueeze(0)

        return (img_tensor,)

    @staticmethod
    def _hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
        """
        将 Hex 颜色字符串解析为 RGB 元组

        支持格式：
          - #RRGGBB: 标准的 6 位 Hex（如 #FF0000）
          - #RGB: 3 位简写（如 #F00，自动展开为 #FF0000）

        Args:
            hex_color: Hex 颜色字符串（带或不带 # 前缀）

        Returns:
            Tuple[int, int, int]: (R, G, B) 元组，每个值范围 0-255

        Raises:
            ValueError: 当格式不正确时抛出异常
        """
        # 去除首尾空格
        hex_color = hex_color.strip()

        # 检查是否以 # 开头
        if not hex_color.startswith('#'):
            raise ValueError(f"Hex 颜色必须以 '#' 开头: {hex_color}")

        # 去掉 # 前缀
        hex_color = hex_color[1:]

        # 处理 3 位简写格式（如 #F00 -> #FF0000）
        if len(hex_color) == 3:
            # 每个字符重复一次：F00 -> FF0000
            hex_color = ''.join([c * 2 for c in hex_color])
        elif len(hex_color) != 6:
            # 既不是 3 位也不是 6 位，格式错误
            raise ValueError(f"无效的 Hex 颜色长度: {hex_color}，期望 3 或 6 位")

        # 解析 RGB 值（16 进制转 10 进制）
        try:
            r = int(hex_color[0:2], 16)  # 红色分量
            g = int(hex_color[2:4], 16)  # 绿色分量
            b = int(hex_color[4:6], 16)  # 蓝色分量
        except ValueError:
            raise ValueError(f"无效的 Hex 颜色值: {hex_color}")

        return (r, g, b)
