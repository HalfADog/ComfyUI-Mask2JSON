# ComfyUI 自定义节点开发规范指南

本指南旨在说明开发 ComfyUI 插件（Custom Nodes）时需要遵循的标准结构、核心规则以及最佳实践。

## 1. 目录结构规范

为了确保插件能被 ComfyUI 正确识别并兼容 **ComfyUI-Manager**，建议采用以下结构：

```text
comfyui-your-plugin-name/
├── __init__.py             # 核心入口，负责注册节点
├── your_logic_code.py      # 存放具体的 Python 类和业务逻辑
├── requirements.txt        # 列出运行所需的依赖库
├── pyproject.toml          # (可选) 现代 Python 项目配置
├── LICENSE                 # 开源协议（建议 MIT 或 GPL）
├── js/                     # (可选) 存放前端 JavaScript 扩展
└── web/                    # (可选) 存放自定义 UI 组件

```

---

## 2. 核心入口：`__init__.py`

ComfyUI 在启动时会扫描每个文件夹下的 `__init__.py`。你必须在此文件中导出两个核心字典：

* **`NODE_CLASS_MAPPINGS`**: 将“内部类名”映射到“类对象”。
* **`NODE_DISPLAY_NAME_MAPPINGS`**: 将“内部类名”映射到“UI 显示名称”。

```python
from .your_logic_code import MyNodeClass

NODE_CLASS_MAPPINGS = {
    "MyUniqueNodeName": MyNodeClass
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MyUniqueNodeName": "🌟 我的自定义节点"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

```

---

## 3. 节点类结构规范

一个标准的节点类必须包含以下四个核心要素：

### 3.1 定义输入 (`INPUT_TYPES`)

必须是一个类方法，返回一个包含 `required` (必填), `optional` (可选), 或 `hidden` (隐藏) 参数的字典。

```python
@classmethod
def INPUT_TYPES(s):
    return {
        "required": {
            "image": ("IMAGE",),  # 大写表示数据类型
            "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            "mode": (["Standard", "Advanced"],), # 下拉菜单
        },
    }

```

### 3.2 定义输出与分类

* **`RETURN_TYPES`**: 元组，定义输出的数据类型。
* **`RETURN_NAMES`**: (可选) 定义 UI 上输出口的标签。
* **`FUNCTION`**: 指定执行逻辑的函数名。
* **`CATEGORY`**: 节点在右键菜单中的路径。

```python
RETURN_TYPES = ("IMAGE", "MASK")
RETURN_NAMES = ("图像", "掩码")
FUNCTION = "process"
CATEGORY = "MyPlugin/ImageProcessing"

```

### 3.3 执行逻辑 (`FUNCTION`)

函数参数名必须与 `INPUT_TYPES` 中定义的 Key 完全一致。**返回值必须是一个元组**（即使只有一个返回值）。

---

## 4. 数据类型标准

在编写逻辑时，务必遵循 ComfyUI 的标准数据交换格式：

| 类型 | Python 对象类型 | 说明 |
| --- | --- | --- |
| **IMAGE** | `torch.Tensor` | 形状为 `[Batch, Height, Width, Channels]`，值范围 **0.0 到 1.0**。 |
| **MASK** | `torch.Tensor` | 形状为 `[Height, Width]` 或 `[Batch, Height, Width]`。 |
| **LATENT** | `dict` | 包含 `{"samples": tensor}` 的字典。 |
| **MODEL / VAE / CLIP** | `object` | 对应的模型类实例。 |
| **INT / FLOAT / STRING** | `int / float / str` | 基础 Python 类型。 |

---

## 5. 开发守则与最佳实践

1. **避免命名冲突**: 节点类名和 `NODE_CLASS_MAPPINGS` 的 Key 应该是唯一的。建议加上个人或插件前缀（如 `MyProject_LoadImage`）。
2. **不污染全局环境**: 在执行函数内部进行耗时的 `import`（如果是可选功能），或者在 `requirements.txt` 中明确依赖。
3. **设备一致性**: 处理 Tensor 时，确保使用 `.to(device)` 或获取输入 Tensor 的 `.device`。ComfyUI 主要使用 GPU 处理图像。
4. **无状态设计**: 节点原则上应该是“纯函数”。如果需要存储模型，建议利用 ComfyUI 的缓存机制，而不是在类中定义全局变量。
5. **前端交互**: 如果需要修改前端行为（如节点的颜色、动态输入口），需在 `js` 文件夹编写脚本，并使用 `LiteGraph` 的 API。