"""
ComfyUI Mask2JSON - Custom Nodes for Mask Contour Extraction and Visualization
"""

from .mask_nodes import MaskToContourJSON, ContourVisualizer

NODE_CLASS_MAPPINGS = {
    "MaskToContourJSON": MaskToContourJSON,
    "ContourVisualizer": ContourVisualizer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskToContourJSON": "Mask to Contour JSON",
    "ContourVisualizer": "Contour Visualizer",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
