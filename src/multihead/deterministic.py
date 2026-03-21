"""Deterministic solver functions (zero-cost, perfect accuracy).

These are pure Python functions that don't use LLMs. They provide:
- Perfect accuracy (no hallucination risk)
- Zero cost (no API calls, no GPU)
- Instant execution (<1ms typically)

Use cases:
- Coordinate transformations
- Schema validation
- Mathematical calculations
- Geometry operations
- Format conversions
"""

from __future__ import annotations

import json
from typing import Any


# ---------------------------------------------------------------------------
# Coordinate Transform Functions
# ---------------------------------------------------------------------------

def coordinate_transform(
    points: list[tuple[float, float]] | dict[str, Any],
    source_resolution: tuple[int, int],
    target_resolution: tuple[int, int],
    **kwargs
) -> dict[str, Any]:
    """Transform coordinates from source to target resolution.

    Args:
        points: List of (x, y) tuples OR dict with coordinate data
        source_resolution: (width, height) of source space
        target_resolution: (width, height) of target space

    Returns:
        Dict with transformed coordinates and metadata

    Example:
        >>> result = coordinate_transform(
        ...     points=[(100, 200), (300, 400)],
        ...     source_resolution=(3975, 6150),
        ...     target_resolution=(4042, 5929)
        ... )
        >>> result['transformed_points']
        [(101.68, 193.14), (304.04, 386.29)]
    """
    source_w, source_h = source_resolution
    target_w, target_h = target_resolution

    scale_x = target_w / source_w
    scale_y = target_h / source_h

    # Handle dict input (extract points from various formats)
    if isinstance(points, dict):
        if "points" in points:
            points_list = points["points"]
        elif "coordinates" in points:
            points_list = points["coordinates"]
        elif "polygon" in points:
            points_list = points["polygon"]
        else:
            # Assume dict IS the point data
            points_list = [(points.get("x", 0), points.get("y", 0))]
    else:
        points_list = points

    # Transform
    transformed = [
        (round(x * scale_x, 2), round(y * scale_y, 2))
        for x, y in points_list
    ]

    return {
        "transformed_points": transformed,
        "source_resolution": source_resolution,
        "target_resolution": target_resolution,
        "scale_factors": {"scale_x": round(scale_x, 4), "scale_y": round(scale_y, 4)},
        "coordinate_space": "target",
        "transformation": "linear_scale"
    }


def transform_bbox(
    bbox: list[float],
    source_resolution: tuple[int, int],
    target_resolution: tuple[int, int],
    format: str = "xyxy",  # "xyxy" | "xywh" | "coco"
    **kwargs
) -> dict[str, Any]:
    """Transform bounding box from source to target resolution.

    Args:
        bbox: Bounding box coords [x1, y1, x2, y2] or [x, y, w, h]
        source_resolution: (width, height) of source
        target_resolution: (width, height) of target
        format: bbox format ("xyxy", "xywh", "coco")

    Returns:
        Dict with transformed bbox and metadata
    """
    source_w, source_h = source_resolution
    target_w, target_h = target_resolution

    scale_x = target_w / source_w
    scale_y = target_h / source_h

    if format == "xyxy":
        x1, y1, x2, y2 = bbox
        transformed = [
            round(x1 * scale_x, 2),
            round(y1 * scale_y, 2),
            round(x2 * scale_x, 2),
            round(y2 * scale_y, 2)
        ]
    elif format in ["xywh", "coco"]:
        x, y, w, h = bbox
        transformed = [
            round(x * scale_x, 2),
            round(y * scale_y, 2),
            round(w * scale_x, 2),
            round(h * scale_y, 2)
        ]
    else:
        raise ValueError(f"Unknown bbox format: {format}")

    return {
        "transformed_bbox": transformed,
        "format": format,
        "source_resolution": source_resolution,
        "target_resolution": target_resolution,
        "scale_factors": {"scale_x": round(scale_x, 4), "scale_y": round(scale_y, 4)}
    }


# ---------------------------------------------------------------------------
# Validation Functions
# ---------------------------------------------------------------------------

def validate_schema(
    data: Any,
    schema: dict[str, Any],
    **kwargs
) -> dict[str, Any]:
    """Validate data against JSON schema.

    Args:
        data: Data to validate (dict, list, etc.)
        schema: JSON schema dict

    Returns:
        Dict with validation result
    """
    try:
        from jsonschema import validate, ValidationError

        validate(instance=data, schema=schema)
        return {
            "valid": True,
            "errors": [],
            "schema_version": schema.get("$schema", "unknown")
        }
    except ValidationError as e:
        return {
            "valid": False,
            "errors": [
                {
                    "message": e.message,
                    "path": list(e.path),
                    "validator": e.validator,
                    "validator_value": e.validator_value
                }
            ],
            "schema_version": schema.get("$schema", "unknown")
        }
    except ImportError:
        return {
            "valid": False,
            "errors": [{"message": "jsonschema library not installed"}],
            "install": "pip install jsonschema"
        }


def validate_format(
    text: str,
    min_length: int | None = None,
    max_length: int | None = None,
    regex: str | None = None,
    allowed_values: list[str] | None = None,
    **kwargs
) -> dict[str, Any]:
    """Validate text format constraints.

    Args:
        text: Text to validate
        min_length: Minimum length
        max_length: Maximum length
        regex: Regex pattern to match
        allowed_values: List of allowed values (enum)

    Returns:
        Dict with validation result
    """
    import re

    errors = []

    if min_length is not None and len(text) < min_length:
        errors.append(f"Length {len(text)} below minimum {min_length}")

    if max_length is not None and len(text) > max_length:
        errors.append(f"Length {len(text)} exceeds maximum {max_length}")

    if regex is not None:
        if not re.search(regex, text, re.DOTALL):
            errors.append(f"Does not match regex: {regex}")

    if allowed_values is not None and text not in allowed_values:
        errors.append(f"Value not in allowed list: {allowed_values}")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "actual_length": len(text),
        "constraints": {
            "min_length": min_length,
            "max_length": max_length,
            "regex": regex,
            "allowed_values": allowed_values
        }
    }


# ---------------------------------------------------------------------------
# Geometry/Spatial Functions
# ---------------------------------------------------------------------------

def calculate_bbox(
    points: list[tuple[float, float]] | dict[str, Any],
    padding: float = 0.0,
    **kwargs
) -> dict[str, Any]:
    """Calculate bounding box from points.

    Args:
        points: List of (x, y) tuples OR dict with point data
        padding: Padding to add to bbox (in same units as points)

    Returns:
        Dict with bbox in multiple formats
    """
    # Extract points from dict if needed
    if isinstance(points, dict):
        if "points" in points:
            points_list = points["points"]
        elif "polygon" in points:
            points_list = points["polygon"]
        else:
            raise ValueError("Dict must contain 'points' or 'polygon' key")
    else:
        points_list = points

    if not points_list:
        raise ValueError("Empty points list")

    xs = [p[0] for p in points_list]
    ys = [p[1] for p in points_list]

    x_min = min(xs) - padding
    y_min = min(ys) - padding
    x_max = max(xs) + padding
    y_max = max(ys) + padding

    width = x_max - x_min
    height = y_max - y_min

    return {
        "bbox_xyxy": [x_min, y_min, x_max, y_max],
        "bbox_xywh": [x_min, y_min, width, height],
        "bbox_coco": [x_min, y_min, width, height],  # COCO format
        "center": [(x_min + x_max) / 2, (y_min + y_max) / 2],
        "width": width,
        "height": height,
        "area": width * height,
        "padding": padding
    }


def calculate_iou(
    bbox1: list[float],
    bbox2: list[float],
    format: str = "xyxy",
    **kwargs
) -> dict[str, Any]:
    """Calculate Intersection over Union (IoU) between two bboxes.

    Args:
        bbox1: First bbox
        bbox2: Second bbox
        format: bbox format ("xyxy" or "xywh")

    Returns:
        Dict with IoU and intersection/union areas
    """
    # Convert to xyxy if needed
    if format == "xywh":
        x1, y1, w1, h1 = bbox1
        bbox1 = [x1, y1, x1 + w1, y1 + h1]
        x2, y2, w2, h2 = bbox2
        bbox2 = [x2, y2, x2 + w2, y2 + h2]

    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2

    # Intersection
    x_inter_min = max(x1_min, x2_min)
    y_inter_min = max(y1_min, y2_min)
    x_inter_max = min(x1_max, x2_max)
    y_inter_max = min(y1_max, y2_max)

    if x_inter_max < x_inter_min or y_inter_max < y_inter_min:
        # No intersection
        return {
            "iou": 0.0,
            "intersection_area": 0.0,
            "union_area": 0.0,
            "bbox1_area": (x1_max - x1_min) * (y1_max - y1_min),
            "bbox2_area": (x2_max - x2_min) * (y2_max - y2_min)
        }

    intersection_area = (x_inter_max - x_inter_min) * (y_inter_max - y_inter_min)
    bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
    bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = bbox1_area + bbox2_area - intersection_area

    iou = intersection_area / union_area if union_area > 0 else 0.0

    return {
        "iou": round(iou, 4),
        "intersection_area": round(intersection_area, 2),
        "union_area": round(union_area, 2),
        "bbox1_area": round(bbox1_area, 2),
        "bbox2_area": round(bbox2_area, 2)
    }


# ---------------------------------------------------------------------------
# Helper: Function Registry
# ---------------------------------------------------------------------------

DETERMINISTIC_FUNCTIONS = {
    "coordinate_transform": coordinate_transform,
    "transform_bbox": transform_bbox,
    "validate_schema": validate_schema,
    "validate_format": validate_format,
    "calculate_bbox": calculate_bbox,
    "calculate_iou": calculate_iou,
}


def get_function(function_name: str):
    """Get deterministic function by name.

    Args:
        function_name: Name of function (e.g., "coordinate_transform")

    Returns:
        Function callable

    Raises:
        KeyError: If function not found
    """
    if function_name not in DETERMINISTIC_FUNCTIONS:
        available = ", ".join(DETERMINISTIC_FUNCTIONS.keys())
        raise KeyError(
            f"Deterministic function '{function_name}' not found. "
            f"Available: {available}"
        )
    return DETERMINISTIC_FUNCTIONS[function_name]
