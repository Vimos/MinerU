"""Small geometry / numeric helpers (pure numpy + torch, no cv2).

Replaces the cv2 calls of the reference implementation:

* ``box_points``           -> ``cv2.boxPoints`` (rotated rectangle corners)
* ``point_in_polygon``     -> the ``is_on_right_side``/``test_point`` test used by
                              MolGrapher's ``GraphConstructor._is_obstructed``
* ``crop_pad``             -> ``torchvision.transforms.functional.crop``
                              (zero padding when the window leaves the image)
* ``get_bond_size``        -> ``molgrapher.utils.utils_dataset.get_bond_size``
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

__all__ = ["box_points", "point_in_polygon", "crop_pad", "get_bond_size",
           "regional_maxima", "percentile_like_numpy"]


def box_points(center, size, angle_degrees):
    """``cv2.boxPoints(((cx, cy), (w, h), angle))`` -> list of 4 (x, y) corners.

    OpenCV's implementation (rotating-calipers order, cyclic polygon)::

        b = (cos(a) * 0.5, sin(a) * 0.5)
        p0 = (cx - b0*w - b1*h, cy - b1*w + b0*h)
        p1 = (cx + b0*w - b1*h, cy + b1*w + b0*h)
        p2 = 2c - p0 ; p3 = 2c - p1
    """
    cx, cy = float(center[0]), float(center[1])
    w, h = float(size[0]), float(size[1])
    a = math.radians(float(angle_degrees))
    b0, b1 = math.cos(a) * 0.5, math.sin(a) * 0.5
    p0 = (cx - b0 * w - b1 * h, cy - b1 * w + b0 * h)
    p1 = (cx + b0 * w - b1 * h, cy + b1 * w + b0 * h)
    p2 = (2 * cx - p0[0], 2 * cy - p0[1])
    p3 = (2 * cx - p1[0], 2 * cy - p1[1])
    return [p0, p1, p2, p3]


def _is_on_right_side(x, y, xy0, xy1):
    x0, y0 = xy0
    x1, y1 = xy1
    a = float(y1 - y0)
    b = float(x0 - x1)
    c = -a * x0 - b * y0
    return a * x + b * y + c >= 0


def point_in_polygon(x, y, vertices):
    """Upstream ``GraphConstructor.test_point`` (all-left or all-right)."""
    num_vert = len(vertices)
    is_right = [_is_on_right_side(x, y, vertices[i], vertices[(i + 1) % num_vert])
                for i in range(num_vert)]
    all_left = not any(is_right)
    all_right = all(is_right)
    return all_left or all_right


def crop_pad(image, top, left, height, width, fill=0.0):
    """``torchvision.transforms.functional.crop`` on a CHW tensor, 0-padded."""
    if image.dim() == 2:
        image = image.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    c, h, w = image.shape
    top, left, height, width = int(top), int(left), int(height), int(width)
    out = torch.full((c, height, width), float(fill), dtype=image.dtype,
                     device=image.device)
    src_y0, src_x0 = max(top, 0), max(left, 0)
    src_y1, src_x1 = min(top + height, h), min(left + width, w)
    if src_y1 > src_y0 and src_x1 > src_x0:
        out[:, src_y0 - top:src_y1 - top, src_x0 - left:src_x1 - left] = \
            image[:, src_y0:src_y1, src_x0:src_x1]
    return out[0] if squeeze else out


def get_bond_size(keypoints):
    """75th percentile of the nearest-neighbour distances (upstream helper)."""
    min_distances = []
    for index_keypoint_query, keypoint_query in enumerate(keypoints):
        distances = []
        for index_keypoint_key, keypoint_key in enumerate(keypoints):
            if index_keypoint_key < index_keypoint_query:
                distances.append(math.dist(keypoint_query, keypoint_key))
        if distances:
            min_distances.append(min(distances))
    if min_distances:
        return float(np.percentile(min_distances, 75))
    print("Get bond size error")
    return 100.0


def regional_maxima(heatmap, footprint=5):
    """``mahotas.regmax(heatmap, Bc=np.ones((n, n)))`` equivalent.

    A pixel is a regional maximum when no neighbour inside the ``n x n``
    footprint is strictly larger (flat plateaus therefore stay maxima, exactly
    like mahotas' flat-zone definition).  Implemented with ``max_pool2d``:
    ``-inf`` padding makes the border behave like "no neighbour".
    """
    if isinstance(heatmap, np.ndarray):
        t = torch.from_numpy(np.ascontiguousarray(heatmap, dtype=np.float32))
    else:
        t = heatmap.detach().float().cpu()
    if t.dim() == 2:
        t = t[None, None]
    elif t.dim() == 3:
        t = t[None]
    pad = footprint // 2
    mx = F.max_pool2d(t, kernel_size=footprint, stride=1, padding=pad)
    return (t == mx)[0, 0].numpy()


def percentile_like_numpy(values):
    return float(np.percentile(values, 75)) if len(values) else 100.0
