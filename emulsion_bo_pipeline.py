from __future__ import annotations

import argparse
import json
import os
import glob
from dataclasses import dataclass, asdict, fields
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import cv2
from scipy import ndimage as ndi
from skimage import filters, morphology, measure
from skimage.segmentation import clear_border

from sklearn.ensemble import ExtraTreesRegressor
from skopt.space import Real


# ============================================================
# 1. Data schema
# ============================================================

@dataclass
class ExperimentCondition:
    # W1
    w1_glycerol_wt: float = 10.0
    w1_tween20_wt: float = 1.0

    # O
    o_dextrin_palmitate_wt: float = 5.0
    o_span80_wt: float = 2.0
    o_adm_wt: float = 0.5

    # W2
    w2_glycerol_wt: float = 10.0
    w2_pva_wt: float = 2.0
    w2_tween80_wt: float = 0.75

    # Flow [mL/min]
    q_w1_ml_min: float = 30.0
    q_o_ml_min: float = 120.0
    q_w2_ml_min: float = 900.0

    # Hidden / controlled variables
    w1_pH: float = 7.0
    w2_pH: float = 7.0
    w1_conductivity_mScm: float = 0.05
    w2_conductivity_mScm: float = 0.05
    temperature_C: float = 25.0
    bath_stirring_speed_rpm: float = 0.0
    bath_carbomer_wt: float = 0.0

    # Metadata
    batch_id: str = "batch_000"
    time_h: float = 0.0
    operator: str = "unknown"


@dataclass
class VialMetrics:
    creaming_index: float = np.nan


@dataclass
class ObjectiveSummary:
    objective_value: float
    feasibility_penalty: float
    total_score: float


@dataclass
class ImageMetrics:
    n_detected_outer: int
    n_valid_double_emulsion: int

    mean_outer_diameter_um: float
    median_outer_diameter_um: float
    std_outer_diameter_um: float
    cv_outer_percent: float
    pdi_outer_img: float

    mean_inner_diameter_um: float
    mean_shell_thickness_um: float
    std_shell_thickness_um: float
    cv_shell_percent: float

    mean_core_outer_ratio: float
    std_core_outer_ratio: float
    mean_concentricity_ratio: float

    mean_outer_circularity: float
    mean_inner_circularity: float

    weak_core_fraction: float
    truncated_fraction: float
    failed_inner_detection_fraction: float
    good_double_emulsion_fraction: float

    estimated_um_per_pixel_from_scale: float
    field_qc_pass: int


# ============================================================
# 2. Utility functions
# ============================================================

def ratio_core(q_w1: float, q_o: float) -> float:
    return q_w1 / max(q_o, 1e-12)


def ratio_sheath(q_w1: float, q_o: float, q_w2: float) -> float:
    return q_w2 / max((q_w1 + q_o), 1e-12)


def total_flow(q_w1: float, q_o: float, q_w2: float) -> float:
    return q_w1 + q_o + q_w2


def format_time_tag(time_h: float) -> str:
    return f"{time_h:.2f}".replace(".", "p")


IMAGE_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp")


def load_json_dict(json_path: str) -> Dict[str, object]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON file must contain an object: {json_path}")
    return data


def dataclass_from_dict(cls, data: Dict[str, object]):
    valid_names = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in data.items() if k in valid_names}
    return cls(**filtered)


def list_image_files(image_dir: str) -> List[str]:
    paths: List[str] = []
    for ext in IMAGE_EXTENSIONS:
        paths.extend(sorted(glob.glob(os.path.join(image_dir, ext))))
    return [p for p in paths if os.path.isfile(p)]


def find_optional_vial_image(experiment_dir: str) -> Optional[str]:
    for ext in IMAGE_EXTENSIONS:
        matches = sorted(glob.glob(os.path.join(experiment_dir, f"vial{ext[1:]}")))
        if len(matches) > 0:
            return matches[0]
    return None


def discover_experiment_dirs(experiments_root: str) -> List[str]:
    if not os.path.isdir(experiments_root):
        raise ValueError(f"Experiments root does not exist: {experiments_root}")

    out = []
    for name in sorted(os.listdir(experiments_root)):
        exp_dir = os.path.join(experiments_root, name)
        if not os.path.isdir(exp_dir):
            continue
        if not os.path.exists(os.path.join(exp_dir, "condition.json")):
            continue
        if not os.path.isdir(os.path.join(exp_dir, "microscopy")):
            continue
        out.append(exp_dir)
    return out


def read_existing_history(history_csv: str) -> pd.DataFrame:
    if not os.path.exists(history_csv):
        return pd.DataFrame()
    df = pd.read_csv(history_csv)
    if "record_type" in df.columns:
        df = df[df["record_type"].fillna("experiment").astype(str) == "experiment"].copy()
    return df


def existing_batch_ids_from_history(history_csv: str) -> set[str]:
    df = read_existing_history(history_csv)
    if df.empty or "batch_id" not in df.columns:
        return set()
    return {
        str(v)
        for v in df["batch_id"].dropna().astype(str).tolist()
        if len(str(v).strip()) > 0
    }


def condition_field_names() -> List[str]:
    return [f.name for f in fields(ExperimentCondition)]


def bo_variable_names() -> List[str]:
    return [
        "w1_glycerol_wt",
        "w1_tween20_wt",
        "o_dextrin_palmitate_wt",
        "o_span80_wt",
        "o_adm_wt",
        "w2_glycerol_wt",
        "w2_pva_wt",
        "w2_tween80_wt",
        "q_w1_ml_min",
        "q_o_ml_min",
        "q_w2_ml_min",
        "bath_stirring_speed_rpm",
        "bath_carbomer_wt",
    ]


def canonical_condition_payload(cond: ExperimentCondition) -> Dict[str, object]:
    return {name: getattr(cond, name) for name in condition_field_names()}


def canonical_condition_json(cond: ExperimentCondition) -> str:
    payload = canonical_condition_payload(cond)
    return json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def existing_condition_map(history_csv: str) -> Dict[str, str]:
    df = read_existing_history(history_csv)
    if df.empty or "batch_id" not in df.columns:
        return {}

    out: Dict[str, str] = {}
    for _, row in df.iterrows():
        batch_id = str(row["batch_id"]).strip() if pd.notna(row["batch_id"]) else ""
        if not batch_id:
            continue
        payload = {
            name: row[name]
            for name in condition_field_names()
            if name in row.index and pd.notna(row[name])
        }
        cond = dataclass_from_dict(ExperimentCondition, payload)
        cond.batch_id = batch_id
        out[batch_id] = canonical_condition_json(cond)
    return out


# ============================================================
# 3. Image processing for double-emulsion microscopy images
# ============================================================

def read_bgr(image_path: str) -> np.ndarray:
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    return img


def mask_red_annotations(bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Mask red scale bar / annotation text.
    Returns cleaned_bgr, red_mask.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, (0, 120, 80), (12, 255, 255))
    mask2 = cv2.inRange(hsv, (168, 120, 80), (179, 255, 255))
    red_mask = cv2.bitwise_or(mask1, mask2)
    red_mask = cv2.dilate(red_mask, np.ones((5, 5), np.uint8), iterations=1)

    cleaned = bgr.copy()
    if np.any(red_mask > 0):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        bg_val = int(np.median(gray[red_mask == 0])) if np.any(red_mask == 0) else 180
        cleaned[red_mask > 0] = (bg_val, bg_val, bg_val)

    return cleaned, red_mask


def estimate_um_per_pixel_from_red_scale_bar(
    bgr: np.ndarray,
    known_scale_um: float = 200.0,
) -> Optional[float]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, (0, 120, 80), (12, 255, 255))
    mask2 = cv2.inRange(hsv, (168, 120, 80), (179, 255, 255))
    red_mask = cv2.bitwise_or(mask1, mask2)

    n, labels, stats, _ = cv2.connectedComponentsWithStats((red_mask > 0).astype(np.uint8), 8)

    best_len = None
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if w >= 80 and h <= 35 and area >= 800:
            best_len = w if best_len is None else max(best_len, w)

    if best_len is None or best_len <= 0:
        return None

    return float(known_scale_um / float(best_len))


def preprocess_double_emulsion_image(bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns cleaned_bgr, gray, flat_dark_emphasis.
    """
    cleaned, _ = mask_red_annotations(bgr)
    gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    bg = cv2.GaussianBlur(gray, (0, 0), 25)
    flat_dark = bg.astype(np.float32) - gray.astype(np.float32)
    flat_dark = cv2.normalize(flat_dark, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return cleaned, gray, flat_dark


def _filter_outer_regions(
    label_img: np.ndarray,
    min_outer_diameter_px: float,
    max_outer_diameter_px: float,
    circularity_min: float,
    solidity_min: float,
) -> np.ndarray:
    out = np.zeros_like(label_img, dtype=np.int32)
    new_id = 1

    for region in measure.regionprops(label_img):
        eqd = float(region.equivalent_diameter_area)
        if not (min_outer_diameter_px <= eqd <= max_outer_diameter_px):
            continue

        area = float(region.area)
        perim = float(max(region.perimeter, 1e-12))
        circ = 4.0 * np.pi * area / (perim ** 2)
        solidity = float(region.solidity)

        if circ < circularity_min or solidity < solidity_min:
            continue

        coords = region.coords
        out[coords[:, 0], coords[:, 1]] = new_id
        new_id += 1

    return out


def segment_outer_droplets(
    flat_dark_emphasis: np.ndarray,
    min_outer_diameter_px: float = 80.0,
    max_outer_diameter_px: float = 1800.0,
    enable_empty_frame_relaxation: bool = True,
) -> np.ndarray:
    """
    Segment OUTER droplet contour for dark-shell / bright-core image morphology.
    """
    thr = filters.threshold_otsu(flat_dark_emphasis)
    mask = flat_dark_emphasis > thr

    min_area = int(np.pi * (min_outer_diameter_px / 2.0) ** 2 * 0.35)
    mask = morphology.remove_small_objects(mask, max_size=max(min_area, 200))
    mask = morphology.remove_small_holes(mask, max_size=800)
    mask = morphology.closing(mask, morphology.disk(5))
    mask = morphology.opening(mask, morphology.disk(3))

    mask = ndi.binary_fill_holes(mask)
    mask = clear_border(mask)

    label_img = measure.label(mask)
    out = _filter_outer_regions(
        label_img=label_img,
        min_outer_diameter_px=min_outer_diameter_px,
        max_outer_diameter_px=max_outer_diameter_px,
        circularity_min=0.65,
        solidity_min=0.90,
    )

    if np.max(out) == 0 and enable_empty_frame_relaxation:
        out = _filter_outer_regions(
            label_img=label_img,
            min_outer_diameter_px=min_outer_diameter_px,
            max_outer_diameter_px=max_outer_diameter_px,
            circularity_min=0.50,
            solidity_min=0.88,
        )

    return out


def _smooth_1d(vals: np.ndarray) -> np.ndarray:
    if vals.size < 5:
        return vals.astype(float)
    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=float)
    kernel /= np.sum(kernel)
    return np.convolve(vals.astype(float), kernel, mode="same")


def _smooth_closed_polygon(points: np.ndarray, passes: int = 3) -> np.ndarray:
    pts = points.astype(float).copy()
    if pts.shape[0] < 5:
        return pts

    for _ in range(max(passes, 0)):
        prev_pts = np.roll(pts, 1, axis=0)
        next_pts = np.roll(pts, -1, axis=0)
        pts = 0.25 * prev_pts + 0.50 * pts + 0.25 * next_pts

    return pts


def _smoothed_contours_from_mask(mask: np.ndarray) -> List[np.ndarray]:
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    if not np.any(mask_u8):
        return []

    smooth = cv2.GaussianBlur(mask_u8, (5, 5), 0)
    _, smooth_bin = cv2.threshold(smooth, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(smooth_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: List[np.ndarray] = []
    for contour in contours:
        if len(contour) < 5:
            out.append(contour)
            continue
        perim = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.008 * max(perim, 1.0), True)
        out.append(approx)
    return out


def _ellipse_contours_from_mask(mask: np.ndarray) -> List[np.ndarray]:
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out: List[np.ndarray] = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area <= 0:
            continue

        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            pts = cv2.ellipse2Poly(
                center=(int(round(ellipse[0][0])), int(round(ellipse[0][1]))),
                axes=(max(1, int(round(ellipse[1][0] / 2.0))), max(1, int(round(ellipse[1][1] / 2.0)))),
                angle=int(round(ellipse[2])),
                arcStart=0,
                arcEnd=360,
                delta=6,
            )
            out.append(pts.reshape(-1, 1, 2))
        else:
            out.append(contour)

    return out


def _ellipse_geom_from_mask(mask: np.ndarray, um_per_pixel: float) -> Optional[Dict[str, float]]:
    mask_u8 = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return None

    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 5:
        return None

    (cx, cy), (axis1, axis2), _ = cv2.fitEllipse(contour)
    major = float(max(axis1, axis2))
    minor = float(min(axis1, axis2))
    if major < 3.0 or minor < 3.0:
        return None
    semi_major = max(major / 2.0, 1e-12)
    semi_minor = max(minor / 2.0, 1e-12)

    area = float(np.pi * semi_major * semi_minor)
    h = ((semi_major - semi_minor) ** 2) / ((semi_major + semi_minor) ** 2 + 1e-12)
    perim = float(
        np.pi
        * (semi_major + semi_minor)
        * (1.0 + (3.0 * h) / (10.0 + np.sqrt(max(4.0 - 3.0 * h, 1e-12))))
    )
    eqd_px = float(2.0 * np.sqrt(area / np.pi))
    circ = float(4.0 * np.pi * area / max(perim ** 2, 1e-12))
    if eqd_px < 3.0 or not np.isfinite(circ) or circ <= 0.0 or circ > 1.2:
        return None

    return {
        "area_px": area,
        "equiv_diameter_px": eqd_px,
        "diameter_um": eqd_px * um_per_pixel,
        "perimeter_px": perim,
        "circularity": circ,
        "solidity": 1.0,
        "centroid_x": float(cx),
        "centroid_y": float(cy),
        "major_axis_px": major,
        "minor_axis_px": minor,
    }


def _trace_expanded_inner_boundary(
    gray_roi: np.ndarray,
    outer_mask_roi: np.ndarray,
    seed_mask_roi: np.ndarray,
    center_yx: Tuple[float, float],
    n_angles: int = 72,
) -> np.ndarray:
    cy, cx = center_yx
    H, W = gray_roi.shape[:2]
    outer_bool = outer_mask_roi > 0
    seed_bool = seed_mask_roi > 0

    if np.sum(seed_bool) < 20:
        return seed_mask_roi.astype(np.uint8)

    boundary_pts: List[Tuple[int, int]] = []
    max_radius = float(np.sqrt(H * H + W * W))

    for theta in np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False):
        ray_vals = []
        ray_seed = []
        ray_coords = []

        for r in np.arange(0.0, max_radius, 1.0):
            x = int(round(cx + r * np.cos(theta)))
            y = int(round(cy + r * np.sin(theta)))
            if x < 0 or x >= W or y < 0 or y >= H:
                break
            if not outer_bool[y, x]:
                if r > 0:
                    break
                continue

            ray_vals.append(float(gray_roi[y, x]))
            ray_seed.append(bool(seed_bool[y, x]))
            ray_coords.append((x, y))

        if len(ray_vals) < 8:
            continue

        seed_indices = np.flatnonzero(np.asarray(ray_seed, dtype=bool))
        if seed_indices.size == 0:
            continue

        seed_end = int(seed_indices[-1])
        outer_end = len(ray_vals) - 1
        if outer_end - seed_end < 4:
            boundary_pts.append(ray_coords[seed_end])
            continue

        search_start = min(seed_end + 1, outer_end - 2)
        search_end = max(search_start + 1, int(seed_end + 0.80 * (outer_end - seed_end)))
        search_end = min(search_end, outer_end - 1)

        smooth_vals = _smooth_1d(np.asarray(ray_vals, dtype=float))
        grad_vals = np.gradient(smooth_vals)

        trough_idx = search_start + int(np.argmin(smooth_vals[search_start:search_end + 1]))
        if trough_idx >= search_end:
            boundary_pts.append(ray_coords[seed_end])
            continue

        post_grad = grad_vals[trough_idx:search_end + 1]
        rise_rel = int(np.argmax(post_grad))
        rise_idx = trough_idx + rise_rel

        rise_strength = float(post_grad[rise_rel])
        rise_delta = float(smooth_vals[rise_idx] - smooth_vals[trough_idx])
        if rise_strength <= 0.8 or rise_delta <= 4.0 or rise_idx <= seed_end:
            boundary_pts.append(ray_coords[seed_end])
            continue

        boundary_pts.append(ray_coords[rise_idx])

    if len(boundary_pts) < 12:
        return seed_mask_roi.astype(np.uint8)

    poly = np.asarray(boundary_pts, dtype=np.float32)
    poly = _smooth_closed_polygon(poly, passes=8)
    poly[:, 0] = np.clip(poly[:, 0], 0, W - 1)
    poly[:, 1] = np.clip(poly[:, 1], 0, H - 1)
    poly = np.round(poly).astype(np.int32)
    expanded = np.zeros_like(seed_mask_roi, dtype=np.uint8)
    cv2.fillPoly(expanded, [poly], 1)
    expanded = (expanded > 0) & outer_bool
    expanded = expanded | seed_bool
    expanded = morphology.closing(expanded, morphology.disk(3))
    expanded = morphology.opening(expanded, morphology.disk(2))
    expanded = ndi.binary_fill_holes(expanded)

    lab = measure.label(expanded)
    regs = measure.regionprops(lab)
    if len(regs) == 0:
        return seed_mask_roi.astype(np.uint8)

    best_region = max(
        regs,
        key=lambda r: r.area - 5.0 * ((r.centroid[0] - cy) ** 2 + (r.centroid[1] - cx) ** 2) ** 0.5,
    )
    out = np.zeros_like(seed_mask_roi, dtype=np.uint8)
    coords = best_region.coords
    out[coords[:, 0], coords[:, 1]] = 1
    return out.astype(np.uint8)


def detect_inner_core_in_roi(
    gray_roi: np.ndarray,
    outer_mask_roi: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Detect inner core only inside each outer droplet ROI.
    """
    stats = {
        "inner_detected": 0,
        "weak_core_boundary": 0,
    }

    vals = gray_roi[outer_mask_roi > 0]
    if vals.size < 100:
        return np.zeros_like(outer_mask_roi, dtype=np.uint8), stats

    roi_blur = cv2.GaussianBlur(gray_roi, (0, 0), 7)
    local = gray_roi.astype(np.float32) - 0.5 * roi_blur.astype(np.float32)
    local = cv2.normalize(local, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    masked_vals = local[outer_mask_roi > 0]
    if masked_vals.size < 50:
        return np.zeros_like(outer_mask_roi, dtype=np.uint8), stats

    try:
        thr = filters.threshold_otsu(masked_vals)
    except Exception:
        thr = np.percentile(masked_vals, 60)

    inner = np.zeros_like(outer_mask_roi, dtype=np.uint8)
    inner[(local >= thr) & (outer_mask_roi > 0)] = 1

    inner = morphology.remove_small_objects(inner.astype(bool), max_size=200)
    inner = morphology.opening(inner, morphology.disk(2))
    inner = morphology.closing(inner, morphology.disk(3))
    inner = ndi.binary_fill_holes(inner)

    label_img = measure.label(inner)
    regions = measure.regionprops(label_img)

    if len(regions) == 0:
        stats["weak_core_boundary"] = 1
        return np.zeros_like(outer_mask_roi, dtype=np.uint8), stats

    cy, cx = np.array(outer_mask_roi.shape) / 2.0
    best_region = None
    best_score = -np.inf

    for r in regions:
        area = float(r.area)
        perim = float(max(r.perimeter, 1e-12))
        circ = 4.0 * np.pi * area / (perim ** 2)
        ry, rx = r.centroid
        dist = np.sqrt((ry - cy) ** 2 + (rx - cx) ** 2)
        score = area + 2000.0 * circ - 10.0 * dist
        if score > best_score:
            best_score = score
            best_region = r

    out = np.zeros_like(outer_mask_roi, dtype=np.uint8)
    coords = best_region.coords
    out[coords[:, 0], coords[:, 1]] = 1

    out = _trace_expanded_inner_boundary(
        gray_roi=gray_roi,
        outer_mask_roi=outer_mask_roi,
        seed_mask_roi=out,
        center_yx=best_region.centroid,
    )

    expanded_lab = measure.label(out)
    expanded_regions = measure.regionprops(expanded_lab)
    if len(expanded_regions) > 0:
        best_region = max(expanded_regions, key=lambda r: r.area)

    stats["inner_detected"] = 1

    outer_area = max(float(np.sum(outer_mask_roi > 0)), 1.0)
    inner_area = float(best_region.area)
    if inner_area / outer_area < 0.08:
        stats["weak_core_boundary"] = 1

    return out.astype(np.uint8), stats


def _region_basic_geom(region, um_per_pixel: float) -> Dict[str, float]:
    area = float(region.area)
    perim_raw = float(region.perimeter)
    perim = float(perim_raw) if np.isfinite(perim_raw) else np.nan
    eqd_px = float(region.equivalent_diameter_area)
    if (
        not np.isfinite(area)
        or not np.isfinite(perim)
        or eqd_px < 3.0
        or area <= 1.0
        or perim <= 1.0
    ):
        circ = np.nan
    else:
        circ = float(4.0 * np.pi * area / (perim ** 2))
        if not np.isfinite(circ) or circ <= 0.0 or circ > 1.2:
            circ = np.nan
    solidity = float(region.solidity)

    return {
        "area_px": area,
        "equiv_diameter_px": eqd_px,
        "diameter_um": eqd_px * um_per_pixel,
        "perimeter_px": perim,
        "circularity": circ,
        "solidity": solidity,
    }


def extract_double_emulsion_objects(
    image_path: str,
    cleaned_bgr: np.ndarray,
    gray: np.ndarray,
    outer_labels: np.ndarray,
    um_per_pixel: float,
) -> pd.DataFrame:
    rows = []
    H, W = gray.shape[:2]

    for region in measure.regionprops(outer_labels):
        y0, x0, y1, x1 = region.bbox

        pad = 10
        y0p = max(0, y0 - pad)
        x0p = max(0, x0 - pad)
        y1p = min(H, y1 + pad)
        x1p = min(W, x1 + pad)

        gray_roi = gray[y0p:y1p, x0p:x1p]
        outer_mask_roi = (outer_labels[y0p:y1p, x0p:x1p] == region.label).astype(np.uint8)

        inner_mask_roi, inner_stats = detect_inner_core_in_roi(gray_roi, outer_mask_roi)

        outer_geom = _region_basic_geom(region, um_per_pixel)
        outer_cy, outer_cx = region.centroid

        inner_area_px = np.nan
        inner_eqd_px = np.nan
        inner_d_um = np.nan
        inner_perim_px = np.nan
        inner_circ = np.nan
        inner_solidity = np.nan
        center_offset_px = np.nan
        concentricity_ratio = np.nan
        shell_mean_px = np.nan
        shell_mean_um = np.nan
        core_outer_ratio = np.nan

        if inner_stats["inner_detected"] == 1:
            inner_lab = measure.label(inner_mask_roi)
            inner_regions = measure.regionprops(inner_lab)

            if len(inner_regions) > 0:
                inner_region = max(inner_regions, key=lambda r: r.area)
                inner_geom = _ellipse_geom_from_mask(inner_mask_roi, um_per_pixel)
                if inner_geom is None:
                    inner_geom = _region_basic_geom(inner_region, um_per_pixel)
                    inner_cy_local, inner_cx_local = inner_region.centroid
                else:
                    inner_cx_local = inner_geom["centroid_x"]
                    inner_cy_local = inner_geom["centroid_y"]

                inner_area_px = inner_geom["area_px"]
                inner_eqd_px = inner_geom["equiv_diameter_px"]
                inner_d_um = inner_geom["diameter_um"]
                inner_perim_px = inner_geom["perimeter_px"]
                inner_circ = inner_geom["circularity"]
                inner_solidity = inner_geom["solidity"]

                inner_cy = y0p + inner_cy_local
                inner_cx = x0p + inner_cx_local

                center_offset_px = float(np.sqrt((outer_cy - inner_cy) ** 2 + (outer_cx - inner_cx) ** 2))
                concentricity_ratio = center_offset_px / max(outer_geom["equiv_diameter_px"] / 2.0, 1e-12)

                shell_mean_px = max((outer_geom["equiv_diameter_px"] - inner_eqd_px) / 2.0, 0.0)
                shell_mean_um = shell_mean_px * um_per_pixel
                core_outer_ratio = inner_eqd_px / max(outer_geom["equiv_diameter_px"], 1e-12)

        border_truncated = int(
            x0 <= 2 or y0 <= 2 or x1 >= W - 2 or y1 >= H - 2
        )

        is_round_outer = int(
            outer_geom["circularity"] >= 0.85 and outer_geom["solidity"] >= 0.97
        )

        is_good_double_emulsion = int(
            is_round_outer == 1
            and inner_stats["inner_detected"] == 1
            and np.isfinite(core_outer_ratio)
            and np.isfinite(concentricity_ratio)
            and (0.35 <= core_outer_ratio <= 0.80)
            and (concentricity_ratio <= 0.12)
            and inner_stats["weak_core_boundary"] == 0
            and border_truncated == 0
        )

        rows.append({
            "image_path": image_path,
            "object_id": int(region.label),
            "outer_area_px": outer_geom["area_px"],
            "outer_equiv_diameter_px": outer_geom["equiv_diameter_px"],
            "outer_diameter_um": outer_geom["diameter_um"],
            "outer_perimeter_px": outer_geom["perimeter_px"],
            "outer_circularity": outer_geom["circularity"],
            "outer_solidity": outer_geom["solidity"],
            "inner_area_px": inner_area_px,
            "inner_equiv_diameter_px": inner_eqd_px,
            "inner_diameter_um": inner_d_um,
            "inner_perimeter_px": inner_perim_px,
            "inner_circularity": inner_circ,
            "inner_solidity": inner_solidity,
            "center_offset_px": center_offset_px,
            "concentricity_ratio": concentricity_ratio,
            "shell_thickness_mean_px": shell_mean_px,
            "shell_thickness_mean_um": shell_mean_um,
            "core_to_outer_diameter_ratio": core_outer_ratio,
            "inner_detected": int(inner_stats["inner_detected"]),
            "border_truncated": border_truncated,
            "weak_core_boundary": int(inner_stats["weak_core_boundary"]),
            "is_round_outer": is_round_outer,
            "is_good_double_emulsion": is_good_double_emulsion,
        })

    return pd.DataFrame(rows)


def summarize_double_emulsion_df(
    df: pd.DataFrame,
    estimated_um_per_pixel_from_scale: float = np.nan,
) -> ImageMetrics:
    if df.empty:
        return ImageMetrics(
            n_detected_outer=0,
            n_valid_double_emulsion=0,
            mean_outer_diameter_um=np.nan,
            median_outer_diameter_um=np.nan,
            std_outer_diameter_um=np.nan,
            cv_outer_percent=np.nan,
            pdi_outer_img=np.nan,
            mean_inner_diameter_um=np.nan,
            mean_shell_thickness_um=np.nan,
            std_shell_thickness_um=np.nan,
            cv_shell_percent=np.nan,
            mean_core_outer_ratio=np.nan,
            std_core_outer_ratio=np.nan,
            mean_concentricity_ratio=np.nan,
            mean_outer_circularity=np.nan,
            mean_inner_circularity=np.nan,
            weak_core_fraction=np.nan,
            truncated_fraction=np.nan,
            failed_inner_detection_fraction=np.nan,
            good_double_emulsion_fraction=np.nan,
            estimated_um_per_pixel_from_scale=estimated_um_per_pixel_from_scale,
            field_qc_pass=0,
        )

    valid_outer = df.copy()
    valid_inner = df[df["inner_detected"] == 1].copy()

    outer_d = valid_outer["outer_diameter_um"].values.astype(float)
    mean_outer = float(np.mean(outer_d))
    std_outer = float(np.std(outer_d, ddof=1)) if len(outer_d) > 1 else 0.0
    cv_outer = 100.0 * std_outer / max(mean_outer, 1e-12)
    pdi_outer = (std_outer / max(mean_outer, 1e-12)) ** 2

    if not valid_inner.empty:
        shell_vals = valid_inner["shell_thickness_mean_um"].values.astype(float)
        mean_shell = float(np.mean(shell_vals))
        std_shell = float(np.std(shell_vals, ddof=1)) if len(shell_vals) > 1 else 0.0
        cv_shell = 100.0 * std_shell / max(mean_shell, 1e-12)

        core_outer_vals = valid_inner["core_to_outer_diameter_ratio"].values.astype(float)
        std_ratio = float(np.std(core_outer_vals, ddof=1)) if len(core_outer_vals) > 1 else 0.0
    else:
        mean_shell = np.nan
        std_shell = np.nan
        cv_shell = np.nan
        std_ratio = np.nan

    field_qc_pass = int(
        len(valid_outer) >= 3
        and float((valid_outer["border_truncated"] == 1).mean()) <= 0.20
    )

    good_fraction = float((valid_outer["is_good_double_emulsion"] == 1).mean())

    return ImageMetrics(
        n_detected_outer=int(len(valid_outer)),
        n_valid_double_emulsion=int((valid_outer["is_good_double_emulsion"] == 1).sum()),
        mean_outer_diameter_um=mean_outer,
        median_outer_diameter_um=float(np.median(outer_d)),
        std_outer_diameter_um=std_outer,
        cv_outer_percent=float(cv_outer),
        pdi_outer_img=float(pdi_outer),
        mean_inner_diameter_um=float(valid_inner["inner_diameter_um"].mean()) if not valid_inner.empty else np.nan,
        mean_shell_thickness_um=float(mean_shell),
        std_shell_thickness_um=float(std_shell) if np.isfinite(std_shell) else np.nan,
        cv_shell_percent=float(cv_shell) if np.isfinite(cv_shell) else np.nan,
        mean_core_outer_ratio=float(valid_inner["core_to_outer_diameter_ratio"].mean()) if not valid_inner.empty else np.nan,
        std_core_outer_ratio=float(std_ratio) if np.isfinite(std_ratio) else np.nan,
        mean_concentricity_ratio=float(valid_inner["concentricity_ratio"].mean()) if not valid_inner.empty else np.nan,
        mean_outer_circularity=float(valid_outer["outer_circularity"].mean()),
        mean_inner_circularity=float(valid_inner["inner_circularity"].mean()) if not valid_inner.empty else np.nan,
        weak_core_fraction=float((valid_outer["weak_core_boundary"] == 1).mean()),
        truncated_fraction=float((valid_outer["border_truncated"] == 1).mean()),
        failed_inner_detection_fraction=float((valid_outer["inner_detected"] == 0).mean()),
        good_double_emulsion_fraction=good_fraction,
        estimated_um_per_pixel_from_scale=float(estimated_um_per_pixel_from_scale) if np.isfinite(estimated_um_per_pixel_from_scale) else np.nan,
        field_qc_pass=field_qc_pass,
    )


def draw_overlay(
    cleaned_bgr: np.ndarray,
    gray: np.ndarray,
    outer_labels: np.ndarray,
    obj_df: pd.DataFrame,
) -> np.ndarray:
    overlay = cleaned_bgr.copy()
    H, W = gray.shape[:2]

    for region in measure.regionprops(outer_labels):
        row = obj_df[obj_df["object_id"] == int(region.label)]
        if row.empty:
            continue
        row = row.iloc[0]

        coords = region.coords
        mask = np.zeros(outer_labels.shape, dtype=np.uint8)
        mask[coords[:, 0], coords[:, 1]] = 1

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if row["is_good_double_emulsion"] == 1:
            color = (0, 255, 0)
        elif row["border_truncated"] == 1 or row["inner_detected"] == 0:
            color = (0, 0, 255)
        else:
            color = (0, 255, 255)

        cv2.drawContours(overlay, contours, -1, color, 3)

        if int(row["inner_detected"]) == 1:
            y0, x0, y1, x1 = region.bbox
            pad = 10
            y0p = max(0, y0 - pad)
            x0p = max(0, x0 - pad)
            y1p = min(H, y1 + pad)
            x1p = min(W, x1 + pad)

            gray_roi = gray[y0p:y1p, x0p:x1p]
            outer_mask_roi = (outer_labels[y0p:y1p, x0p:x1p] == region.label).astype(np.uint8)
            inner_mask_roi, inner_stats = detect_inner_core_in_roi(gray_roi, outer_mask_roi)

            if inner_stats["inner_detected"] == 1 and np.any(inner_mask_roi > 0):
                inner_mask_full = np.zeros_like(outer_labels, dtype=np.uint8)
                inner_mask_full[y0p:y1p, x0p:x1p][inner_mask_roi > 0] = 1
                inner_contours = _ellipse_contours_from_mask(inner_mask_full)
                cv2.drawContours(overlay, inner_contours, -1, color, 2)

        cy, cx = region.centroid
        text = f"OD={row['outer_diameter_um']:.0f}um"
        cv2.putText(
            overlay,
            text,
            (int(cx) - 40, int(cy)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    return overlay


def analyze_microscopy_images(
    image_paths: List[str],
    um_per_pixel: Optional[float],
    min_outer_diameter_um: float = 50.0,
    max_outer_diameter_um: float = 2000.0,
    save_overlay_dir: Optional[str] = None,
    try_scale_bar_estimation: bool = True,
    known_scale_um: float = 200.0,
    enable_empty_frame_relaxation: bool = True,
) -> Tuple[pd.DataFrame, ImageMetrics]:
    if len(image_paths) == 0:
        empty = summarize_double_emulsion_df(pd.DataFrame(), np.nan)
        return pd.DataFrame(), empty

    if save_overlay_dir is not None:
        os.makedirs(save_overlay_dir, exist_ok=True)

    all_rows = []
    est_scales = []

    resolved_um_per_pixel = um_per_pixel
    if resolved_um_per_pixel is None:
        for image_path in image_paths:
            bgr = read_bgr(image_path)
            est_scale = estimate_um_per_pixel_from_red_scale_bar(bgr, known_scale_um=known_scale_um)
            if est_scale is not None and np.isfinite(est_scale):
                est_scales.append(float(est_scale))

        if len(est_scales) == 0:
            raise ValueError(
                "um_per_pixel was not provided and no red scale bar could be detected "
                "to estimate it automatically."
            )

        resolved_um_per_pixel = float(np.mean(est_scales))

    for image_path in image_paths:
        bgr = read_bgr(image_path)

        if try_scale_bar_estimation:
            est_scale = estimate_um_per_pixel_from_red_scale_bar(bgr, known_scale_um=known_scale_um)
            if est_scale is not None and np.isfinite(est_scale):
                est_scales.append(float(est_scale))

        cleaned_bgr, gray, flat_dark = preprocess_double_emulsion_image(bgr)

        min_outer_diameter_px = min_outer_diameter_um / max(resolved_um_per_pixel, 1e-12)
        max_outer_diameter_px = max_outer_diameter_um / max(resolved_um_per_pixel, 1e-12)

        outer_labels = segment_outer_droplets(
            flat_dark_emphasis=flat_dark,
            min_outer_diameter_px=min_outer_diameter_px,
            max_outer_diameter_px=max_outer_diameter_px,
            enable_empty_frame_relaxation=enable_empty_frame_relaxation,
        )

        obj_df = extract_double_emulsion_objects(
            image_path=image_path,
            cleaned_bgr=cleaned_bgr,
            gray=gray,
            outer_labels=outer_labels,
            um_per_pixel=resolved_um_per_pixel,
        )

        if not obj_df.empty:
            all_rows.append(obj_df)

        if save_overlay_dir is not None:
            overlay = draw_overlay(cleaned_bgr, gray, outer_labels, obj_df)
            out_name = os.path.splitext(os.path.basename(image_path))[0] + "_overlay.png"
            cv2.imwrite(os.path.join(save_overlay_dir, out_name), overlay)

    if len(all_rows) == 0:
        est_mean = float(np.mean(est_scales)) if len(est_scales) > 0 else np.nan
        empty = summarize_double_emulsion_df(pd.DataFrame(), est_mean)
        return pd.DataFrame(), empty

    df_all = pd.concat(all_rows, ignore_index=True)

    est_mean = float(np.mean(est_scales)) if len(est_scales) > 0 else np.nan
    metrics = summarize_double_emulsion_df(
        df_all,
        estimated_um_per_pixel_from_scale=est_mean,
    )

    return df_all, metrics


# ============================================================
# 4. Vial/macroscopic image processing for creaming index
# ============================================================

def analyze_vial_image_creaming_index(image_path: str) -> VialMetrics:
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return VialMetrics(np.nan)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    profile = gray.mean(axis=1).astype(float)
    kernel = np.ones(21) / 21.0
    smoothed = np.convolve(profile, kernel, mode="same")

    grad = np.abs(np.gradient(smoothed))
    if len(grad) < 10:
        return VialMetrics(np.nan)

    idx_sorted = np.argsort(grad)[::-1]
    candidates = np.sort(idx_sorted[:10])

    if len(candidates) < 3:
        return VialMetrics(np.nan)

    y_top = int(candidates[0])
    y_bottom = int(candidates[-1])
    if y_bottom <= y_top + 10:
        y_top = int(np.min(candidates))
        y_bottom = int(np.max(candidates))

    if y_bottom <= y_top + 10:
        return VialMetrics(np.nan)

    internal_grad = grad[y_top:y_bottom]
    if len(internal_grad) < 5:
        return VialMetrics(np.nan)

    y_internal = int(y_top + np.argmax(internal_grad))

    h_total = y_bottom - y_top
    h_cream = max(0, y_internal - y_top)
    ci = h_cream / max(h_total, 1)

    return VialMetrics(creaming_index=float(np.clip(ci, 0.0, 1.0)))


# ============================================================
# 5. Objective function
# ============================================================

def compute_objective(
    img: ImageMetrics,
    target_outer_diameter_um: float = 350.0,
    target_core_outer_ratio: float = 0.60,
    target_shell_thickness_um: Optional[float] = None,
) -> ObjectiveSummary:
    """
    Lower is better.
    Objective is aligned with double-emulsion morphology in attached image type.
    """
    feasibility_penalty = 0.0

    if img.field_qc_pass == 0:
        feasibility_penalty += 2.0
    if img.n_detected_outer < 3:
        feasibility_penalty += 1.0

    if not np.isfinite(img.mean_outer_diameter_um):
        return ObjectiveSummary(
            objective_value=10.0,
            feasibility_penalty=feasibility_penalty + 3.0,
            total_score=13.0 + feasibility_penalty,
        )

    size_term = abs(img.mean_outer_diameter_um - target_outer_diameter_um) / max(target_outer_diameter_um, 1e-12)
    cv_term = img.cv_outer_percent / 10.0 if np.isfinite(img.cv_outer_percent) else 3.0
    ratio_term = abs(img.mean_core_outer_ratio - target_core_outer_ratio) / 0.10 if np.isfinite(img.mean_core_outer_ratio) else 3.0
    concentricity_term = img.mean_concentricity_ratio / 0.05 if np.isfinite(img.mean_concentricity_ratio) else 3.0
    weak_core_term = img.weak_core_fraction / 0.10 if np.isfinite(img.weak_core_fraction) else 3.0
    inner_fail_term = img.failed_inner_detection_fraction / 0.10 if np.isfinite(img.failed_inner_detection_fraction) else 3.0
    trunc_term = img.truncated_fraction / 0.10 if np.isfinite(img.truncated_fraction) else 3.0
    good_fraction_term = (1.0 - img.good_double_emulsion_fraction) / 0.20 if np.isfinite(img.good_double_emulsion_fraction) else 3.0

    shell_term = 0.0
    if target_shell_thickness_um is not None and np.isfinite(img.mean_shell_thickness_um):
        shell_term = abs(img.mean_shell_thickness_um - target_shell_thickness_um) / max(target_shell_thickness_um, 1e-12)

    objective_value = float(
        0.20 * size_term
        + 0.20 * cv_term
        + 0.15 * ratio_term
        + 0.10 * concentricity_term
        + 0.10 * weak_core_term
        + 0.10 * inner_fail_term
        + 0.05 * trunc_term
        + 0.10 * good_fraction_term
        + 0.00 * shell_term
    )

    return ObjectiveSummary(
        objective_value=objective_value,
        feasibility_penalty=feasibility_penalty,
        total_score=objective_value + feasibility_penalty,
    )


# ============================================================
# 6. Bayesian optimization
# ============================================================

class ObjectiveModel:
    """
    Continuous surrogate for objective score.
    Trained only on valid runs.
    """

    def __init__(self, random_state: int = 42):
        self.model = ExtraTreesRegressor(
            n_estimators=400,
            max_depth=10,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        )
        self.is_fitted = False

    @staticmethod
    def _to_feature_array(X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X.to_numpy(dtype=float, copy=False)
        return np.asarray(X, dtype=float)

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if len(X) < 8:
            self.is_fitted = False
            return
        X_arr = self._to_feature_array(X)
        y_arr = np.asarray(y, dtype=float)
        self.model.fit(X_arr, y_arr)
        self.is_fitted = True

    def predict_mean_std(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if not self.is_fitted:
            mu = np.full(len(X), np.inf, dtype=float)
            sigma = np.zeros(len(X), dtype=float)
            return mu, sigma

        X_arr = self._to_feature_array(X)
        all_tree_preds = np.vstack([est.predict(X_arr) for est in self.model.estimators_])
        mu = np.mean(all_tree_preds, axis=0)
        sigma = np.std(all_tree_preds, axis=0, ddof=1) if all_tree_preds.shape[0] > 1 else np.zeros(len(X), dtype=float)
        return mu.astype(float), sigma.astype(float)


class EmulsionBOConstrainedV2:
    """
    BO on objective score with a soft physical penalty.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = np.random.RandomState(random_state)

        self.space = [
            Real(0.0, 20.0, name="w1_glycerol_wt"),
            Real(0.5, 2.5, name="w1_tween20_wt"),
            Real(1.0, 6.0, name="o_dextrin_palmitate_wt"),
            Real(1.0, 2.5, name="o_span80_wt"),
            Real(0.2, 0.8, name="o_adm_wt"),
            Real(0.0, 20.0, name="w2_glycerol_wt"),
            Real(1.0, 2.5, name="w2_pva_wt"),
            Real(0.3, 2.5, name="w2_tween80_wt"),
            Real(1.0, 500.0, name="q_w1_ml_min"),
            Real(5.0, 500.0, name="q_o_ml_min"),
            Real(50.0, 500.0, name="q_w2_ml_min"),
            Real(50.0, 500.0, name="bath_stirring_speed_rpm"),
            Real(0.0, 0.3, name="bath_carbomer_wt"),
        ]

        self.objective_model = ObjectiveModel(random_state=random_state)
        self.history: List[Dict[str, float]] = []

    @property
    def param_names(self) -> List[str]:
        return [dim.name for dim in self.space]

    def _sample_uniform_candidates(self, n: int) -> List[Dict[str, float]]:
        out = []
        for _ in range(n):
            row = {}
            for dim in self.space:
                row[dim.name] = float(self.random_state.uniform(dim.low, dim.high))
            out.append(row)
        return out

    def _sample_local_candidates(self, n: int) -> List[Dict[str, float]]:
        if len(self.history) == 0:
            return []

        hist = pd.DataFrame(self.history)
        sort_col = "penalized_score" if "penalized_score" in hist.columns else "total_score"
        hist = hist[np.isfinite(hist[sort_col])].sort_values(sort_col)
        if hist.empty:
            return []

        anchors = hist.head(min(5, len(hist)))[self.param_names].to_dict(orient="records")
        out = []

        for _ in range(n):
            anchor = anchors[self.random_state.randint(len(anchors))]
            row = {}
            for dim in self.space:
                width = dim.high - dim.low
                step = 0.10 * width
                val = anchor[dim.name] + self.random_state.normal(0.0, step)
                val = float(np.clip(val, dim.low, dim.high))
                row[dim.name] = val
            out.append(row)

        return out

    def _soft_physical_penalty(self, params: Dict[str, float]) -> float:
        q_w1 = params["q_w1_ml_min"]
        q_o = params["q_o_ml_min"]
        q_w2 = params["q_w2_ml_min"]

        r_core = ratio_core(q_w1, q_o)
        r_sheath = ratio_sheath(q_w1, q_o, q_w2)

        penalty = 0.0
        if not (0.15 <= r_core <= 0.80):
            penalty += 1.5
        if not (2.0 <= r_sheath <= 20.0):
            penalty += 1.5
        if r_core > 0.60 and params["o_dextrin_palmitate_wt"] < 4.5:
            penalty += 0.8
        if params["w2_pva_wt"] < 1.3 and params["w2_tween80_wt"] > 1.0:
            penalty += 0.8

        return float(penalty)

    def tell(
        self,
        params: Dict[str, float],
        total_score: float,
        measurement_valid: int = 1,
        field_qc_pass: Optional[int] = None,
        metadata: Optional[Dict[str, float]] = None,
    ):
        if not np.isfinite(total_score):
            measurement_valid = 0

        penalized = (
            float(total_score) + self._soft_physical_penalty(params)
            if np.isfinite(total_score)
            else np.inf
        )

        row = params.copy()
        row["measurement_valid"] = int(measurement_valid)
        row["total_score"] = float(total_score) if np.isfinite(total_score) else np.nan
        row["penalized_score"] = float(penalized) if np.isfinite(penalized) else np.nan
        row["field_qc_pass"] = np.nan if field_qc_pass is None else int(field_qc_pass)

        if metadata is not None:
            for k, v in metadata.items():
                row[k] = v

        self.history.append(row)
        self.fit_models()

    def fit_models(self):
        if len(self.history) < 8:
            return

        hist = pd.DataFrame(self.history)

        valid_mask = (
            np.isfinite(hist["total_score"].values)
            & (hist["measurement_valid"].astype(int) == 1)
        )

        if "field_qc_pass" in hist.columns:
            qc_known = hist["field_qc_pass"].notna()
            valid_mask &= (~qc_known) | (hist["field_qc_pass"].fillna(1).astype(int) == 1)

        hist_obj = hist.loc[valid_mask].copy()
        if len(hist_obj) >= 8:
            X_obj = hist_obj[self.param_names]
            y_obj = hist_obj["total_score"].astype(float)
            self.objective_model.fit(X_obj, y_obj)

    def _current_best_score(self) -> Optional[float]:
        if len(self.history) == 0:
            return None

        hist = pd.DataFrame(self.history)
        mask = (
            (hist["measurement_valid"].astype(int) == 1)
            & np.isfinite(hist["total_score"].values)
        )

        if "field_qc_pass" in hist.columns:
            qc_known = hist["field_qc_pass"].notna()
            mask &= (~qc_known) | (hist["field_qc_pass"].fillna(1).astype(int) == 1)

        if not mask.any():
            return None

        return float(hist.loc[mask, "total_score"].min())

    def suggest(
        self,
        n_uniform: int = 512,
        n_local: int = 256,
        exploration_weight: float = 0.25,
        penalty_weight: float = 0.20,
    ) -> Dict[str, float]:
        candidates = self._sample_uniform_candidates(n_uniform) + self._sample_local_candidates(n_local)
        cand_df = pd.DataFrame(candidates)

        mu, sigma = self.objective_model.predict_mean_std(cand_df)
        penalties = np.array([self._soft_physical_penalty(r) for r in candidates], dtype=float)

        y_best = self._current_best_score()

        if y_best is None or not self.objective_model.is_fitted:
            score = -penalty_weight * penalties
        else:
            improvement = np.maximum(0.0, y_best - mu)
            exploration = sigma
            score = (
                improvement
                + exploration_weight * exploration
                - penalty_weight * penalties
            )

        best_idx = int(np.argmax(score))
        return candidates[best_idx]

    def suggest_with_diagnostics(
        self,
        n_uniform: int = 512,
        n_local: int = 256,
        exploration_weight: float = 0.25,
        penalty_weight: float = 0.20,
    ) -> pd.DataFrame:
        candidates = self._sample_uniform_candidates(n_uniform) + self._sample_local_candidates(n_local)
        cand_df = pd.DataFrame(candidates)

        mu, sigma = self.objective_model.predict_mean_std(cand_df)
        penalties = np.array([self._soft_physical_penalty(r) for r in candidates], dtype=float)

        y_best = self._current_best_score()

        if y_best is None or not self.objective_model.is_fitted:
            improvement = np.zeros(len(candidates), dtype=float)
            score = -penalty_weight * penalties
        else:
            improvement = np.maximum(0.0, y_best - mu)
            score = (
                improvement
                + exploration_weight * sigma
                - penalty_weight * penalties
            )

        out = cand_df.copy()
        out["pred_objective_mean"] = mu
        out["pred_objective_std"] = sigma
        out["pred_improvement"] = improvement
        out["soft_physical_penalty"] = penalties
        out["acquisition_score"] = score

        return out.sort_values("acquisition_score", ascending=False).reset_index(drop=True)

    def history_df(self) -> pd.DataFrame:
        if len(self.history) == 0:
            return pd.DataFrame()
        return pd.DataFrame(self.history).sort_values("penalized_score")


# ============================================================
# 7. Data registration and BO fitting
# ============================================================

def build_history_column_order(df: pd.DataFrame) -> List[str]:
    bo_cols = [name for name in bo_variable_names() if name != "batch_id"]
    condition_record_cols = [
        name
        for name in condition_field_names()
        if name not in {"batch_id", *bo_cols}
    ]
    preferred = [
        "record_type",
        "batch_id",
        *bo_cols,
        *condition_record_cols,
        "n_detected_outer",
        "n_valid_double_emulsion",
        "mean_outer_diameter_um",
        "median_outer_diameter_um",
        "std_outer_diameter_um",
        "cv_outer_percent",
        "pdi_outer_img",
        "mean_inner_diameter_um",
        "mean_shell_thickness_um",
        "std_shell_thickness_um",
        "cv_shell_percent",
        "mean_core_outer_ratio",
        "std_core_outer_ratio",
        "mean_concentricity_ratio",
        "mean_outer_circularity",
        "mean_inner_circularity",
        "weak_core_fraction",
        "truncated_fraction",
        "failed_inner_detection_fraction",
        "good_double_emulsion_fraction",
        "estimated_um_per_pixel_from_scale",
        "field_qc_pass",
        "creaming_index",
        "objective_value",
        "feasibility_penalty",
        "total_score",
        "measurement_valid",
        "pred_objective_mean",
        "pred_objective_std",
        "pred_improvement",
        "soft_physical_penalty",
        "acquisition_score",
    ]
    ordered = [col for col in preferred if col in df.columns]
    ordered.extend(col for col in df.columns if col not in ordered)
    return ordered


def finalize_history_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.replace([np.inf, -np.inf], np.nan).copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        df[numeric_cols] = df[numeric_cols].astype(float).round(3)
    return df[build_history_column_order(df)]


def register_result(
    out_csv: str,
    cond: ExperimentCondition,
    img: ImageMetrics,
    vial: Optional[VialMetrics],
    obj: ObjectiveSummary,
):
    row = {}
    row["record_type"] = "experiment"
    row.update(asdict(cond))
    row.update(asdict(img))

    if vial is not None:
        row.update(asdict(vial))
    else:
        row["creaming_index"] = np.nan

    row.update(asdict(obj))

    measurement_valid = int(
        np.isfinite(obj.total_score)
        and img.n_detected_outer > 0
        and img.field_qc_pass == 1
    )

    row["measurement_valid"] = measurement_valid

    if os.path.exists(out_csv):
        df = read_existing_history(out_csv)
        if "batch_id" in df.columns:
            df = df[df["batch_id"].astype(str) != str(cond.batch_id)].copy()
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df = finalize_history_frame(df)
    df.to_csv(out_csv, index=False, float_format="%.3f")
    return df


def fit_bo_from_history(
    history_csv: str,
) -> EmulsionBOConstrainedV2:
    df = read_existing_history(history_csv)
    bo = EmulsionBOConstrainedV2(random_state=42)

    required_cols = bo.param_names + ["total_score"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if len(missing_cols) > 0:
        raise ValueError(f"Missing required columns in history CSV: {missing_cols}")

    if "measurement_valid" not in df.columns:
        if "field_qc_pass" in df.columns:
            df["measurement_valid"] = (
                np.isfinite(df["total_score"])
                & (df["field_qc_pass"].fillna(0).astype(int) == 1)
            ).astype(int)
        else:
            df["measurement_valid"] = np.isfinite(df["total_score"]).astype(int)

    if "field_qc_pass" not in df.columns:
        df["field_qc_pass"] = np.nan

    for c in bo.param_names + ["total_score", "measurement_valid"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=bo.param_names).copy()

    for k, dim in zip(bo.param_names, bo.space):
        df = df[(df[k] >= dim.low) & (df[k] <= dim.high)]

    if df.empty:
        return bo

    meta_cols = [
        "batch_id",
        "time_h",
        "operator",
        "field_qc_pass",
        "n_detected_outer",
        "n_valid_double_emulsion",
        "mean_outer_diameter_um",
        "cv_outer_percent",
        "mean_core_outer_ratio",
        "mean_concentricity_ratio",
        "weak_core_fraction",
        "failed_inner_detection_fraction",
        "good_double_emulsion_fraction",
    ]
    existing_meta_cols = [c for c in meta_cols if c in df.columns]

    for _, row in df.iterrows():
        params = {k: float(row[k]) for k in bo.param_names}
        total_score = float(row["total_score"]) if np.isfinite(row["total_score"]) else np.nan
        measurement_valid = int(row["measurement_valid"]) if np.isfinite(row["measurement_valid"]) else 0

        field_qc_pass = None
        if "field_qc_pass" in row.index and np.isfinite(row["field_qc_pass"]):
            field_qc_pass = int(row["field_qc_pass"])

        metadata = {}
        for c in existing_meta_cols:
            val = row[c]
            if isinstance(val, str):
                metadata[c] = val
            elif pd.notna(val):
                metadata[c] = float(val)
            else:
                metadata[c] = np.nan

        bo.tell(
            params=params,
            total_score=total_score,
            measurement_valid=measurement_valid,
            field_qc_pass=field_qc_pass,
            metadata=metadata,
        )

    return bo


# ============================================================
# 8. High-level workflow
# ============================================================

def load_experiment_inputs_from_dir(
    experiment_dir: str,
) -> Tuple[ExperimentCondition, Optional[float]]:
    condition_json_path = os.path.join(experiment_dir, "condition.json")

    payload = load_json_dict(condition_json_path)

    um_per_pixel_raw = payload.get("um_per_pixel")
    um_per_pixel = float(um_per_pixel_raw) if um_per_pixel_raw is not None else None

    condition = dataclass_from_dict(ExperimentCondition, payload)
    return condition, um_per_pixel


def analyze_one_condition(
    microscopy_dir: str,
    um_per_pixel: Optional[float],
    batch_id: str,
    time_h: float,
    condition: ExperimentCondition,
    out_history_csv: str = "bo_history.csv",
    vial_image_path: Optional[str] = None,
    overlay_dir: Optional[str] = None,
    droplets_csv_path: Optional[str] = None,
    target_outer_diameter_um: float = 350.0,
    target_core_outer_ratio: float = 0.60,
    target_shell_thickness_um: Optional[float] = None,
    known_scale_um: float = 200.0,
    enable_empty_frame_relaxation: bool = True,
) -> pd.DataFrame:
    image_paths = list_image_files(microscopy_dir)

    if len(image_paths) == 0:
        raise ValueError(f"No microscopy images found in directory: {microscopy_dir}")

    droplet_df, img_metrics = analyze_microscopy_images(
        image_paths=image_paths,
        um_per_pixel=um_per_pixel,
        min_outer_diameter_um=50.0,
        max_outer_diameter_um=2000.0,
        save_overlay_dir=overlay_dir,
        try_scale_bar_estimation=True,
        known_scale_um=known_scale_um,
        enable_empty_frame_relaxation=enable_empty_frame_relaxation,
    )

    vial_metrics = analyze_vial_image_creaming_index(vial_image_path) if vial_image_path else VialMetrics(np.nan)

    condition_local = ExperimentCondition(**asdict(condition))
    condition_local.batch_id = batch_id
    condition_local.time_h = time_h

    objective = compute_objective(
        img=img_metrics,
        target_outer_diameter_um=target_outer_diameter_um,
        target_core_outer_ratio=target_core_outer_ratio,
        target_shell_thickness_um=target_shell_thickness_um,
    )

    df = register_result(
        out_csv=out_history_csv,
        cond=condition_local,
        img=img_metrics,
        vial=vial_metrics,
        obj=objective,
    )

    if not droplet_df.empty:
        droplets_csv = droplets_csv_path
        if droplets_csv is None:
            droplets_csv = "droplets.csv"
        droplet_df.to_csv(droplets_csv, index=False)

    return df


def analyze_experiment_dir(
    experiment_dir: str,
    out_history_csv: str = "bo_history.csv",
    overlay_root: Optional[str] = None,
    target_outer_diameter_um: float = 350.0,
    target_core_outer_ratio: float = 0.60,
    target_shell_thickness_um: Optional[float] = None,
    known_scale_um: float = 200.0,
    enable_empty_frame_relaxation: bool = True,
) -> pd.DataFrame:
    microscopy_dir = os.path.join(experiment_dir, "microscopy")

    condition, um_per_pixel = load_experiment_inputs_from_dir(experiment_dir)

    exp_name = os.path.basename(os.path.abspath(experiment_dir))
    batch_id = exp_name
    time_h = float(condition.time_h)

    if overlay_root is None:
        overlay_dir = os.path.join(experiment_dir, "overlays")
    else:
        overlay_dir = os.path.join(overlay_root, exp_name)

    droplets_csv_path = os.path.join(
        experiment_dir,
        "droplets.csv",
    )

    vial_image_path = find_optional_vial_image(experiment_dir)

    return analyze_one_condition(
        microscopy_dir=microscopy_dir,
        um_per_pixel=um_per_pixel,
        batch_id=batch_id,
        time_h=time_h,
        condition=condition,
        out_history_csv=out_history_csv,
        vial_image_path=vial_image_path,
        overlay_dir=overlay_dir,
        droplets_csv_path=droplets_csv_path,
        target_outer_diameter_um=target_outer_diameter_um,
        target_core_outer_ratio=target_core_outer_ratio,
        target_shell_thickness_um=target_shell_thickness_um,
        known_scale_um=known_scale_um,
        enable_empty_frame_relaxation=enable_empty_frame_relaxation,
    )


def run_experiment_batch(
    experiments_root: str = "./experiments",
    out_history_csv: str = "bo_history.csv",
    overlay_root: Optional[str] = None,
    target_outer_diameter_um: float = 350.0,
    target_core_outer_ratio: float = 0.60,
    target_shell_thickness_um: Optional[float] = None,
    known_scale_um: float = 200.0,
    skip_unchanged_batches: bool = True,
    enable_empty_frame_relaxation: bool = True,
) -> pd.DataFrame:
    experiment_dirs = discover_experiment_dirs(experiments_root)
    if len(experiment_dirs) == 0:
        raise ValueError(
            "No experiment directories found. Expected folders like "
            "'experiments/exp_001' with condition.json and microscopy/."
        )

    existing_batch_ids = existing_batch_ids_from_history(out_history_csv) if skip_unchanged_batches else set()
    existing_conditions = existing_condition_map(out_history_csv) if skip_unchanged_batches else {}

    df_last = read_existing_history(out_history_csv)
    for experiment_dir in experiment_dirs:
        batch_id = os.path.basename(os.path.abspath(experiment_dir))
        condition_payload = load_json_dict(os.path.join(experiment_dir, "condition.json"))
        condition_local = dataclass_from_dict(ExperimentCondition, condition_payload)
        condition_local.batch_id = batch_id
        current_condition_text = canonical_condition_json(condition_local)
        previous_condition_text = existing_conditions.get(batch_id)

        is_unchanged = False
        if batch_id in existing_batch_ids and previous_condition_text:
            is_unchanged = previous_condition_text == current_condition_text

        if is_unchanged:
            print(f"[skip] batch_id and inputs unchanged: {batch_id}")
            continue

        df_last = analyze_experiment_dir(
            experiment_dir=experiment_dir,
            out_history_csv=out_history_csv,
            overlay_root=overlay_root,
            target_outer_diameter_um=target_outer_diameter_um,
            target_core_outer_ratio=target_core_outer_ratio,
            target_shell_thickness_um=target_shell_thickness_um,
            known_scale_um=known_scale_um,
            enable_empty_frame_relaxation=enable_empty_frame_relaxation,
        )

    return df_last


def remove_file_if_exists(path: str):
    if os.path.isfile(path):
        os.remove(path)


def remove_tree_if_exists(path: str):
    if not os.path.exists(path):
        return
    if os.path.isfile(path):
        os.remove(path)
        return

    for root, dirnames, filenames in os.walk(path, topdown=False):
        for filename in filenames:
            os.remove(os.path.join(root, filename))
        for dirname in dirnames:
            os.rmdir(os.path.join(root, dirname))
    os.rmdir(path)


def reset_analysis_outputs(
    experiments_root: str = "./experiments",
    out_history_csv: str = "bo_history.csv",
    overlay_root: Optional[str] = None,
    bo_visualization_path: str = "bo_visualization.png",
):
    remove_file_if_exists(out_history_csv)
    remove_file_if_exists(bo_visualization_path)

    for experiment_dir in discover_experiment_dirs(experiments_root):
        remove_file_if_exists(os.path.join(experiment_dir, "droplets.csv"))
        if overlay_root is None:
            remove_tree_if_exists(os.path.join(experiment_dir, "overlays"))
        else:
            exp_name = os.path.basename(os.path.abspath(experiment_dir))
            remove_tree_if_exists(os.path.join(overlay_root, exp_name))


def build_suggested_condition_row(
    experiment_df: pd.DataFrame,
    suggestion_params: Dict[str, float],
) -> Dict[str, object]:
    base = ExperimentCondition()

    if not experiment_df.empty:
        last_row = experiment_df.iloc[-1]
        for field_name in condition_field_names():
            if field_name in last_row.index and pd.notna(last_row[field_name]):
                setattr(base, field_name, last_row[field_name])

    for k, v in suggestion_params.items():
        if hasattr(base, k):
            setattr(base, k, float(v))

    base.batch_id = "suggested_next_experiment"
    return asdict(base)


def save_bo_history_with_suggestion(
    experiment_csv_path: str,
    suggestion_df: pd.DataFrame,
):
    experiment_df = read_existing_history(experiment_csv_path)
    if suggestion_df.empty:
        finalize_history_frame(experiment_df).to_csv(
            experiment_csv_path,
            index=False,
            float_format="%.3f",
        )
        return

    top = suggestion_df.iloc[0].to_dict()
    suggestion_params = {
        k: float(top[k])
        for k in suggestion_df.columns
        if k in condition_field_names() and pd.notna(top[k])
    }
    suggestion_row: Dict[str, object] = {"record_type": "suggestion"}
    suggestion_row.update(build_suggested_condition_row(experiment_df, suggestion_params))

    for key, value in top.items():
        suggestion_row[key] = value

    out_df = pd.concat([experiment_df, pd.DataFrame([suggestion_row])], ignore_index=True, sort=False)
    out_df = finalize_history_frame(out_df)
    out_df.to_csv(experiment_csv_path, index=False, float_format="%.3f")


def save_bo_visualization(
    bo: EmulsionBOConstrainedV2,
    diag: pd.DataFrame,
    out_png: str = "bo_visualization.png",
    top_n: int = 40,
):
    mpl_cache_dir = os.path.join("/tmp", "wow_emulsion_mpl_cache")
    os.makedirs(mpl_cache_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", mpl_cache_dir)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if diag.empty:
        return

    plot_df = diag.head(min(top_n, len(diag))).copy()
    hist_df = bo.history_df().copy()
    suggestion = plot_df.iloc[0]

    plot_df["core_ratio"] = plot_df["q_w1_ml_min"] / np.maximum(plot_df["q_o_ml_min"], 1e-12)
    plot_df["sheath_ratio"] = plot_df["q_w2_ml_min"] / np.maximum(
        plot_df["q_w1_ml_min"] + plot_df["q_o_ml_min"], 1e-12
    )

    if not hist_df.empty:
        hist_df["core_ratio"] = hist_df["q_w1_ml_min"] / np.maximum(hist_df["q_o_ml_min"], 1e-12)
        hist_df["sheath_ratio"] = hist_df["q_w2_ml_min"] / np.maximum(
            hist_df["q_w1_ml_min"] + hist_df["q_o_ml_min"], 1e-12
        )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    sigma = plot_df["pred_objective_std"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    marker_sizes = 50.0 + 250.0 * (sigma / max(float(sigma.max()), 1e-12))
    sc = ax.scatter(
        plot_df["core_ratio"],
        plot_df["sheath_ratio"],
        c=plot_df["acquisition_score"],
        s=marker_sizes,
        cmap="viridis",
        alpha=0.8,
        edgecolors="black",
        linewidths=0.4,
        label="candidate",
    )
    if not hist_df.empty:
        ax.scatter(
            hist_df["core_ratio"],
            hist_df["sheath_ratio"],
            c="white",
            s=70,
            edgecolors="black",
            linewidths=1.2,
            marker="o",
            label="history",
            zorder=3,
        )

    sugg_core_ratio = float(suggestion["q_w1_ml_min"]) / max(float(suggestion["q_o_ml_min"]), 1e-12)
    sugg_sheath_ratio = float(suggestion["q_w2_ml_min"]) / max(
        float(suggestion["q_w1_ml_min"] + suggestion["q_o_ml_min"]),
        1e-12,
    )
    ax.scatter(
        [sugg_core_ratio],
        [sugg_sheath_ratio],
        marker="*",
        s=280,
        c="red",
        edgecolors="black",
        linewidths=1.0,
        label="next experiment",
        zorder=4,
    )
    ax.set_xlabel("Core Flow Ratio q_w1 / q_o")
    ax.set_ylabel("Sheath Flow Ratio q_w2 / (q_w1 + q_o)")
    ax.set_title("BO Candidate Map")
    ax.legend(loc="best")
    fig.colorbar(sc, ax=ax, label="Acquisition Score")

    ax = axes[1]
    rank_df = plot_df.reset_index(drop=True)
    ranks = np.arange(1, len(rank_df) + 1)
    finite_mu = np.isfinite(rank_df["pred_objective_mean"]).any()

    if finite_mu:
        mu = rank_df["pred_objective_mean"].astype(float).values
        yerr = rank_df["pred_objective_std"].astype(float).fillna(0.0).values
        ax.errorbar(ranks, mu, yerr=yerr, fmt="o", color="#1f77b4", ecolor="#7aa6d8", capsize=3)
        ax.set_ylabel("Predicted Objective Mean +/- Std")
        ax.set_title("Top Candidate Uncertainty")
    else:
        ax.bar(ranks, rank_df["acquisition_score"].astype(float).values, color="#1f77b4", alpha=0.85)
        ax.set_ylabel("Acquisition Score")
        ax.set_title("Top Candidates (Surrogate not fitted yet)")

    ax.axvline(1, color="red", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Candidate Rank")
    ax.set_xlim(0.5, len(rank_df) + 0.5)

    fig.suptitle("BO Recommendation Overview", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# 9. CLI entry point
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze experiment folders and fit BO suggestions."
    )
    parser.add_argument(
        "--experiments-root",
        default="./experiments",
        help="Directory containing experiment subfolders.",
    )
    parser.add_argument(
        "--out-history-csv",
        default="bo_history.csv",
        help="Output CSV for experiment history and appended BO suggestion.",
    )
    parser.add_argument(
        "--overlay-root",
        default=None,
        help="Optional root directory for overlays. Defaults to each experiment folder.",
    )
    parser.add_argument(
        "--known-scale-um",
        type=float,
        default=200.0,
        help="Physical length of the red microscopy scale bar, in micrometers.",
    )
    parser.add_argument(
        "--target-outer-diameter-um",
        type=float,
        default=350.0,
        help="Target outer droplet diameter for the objective.",
    )
    parser.add_argument(
        "--target-core-outer-ratio",
        type=float,
        default=0.60,
        help="Target inner-core to outer-droplet diameter ratio for the objective.",
    )
    parser.add_argument(
        "--target-shell-thickness-um",
        type=float,
        default=None,
        help="Optional target shell thickness for the objective.",
    )
    parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help="Re-run batches even if the batch_id and inputs are unchanged from the history CSV.",
    )
    parser.add_argument(
        "--reset-analysis",
        action="store_true",
        help=(
            "Delete existing analysis outputs and rebuild everything from scratch while "
            "keeping the experiments folders and input files."
        ),
    )
    parser.add_argument(
        "--no-empty-frame-relaxation",
        action="store_true",
        help=(
            "Disable the fallback that relaxes outer-droplet shape filtering when a frame "
            "would otherwise produce zero detections."
        ),
    )
    args = parser.parse_args()

    history_csv_path = args.out_history_csv
    bo_visualization_path = "bo_visualization.png"

    if args.reset_analysis:
        reset_analysis_outputs(
            experiments_root=args.experiments_root,
            out_history_csv=args.out_history_csv,
            overlay_root=args.overlay_root,
            bo_visualization_path=bo_visualization_path,
        )

    run_experiment_batch(
        experiments_root=args.experiments_root,
        out_history_csv=args.out_history_csv,
        overlay_root=args.overlay_root,
        target_outer_diameter_um=args.target_outer_diameter_um,
        target_core_outer_ratio=args.target_core_outer_ratio,
        target_shell_thickness_um=args.target_shell_thickness_um,
        known_scale_um=args.known_scale_um,
        skip_unchanged_batches=not args.force_reprocess,
        enable_empty_frame_relaxation=not args.no_empty_frame_relaxation,
    )

    bo = fit_bo_from_history(args.out_history_csv)

    diag = bo.suggest_with_diagnostics(
        n_uniform=512,
        n_local=256,
        exploration_weight=0.25,
        penalty_weight=0.20,
    )
    experiment_history_df = read_existing_history(args.out_history_csv)
    suggestion = build_suggested_condition_row(
        experiment_df=experiment_history_df,
        suggestion_params=diag.iloc[0][bo.param_names].to_dict(),
    )

    print("=== Suggested next experiment ===")
    for field_name in condition_field_names():
        value = suggestion[field_name]
        if isinstance(value, (int, float, np.floating)) and not isinstance(value, bool):
            print(f"{field_name}: {float(value):.3f}")
        else:
            print(f"{field_name}: {value}")

    save_bo_history_with_suggestion(
        experiment_csv_path=history_csv_path,
        suggestion_df=diag,
    )
    save_bo_visualization(
        bo=bo,
        diag=diag,
        out_png=bo_visualization_path,
    )


if __name__ == "__main__":
    main()
