"""SAM-based sky segmentation for GLOBE observations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
import requests
import torch

# MPS requires float32; float64 is not fully supported
torch.set_default_dtype(torch.float32)

# Default grid size and batch size for SAM (avoids internal float64 grid creation)
SAM_POINTS_PER_SIDE = 24
SAM_POINTS_PER_BATCH = 64

logger = logging.getLogger(__name__)


def _build_point_grid_float32(n_per_side: int) -> np.ndarray:
    """Build a 2D point grid in [0,1]^2 with explicit float32 to avoid MPS float64 errors."""
    offset = 1.0 / (2.0 * n_per_side)
    points_one_side = np.linspace(offset, 1.0 - offset, n_per_side, dtype=np.float32)
    points_x = np.tile(points_one_side[None, :], (n_per_side, 1))
    points_y = np.tile(points_one_side[:, None], (1, n_per_side))
    points = np.stack([points_x, points_y], axis=-1).reshape(-1, 2)
    return points

# Project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Valid sky directions (exclude Down, Ground, Calibration)
_DIRECTIONS = ("Up", "North", "South", "East", "West")
_DIR_COLUMNS = {
    "Up": "skyconditionsUpwardPhotoUrl",
    "North": "skyconditionsNorthPhotoUrl",
    "South": "skyconditionsSouthPhotoUrl",
    "East": "skyconditionsEastPhotoUrl",
    "West": "skyconditionsWestPhotoUrl",
}
_CAPTION_COLUMNS = {
    "Up": "skyconditionsUpwardCaption",
    "North": "skyconditionsNorthCaption",
    "South": "skyconditionsSouthCaption",
    "East": "skyconditionsEastCaption",
    "West": "skyconditionsWestCaption",
}

_INVALID_URL_VALUES = frozenset({None, "", "null", "pending approval", "nan"})


def _is_valid_url(url: str | object) -> bool:
    """Check if URL is valid and fetchable."""
    if url is None or (isinstance(url, float) and np.isnan(url)):
        return False
    s = str(url).strip().lower()
    if not s or s in _INVALID_URL_VALUES:
        return False
    return s.startswith("http")


def _should_skip_caption(caption: str | object) -> bool:
    """Skip if caption contains ground or calibration."""
    if caption is None or (isinstance(caption, float) and np.isnan(caption)):
        return False
    c = str(caption).strip().lower()
    return "ground" in c or "calibration" in c


def load_sam_model(
    checkpoint_path: str | Path,
    device: str | torch.device = "mps",
    points_per_side: int = SAM_POINTS_PER_SIDE,
    points_per_batch: int = SAM_POINTS_PER_BATCH,
) -> "SamAutomaticMaskGenerator":
    """Load SAM vit_b and return SamAutomaticMaskGenerator with float32 point grid for MPS."""
    torch.set_default_dtype(torch.float32)
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    path = Path(checkpoint_path)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(
            f"SAM checkpoint not found: {path}. Download from "
            "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth "
            "and place in models/sam_vit_b_01ec64.pth"
        )
    dev = torch.device(device)
    if str(device) == "mps" and not torch.backends.mps.is_available():
        logger.warning("MPS not available; falling back to CPU")
        dev = torch.device("cpu")

    point_grid = _build_point_grid_float32(points_per_side)
    point_grids = [point_grid]

    sam = sam_model_registry["vit_b"](checkpoint=str(path))
    sam = sam.float()
    sam = sam.to(dev)
    return SamAutomaticMaskGenerator(
        sam,
        points_per_side=None,
        point_grids=point_grids,
        points_per_batch=points_per_batch,
    )


def _mask_to_numpy(mask: "np.ndarray | torch.Tensor") -> np.ndarray:
    """Convert mask to numpy, handling tensors for MPS compatibility."""
    if hasattr(mask, "detach"):
        return mask.detach().cpu().to(torch.float32).numpy()
    return np.asarray(mask)

def get_sky_mask(image: np.ndarray, mask_generator: "SamAutomaticMaskGenerator") -> np.ndarray | None:
    """
    Generate masks and select sky region: largest contiguous region in upper half.
    Uses inference_mode and MPS autocast to keep all intermediates in float32.
    Returns binary mask (H, W) or None if no suitable mask found.
    """

    # 1. Get the current device from the model
    model_dev = next(mask_generator.predictor.model.parameters()).device
    
    # 2. Attempt the MPS segmentation
    try:
        with torch.inference_mode():
            # If on MPS, we still attempt the autocast block
            if model_dev.type == "mps":
                with torch.amp.autocast(device_type="mps", enabled=True):
                    masks = mask_generator.generate(image)
            else:
                masks = mask_generator.generate(image)
    except TypeError as e:
        if "float64" in str(e) and model_dev.type == "mps":
            logger.warning("MPS float64 error detected. Falling back to CPU for segmentation.")
            
            # Switch model to CPU temporarily
            mask_generator.predictor.model.to("cpu")
            masks = mask_generator.generate(image)
            # Switch back
            mask_generator.predictor.model.to("mps")
        else:
            raise e

    if not masks:
        logger.warning("No masks generated")
        return None
    h, w = image.shape[:2]
    upper_half_y = h / 2.0
    candidates = []
    for ann in masks:
        seg = ann.get("segmentation")
        if seg is None:
            continue
        seg = _mask_to_numpy(seg)
        bbox = ann.get("bbox", [0, 0, w, h])
        x, y, bw, bh = bbox
        y_center = y + bh / 2.0
        if y_center < upper_half_y:
            area = int(ann.get("area", 0))
            candidates.append((area, seg))
    if not candidates:
        logger.warning("No mask in upper half; using largest overall")
        best = max(masks, key=lambda m: int(m.get("area", 0)))
        seg = best.get("segmentation")
        if seg is None:
            return None
        seg = _mask_to_numpy(seg)
        return seg.astype(bool) if seg.dtype != bool else seg
    _, best_mask = max(candidates, key=lambda t: t[0])
    arr = _mask_to_numpy(best_mask)
    return arr.astype(bool) if arr.dtype != bool else arr


def calculate_metrics(image: np.ndarray, mask: np.ndarray) -> dict:
    """
    Compute Sky Index B/(R+G) for masked pixels.
    Returns dict with mean_index, std_dev, mask_pixel_count.
    """
    eps = 1e-8
    r = image[:, :, 0].astype(np.float64)
    g = image[:, :, 1].astype(np.float64)
    b = image[:, :, 2].astype(np.float64)
    denom = r + g + eps
    idx = np.where(mask)
    if len(idx[0]) == 0:
        return {"mean_index": np.nan, "std_dev": np.nan, "mask_pixel_count": 0}
    b_vals = b[idx]
    denom_vals = denom[idx]
    sky_index = b_vals / denom_vals
    return {
        "mean_index": float(np.nanmean(sky_index)),
        "std_dev": float(np.nanstd(sky_index)),
        "mask_pixel_count": int(np.sum(mask)),
    }


def load_image_from_url(url: str) -> np.ndarray | None:
    """Fetch image from URL and return RGB numpy array (H, W, 3) uint8, or None on failure."""
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        logger.warning("Failed to fetch %s: %s", url[:80], e)
        return None
    try:
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(resp.content))
        arr = np.asarray(img.convert("RGB"))
        return arr
    except Exception as e:
        logger.warning("Failed to decode image from %s: %s", url[:80], e)
        return None


def save_verification_thumbnail(
    image: np.ndarray,
    mask: np.ndarray,
    output_path: Path,
    max_size: int = 640,
) -> None:
    """Save thumbnail with semi-transparent green sky mask overlay."""
    try:
        import cv2
    except ImportError:
        logger.warning("opencv-python not installed; skipping thumbnail")
        return
    h, w = image.shape[:2]
    scale = min(1.0, max_size / max(h, w))
    if scale < 1:
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(mask.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST).astype(bool)
    else:
        img = image.copy()
        msk = mask
    overlay = img.copy()
    overlay[msk] = (overlay[msk] * 0.5 + np.array([0, 128, 0]) * 0.5).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def collect_observations_from_csv(
    csv_path: Path,
) -> Generator[tuple[str, str, str, str], None, None]:
    """
    Yield (observation_id, timestamp, direction, url) for valid sky images.
    Excludes Down, Ground, Calibration; skips invalid URLs.
    """
    df = pd.read_csv(csv_path)
    id_col = "skyconditionsObservationId"
    ts_col = "skyconditionsMeasuredAt"
    for _, row in df.iterrows():
        obs_id = str(row.get(id_col, "")).strip()
        if not obs_id or obs_id == "nan":
            continue
        timestamp = str(row.get(ts_col, "")).strip()
        for direction in _DIRECTIONS:
            url_col = _DIR_COLUMNS.get(direction)
            cap_col = _CAPTION_COLUMNS.get(direction)
            if not url_col:
                continue
            url_val = row.get(url_col)
            if not _is_valid_url(url_val):
                continue
            caption = row.get(cap_col) if cap_col else None
            if _should_skip_caption(caption):
                continue
            yield obs_id, timestamp, direction, str(url_val).strip()


def run_segmentation(
    input_csv: Path,
    output_csv: Path,
    output_masks_dir: Path,
    checkpoint_path: str | Path = "models/sam_vit_b_01ec64.pth",
    device: str | torch.device = "mps",
    project_root: Path | None = None,
) -> int:
    """
    Run sky segmentation pipeline on observations from input CSV.
    Returns 0 on success, 1 on failure.
    """
    root = project_root or _PROJECT_ROOT
    in_path = input_csv if input_csv.is_absolute() else root / input_csv
    out_path = output_csv if output_csv.is_absolute() else root / output_csv
    masks_dir = output_masks_dir if output_masks_dir.is_absolute() else root / output_masks_dir

    if not in_path.exists():
        logger.error("Input CSV not found: %s", in_path)
        return 1

    logger.info("Loading SAM model (device=%s)...", device)
    try:
        mask_generator = load_sam_model(checkpoint_path, device)
    except FileNotFoundError as e:
        logger.error("%s", e)
        return 1

    results: list[dict] = []
    for obs_id, timestamp, direction, url in collect_observations_from_csv(in_path):
        image = load_image_from_url(url)
        if image is None:
            continue
        mask = get_sky_mask(image, mask_generator)
        if mask is None:
            logger.warning("No sky mask for %s %s", obs_id, direction)
            continue
        metrics = calculate_metrics(image, mask)
        row = {
            "observation_id": obs_id,
            "timestamp": timestamp,
            "direction": direction,
            "mean_index": metrics["mean_index"],
            "std_dev": metrics["std_dev"],
            "mask_pixel_count": metrics["mask_pixel_count"],
        }
        results.append(row)
        thumb_path = masks_dir / f"{obs_id}_{direction}.png"
        save_verification_thumbnail(image, mask, thumb_path)
        logger.info("Processed %s %s: mean_index=%.4f", obs_id, direction, row["mean_index"])

    if not results:
        logger.warning("No observations processed")
    else:
        out_df = pd.DataFrame(results)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)
        logger.info("Saved %d rows to %s", len(out_df), out_path)
    return 0
