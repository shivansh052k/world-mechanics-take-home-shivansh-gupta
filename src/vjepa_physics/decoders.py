"""Cross-check PyAV decoding against OpenCV's FFmpeg backend, within a per-clip tolerance."""
from pathlib import Path

import numpy as np

# Per-clip tolerance on PyAV - OpenCV, per RGB channel. Pixel-identical output is not achievable
# here: the two libraries bundle different FFmpeg builds and differ by about +1 in R and B.
FAIL_MAX_ABS = 8  # any channel's max |diff| above this fails
FAIL_MEAN_ABS = 2.0  # any channel's |whole-clip mean signed diff| above this fails
FAIL_DISK_MEAN_ABS = 5.0  # any channel's |mean signed diff over disk pixels| above this fails
FLAG_MAX_ABS = 3  # any channel's max |diff| above this is flagged
EXPECTED_MEAN = (1.0, 0.0, 1.0)  # typical whole-clip mean signed diff (R, G, B)
FLAG_MEAN_TOLERANCE = 0.5  # flagged if any channel's mean is further than this from EXPECTED_MEAN
DISK_THRESHOLD = 128  # disk pixels: brightest channel above this, in the PyAV frames


def load_clip_opencv(path: str | Path) -> tuple[np.ndarray, str]:
    """Independent decode with OpenCV's FFmpeg backend: (T, H, W, 3) uint8 RGB, backend name."""
    # Imported here only: cv2 bundles its own FFmpeg, and loading it next to PyAV's
    # duplicates some Objective-C classes on macOS. Only the decoder cross-check needs both.
    import cv2

    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open {path} with the FFmpeg backend")
    backend = cap.getBackendName()
    frames = []
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames), backend


def compare_decoders(pyav: np.ndarray, opencv: np.ndarray) -> dict:
    """Verdict "ok", "flag" or "fail" for one clip, with the reasons and per-channel statistics.

    Both inputs are (T, H, W, 3) uint8 RGB; differences are PyAV - OpenCV. A clip with no disk
    pixels is flagged, since the disk criterion cannot be evaluated on it.
    """
    if pyav.shape != opencv.shape:
        return {"verdict": "fail", "reasons": [f"shape {pyav.shape} vs {opencv.shape}"], "stats": None}

    diff = pyav.astype(np.int16) - opencv.astype(np.int16)
    disk = pyav.max(axis=-1) > DISK_THRESHOLD  # (T, H, W)
    max_abs = np.abs(diff).max(axis=(0, 1, 2))
    mean = diff.reshape(-1, 3).mean(axis=0)
    disk_mean = diff[disk].mean(axis=0) if disk.any() else None

    fail, flag = [], []
    for c, channel in enumerate("RGB"):
        if max_abs[c] > FAIL_MAX_ABS:
            fail.append(f"{channel}: max |diff| {max_abs[c]} > {FAIL_MAX_ABS}")
        elif max_abs[c] > FLAG_MAX_ABS:
            flag.append(f"{channel}: max |diff| {max_abs[c]} > {FLAG_MAX_ABS}")
        if abs(mean[c]) > FAIL_MEAN_ABS:
            fail.append(f"{channel}: |mean signed diff| {mean[c]:.3f} > {FAIL_MEAN_ABS}")
        if abs(mean[c] - EXPECTED_MEAN[c]) > FLAG_MEAN_TOLERANCE:
            flag.append(f"{channel}: mean signed diff {mean[c]:.3f} outside {EXPECTED_MEAN[c]} ± {FLAG_MEAN_TOLERANCE}")
        if disk_mean is not None and abs(disk_mean[c]) > FAIL_DISK_MEAN_ABS:
            fail.append(f"{channel}: |disk mean signed diff| {disk_mean[c]:.3f} > {FAIL_DISK_MEAN_ABS}")
    if disk_mean is None:
        flag.append("no disk pixels: disk criterion not evaluated")

    return {
        "verdict": "fail" if fail else "flag" if flag else "ok",
        "reasons": fail + flag,
        "stats": {
            "max_abs_diff": max_abs.tolist(),
            "mean_signed_diff": [round(float(v), 4) for v in mean],
            "disk_mean_signed_diff": None if disk_mean is None else [round(float(v), 4) for v in disk_mean],
            "disk_pixels": int(disk.sum()),
            "differing_pixels": int((diff != 0).any(axis=-1).sum()),
        },
    }