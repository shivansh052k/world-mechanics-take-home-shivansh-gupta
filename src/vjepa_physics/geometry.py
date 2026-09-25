"""Where the disk is, from its metadata: straight-line motion with constant acceleration.

A clip's disk starts at start_position_xy_m and moves along its direction theta. After t seconds it
has travelled s(t) = v t + a t^2 / 2 metres and moves at v + a t. Frame k shows the disk at t = k / fps,
so frame 0 shows the start position.
"""
import numpy as np

# Pixel mapping (verified on one clip; checked on every clip by disk tracking): 32 px per metre, world
# origin at pixel 128, image rows pointing down (y flipped). Pixel coordinates are pixel indices
# (col, row), like a mask centroid, so a 256 px image spans -0.5 ... 255.5 on both axes.
PX_PER_M = 32
ORIGIN_PX = 128
IMAGE_SIZE = 256


def frame_times(fps: int, frames: int) -> np.ndarray:
    """Time of each frame in seconds: frame k at k / fps."""
    return np.arange(frames) / fps


def distance_travelled(
    speed: float | np.ndarray, acceleration: float | np.ndarray, t: float | np.ndarray
) -> np.ndarray:
    """Distance along the direction of motion after t seconds: v t + a t^2 / 2 (metres).

    Elementwise on arrays; always returns an ndarray (0-d for scalar inputs).
    """
    t = np.asarray(t, dtype=float)
    return np.asarray(np.asarray(speed) * t + 0.5 * np.asarray(acceleration) * t**2)


def speed_at(speed: float | np.ndarray, acceleration: float | np.ndarray, t: float | np.ndarray) -> np.ndarray:
    """Speed after t seconds: v + a t (metres per second). Elementwise; always returns an ndarray."""
    return np.asarray(np.asarray(speed) + np.asarray(acceleration) * np.asarray(t, dtype=float))


def world_to_pixel(x: float | np.ndarray, y: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World position in metres -> (col, row) in pixel-index coordinates; y is flipped."""
    col = ORIGIN_PX + PX_PER_M * np.asarray(x, dtype=float)
    row = ORIGIN_PX - PX_PER_M * np.asarray(y, dtype=float)
    return np.asarray(col), np.asarray(row)


def pixel_to_world(col: float | np.ndarray, row: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(col, row) in pixel-index coordinates -> world position in metres; inverse of world_to_pixel (y flipped)."""
    x = (np.asarray(col, dtype=float) - ORIGIN_PX) / PX_PER_M
    y = (ORIGIN_PX - np.asarray(row, dtype=float)) / PX_PER_M
    return np.asarray(x), np.asarray(y)


def disk_centres(meta: dict) -> np.ndarray:
    """(frames, 2) predicted disk centre (col, row) in every frame of a clip, from its metadata."""
    t = frame_times(meta["fps"], meta["frames"])
    s = distance_travelled(meta["speed_mps"], meta["acceleration_mps2"], t)
    theta = np.deg2rad(meta["theta_degrees"])
    x0, y0 = meta["start_position_xy_m"]
    col, row = world_to_pixel(x0 + s * np.cos(theta), y0 + s * np.sin(theta))
    return np.stack([col, row], axis=1)


def distance_outside_image(
    col: float | np.ndarray, row: float | np.ndarray, size: int = IMAGE_SIZE
) -> np.ndarray:
    """Distance in px from (col, row) to the nearest point of the image; 0 if the point is inside.

    The image covers pixel indices 0 ... size - 1, i.e. the square -0.5 ... size - 0.5. A disk of
    radius r centred at (col, row) lies entirely outside the image exactly when this exceeds r.
    """
    col, row = np.asarray(col, dtype=float), np.asarray(row, dtype=float)
    dx = np.maximum(0.0, np.maximum(-0.5 - col, col - (size - 0.5)))
    dy = np.maximum(0.0, np.maximum(-0.5 - row, row - (size - 0.5)))
    return np.asarray(np.hypot(dx, dy))


# Patch grid (16 x 16 patches of 16 px, patch index = row * 16 + col) and the rendered disk's radius:
# a disk of 350 px area (measured 346-353 px on fully visible frames) has radius sqrt(350 / pi) = 10.56 px.
PATCH_PX = 16
PATCH_GRID = IMAGE_SIZE // PATCH_PX  # 16
DISK_RADIUS_PX = float(np.sqrt(350 / np.pi))


def distance_to_patches(col: np.ndarray, row: np.ndarray) -> np.ndarray:
    """(...,) points in pixel-index coordinates -> (..., 256) distance in px from each point to each patch square.

    Patch (r, c) covers pixel indices 16r ... 16r + 15 (rows) and 16c ... 16c + 15 (cols), i.e. the square
    16c - 0.5 ... 16c + 15.5 (cols) and likewise for rows; the distance is 0 inside it. Axis -1 is the patch
    index r * 16 + c, the token layout's order.
    """
    col, row = np.asarray(col, dtype=float), np.asarray(row, dtype=float)
    low = np.arange(PATCH_GRID) * PATCH_PX - 0.5
    high = low + PATCH_PX
    dx = np.maximum(0.0, np.maximum(low - col[..., None], col[..., None] - high))  # (..., 16) per patch column
    dy = np.maximum(0.0, np.maximum(low - row[..., None], row[..., None] - high))  # (..., 16) per patch row
    return np.hypot(dy[..., :, None], dx[..., None, :]).reshape(*col.shape, PATCH_GRID * PATCH_GRID)