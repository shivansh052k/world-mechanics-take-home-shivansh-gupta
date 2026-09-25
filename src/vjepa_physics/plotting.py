"""Shared figure style: colours and axis styling for every figure in the project (light surface, slides)."""
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap

# Categorical colours: the first three slots of a validated palette (blue, orange, aqua), which stay
# distinguishable for every pair under colour-vision deficiencies. One fixed colour per dataset.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
DATASET_COLOUR = {"direction": SERIES[0], "speed": SERIES[1], "acceleration": SERIES[2]}

# Chart chrome and ink: text never takes a series colour.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# One-hue blue ramp, light to dark, for magnitudes (heatmaps).
SEQUENTIAL = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")


def sequential_cmap() -> LinearSegmentedColormap:
    """Colormap from the one-hue blue ramp (light = small, dark = large)."""
    return LinearSegmentedColormap.from_list("sequential_blue", SEQUENTIAL)


def style_axes(ax: Axes, grid_axis: str | None = "y") -> None:
    """Recessive axes: no top/right spines, hairline grid on `grid_axis` (None = no grid), muted ticks."""
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=AXIS, labelcolor=INK_SECONDARY, labelsize=9)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(INK_SECONDARY)
    ax.yaxis.label.set_color(INK_SECONDARY)