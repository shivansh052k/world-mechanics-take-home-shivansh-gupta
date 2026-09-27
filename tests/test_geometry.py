"""Kinematics and the pixel mapping behind every label and tracking check."""
import numpy as np

from vjepa_physics.geometry import distance_travelled, frame_times, pixel_to_world, speed_at, world_to_pixel


def test_constant_acceleration_kinematics():
    t = frame_times(24, 16)
    assert t[-1] == 15 / 24  # a clip spans frame 0 to frame 15
    np.testing.assert_allclose(distance_travelled(2.0, 4.0, t), 2.0 * t + 0.5 * 4.0 * t**2)
    fine = np.linspace(0.0, 1.0, 10_001)  # speed = d(distance)/dt; central differences are exact for a quadratic
    derivative = np.gradient(distance_travelled(2.0, 4.0, fine), fine)
    np.testing.assert_allclose(derivative[1:-1], speed_at(2.0, 4.0, fine)[1:-1], atol=1e-9)


def test_pixel_mapping_origin_scale_and_flipped_y():
    col, row = world_to_pixel(np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0]))
    np.testing.assert_array_equal(col, [128.0, 160.0, 128.0])  # origin at pixel 128, 32 px per metre
    np.testing.assert_array_equal(row, [128.0, 128.0, 96.0])  # +y (up on screen) is a smaller row index


def test_pixel_mapping_round_trip():
    rng = np.random.default_rng(0)
    x, y = rng.uniform(-4, 4, 100), rng.uniform(-4, 4, 100)
    np.testing.assert_allclose(pixel_to_world(*world_to_pixel(x, y)), (x, y), atol=1e-12)