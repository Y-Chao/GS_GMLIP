"""Tests for the Region classes."""

from __future__ import annotations

import numpy as np
import pytest

from gs_gmlip.search.region import BoxRegion, SphereRegion


class TestBoxRegion:
    """Tests for BoxRegion."""

    @pytest.fixture
    def rng(self):
        return np.random.default_rng(42)

    @pytest.fixture
    def box(self):
        return BoxRegion(
            cell=np.array([[10, 0, 0], [0, 10, 0], [0, 0, 20]]),
            z_min=15.0,
            z_max=20.0,
            margin=1.0,
        )

    def test_random_position_returns_3d(self, box, rng):
        """random_position returns a (3,) array."""
        pos = box.random_position(rng)
        assert pos.shape == (3,)
        assert pos.dtype == np.float64

    def test_random_position_z_in_range(self, box, rng):
        """z coordinate is within [z_min, z_max]."""
        for _ in range(100):
            pos = box.random_position(rng)
            assert box.z_min <= pos[2] <= box.z_max

    def test_random_position_x_in_range(self, box, rng):
        """x coordinate is within [margin, ||a|| - margin]."""
        a_norm = np.linalg.norm(box.cell[0])
        for _ in range(100):
            pos = box.random_position(rng)
            assert box.margin <= pos[0] <= a_norm - box.margin

    def test_random_position_y_in_range(self, box, rng):
        """y coordinate is within [margin, ||b|| - margin]."""
        b_norm = np.linalg.norm(box.cell[1])
        for _ in range(100):
            pos = box.random_position(rng)
            assert box.margin <= pos[1] <= b_norm - box.margin


class TestSphereRegion:
    """Tests for SphereRegion."""

    @pytest.fixture
    def rng(self):
        return np.random.default_rng(42)

    @pytest.fixture
    def sphere(self):
        return SphereRegion(center=np.array([5.0, 5.0, 5.0]), radius=3.0)

    def test_random_position_in_sphere(self, sphere, rng):
        """All sampled points lie within the sphere."""
        for _ in range(100):
            pos = sphere.random_position(rng)
            dist = np.linalg.norm(pos - sphere.center)
            assert dist <= sphere.radius + 1e-10

    def test_random_position_coverage(self, sphere, rng):
        """Points are distributed throughout the sphere (not all at center)."""
        distances = []
        for _ in range(50):
            pos = sphere.random_position(rng)
            distances.append(np.linalg.norm(pos - sphere.center))
        # At least some points should be far from the center
        assert max(distances) > sphere.radius * 0.3
        # And some near (covers the interior)
        assert min(distances) < sphere.radius * 0.9
