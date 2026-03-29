"""Tests for gs_gmlip.search.metadynamics module."""

import pytest

from gs_gmlip.search.metadynamics.collective_vars import (
    Angle,
    CoordinationNumber,
    Distance,
    Position,
    Torsion,
    build_cvs_from_config,
)
from gs_gmlip.search.metadynamics.plumed_interface import generate_plumed_input


class TestCollectiveVars:
    def test_distance(self):
        cv = Distance(label="d1", atom1=1, atom2=2)
        s = cv.to_plumed_string()
        assert "DISTANCE" in s
        assert "d1" in s
        assert "1,2" in s

    def test_angle(self):
        cv = Angle(label="a1", atom1=1, atom2=2, atom3=3)
        s = cv.to_plumed_string()
        assert "ANGLE" in s

    def test_torsion(self):
        cv = Torsion(label="t1", atom1=1, atom2=2, atom3=3, atom4=4)
        s = cv.to_plumed_string()
        assert "TORSION" in s

    def test_coordination(self):
        cv = CoordinationNumber(label="cn1", group_a=[1, 2], group_b=[3, 4, 5], r_0=2.5)
        s = cv.to_plumed_string()
        assert "COORDINATION" in s
        assert "R_0=2.5" in s

    def test_position(self):
        cv = Position(label="z1", atom=1, axis="z")
        s = cv.to_plumed_string()
        assert "POSITION" in s
        assert "COMPONENT=Z" in s

    def test_build_from_config(self):
        configs = [
            {"type": "distance", "label": "d1", "atom1": 1, "atom2": 2},
            {"type": "angle", "label": "a1", "atom1": 1, "atom2": 2, "atom3": 3},
        ]
        cvs = build_cvs_from_config(configs)
        assert len(cvs) == 2
        assert isinstance(cvs[0], Distance)
        assert isinstance(cvs[1], Angle)

    def test_unknown_cv_type(self):
        with pytest.raises(ValueError, match="Unknown CV type"):
            build_cvs_from_config([{"type": "foobar", "label": "x"}])


class TestPlumedInput:
    def test_standard_metadynamics(self):
        cvs = [Distance(label="d1", atom1=1, atom2=2)]
        inp = generate_plumed_input(cvs, height=0.1, pace=500)
        assert "METAD" in inp
        assert "d1" in inp
        assert "HEIGHT=0.1" in inp
        assert "PACE=500" in inp

    def test_well_tempered(self):
        cvs = [Distance(label="d1", atom1=1, atom2=2)]
        inp = generate_plumed_input(
            cvs, height=0.1, pace=500, biasfactor=10.0, temperature=300.0
        )
        assert "BIASFACTOR=10.0" in inp
        assert "TEMP=300.0" in inp

    def test_multiple_cvs(self):
        cvs = [
            Distance(label="d1", atom1=1, atom2=2),
            Distance(label="d2", atom1=3, atom2=4),
        ]
        inp = generate_plumed_input(cvs, sigma=[0.1, 0.2])
        assert "d1,d2" in inp
        assert "0.1000,0.2000" in inp
