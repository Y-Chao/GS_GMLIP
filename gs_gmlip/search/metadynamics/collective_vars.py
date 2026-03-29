"""Collective variable definitions for Metadynamics.

Provides helper classes that generate PLUMED-compatible CV
specifications from atomic structure definitions.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CollectiveVariable:
    """Base class for collective variables.

    Subclasses must implement `to_plumed_string()` to produce the
    PLUMED input syntax for this CV.
    """

    label: str

    def to_plumed_string(self) -> str:
        raise NotImplementedError


@dataclass
class Distance(CollectiveVariable):
    """Distance between two atoms.

    Parameters
    ----------
    label : str
        PLUMED label for this CV.
    atom1 : int
        1-based atom index.
    atom2 : int
        1-based atom index.
    """

    atom1: int
    atom2: int

    def to_plumed_string(self) -> str:
        return f"{self.label}: DISTANCE ATOMS={self.atom1},{self.atom2}"


@dataclass
class Angle(CollectiveVariable):
    """Angle between three atoms.

    Parameters
    ----------
    label : str
        PLUMED label.
    atom1, atom2, atom3 : int
        1-based atom indices.
    """

    atom1: int
    atom2: int
    atom3: int

    def to_plumed_string(self) -> str:
        return f"{self.label}: ANGLE ATOMS={self.atom1},{self.atom2},{self.atom3}"


@dataclass
class Torsion(CollectiveVariable):
    """Torsion angle between four atoms."""

    atom1: int
    atom2: int
    atom3: int
    atom4: int

    def to_plumed_string(self) -> str:
        return (
            f"{self.label}: TORSION "
            f"ATOMS={self.atom1},{self.atom2},{self.atom3},{self.atom4}"
        )


@dataclass
class CoordinationNumber(CollectiveVariable):
    """Coordination number CV.

    Parameters
    ----------
    label : str
        PLUMED label.
    group_a : list of int
        1-based indices of central atoms.
    group_b : list of int
        1-based indices of neighbor atoms.
    r_0 : float
        Switching function distance parameter (Å).
    nn : int
        Numerator exponent.
    mm : int
        Denominator exponent.
    """

    group_a: list[int]
    group_b: list[int]
    r_0: float = 2.5
    nn: int = 6
    mm: int = 12

    def to_plumed_string(self) -> str:
        ga = ",".join(str(a) for a in self.group_a)
        gb = ",".join(str(b) for b in self.group_b)
        return (
            f"{self.label}: COORDINATION GROUPA={ga} GROUPB={gb} "
            f"R_0={self.r_0} NN={self.nn} MM={self.mm}"
        )


@dataclass
class Position(CollectiveVariable):
    """Position of an atom along an axis (x, y, or z)."""

    atom: int
    axis: str = "z"  # "x", "y", or "z"

    def to_plumed_string(self) -> str:
        component = self.axis.upper()
        return f"{self.label}: POSITION ATOM={self.atom} COMPONENT={component}"


def build_cvs_from_config(cv_configs: list[dict]) -> list[CollectiveVariable]:
    """Build CVs from a list of config dicts.

    Parameters
    ----------
    cv_configs : list of dict
        Each dict has 'type' and type-specific parameters.

    Returns
    -------
    cvs : list of CollectiveVariable
    """
    builders = {
        "distance": lambda c: Distance(
            label=c["label"], atom1=c["atom1"], atom2=c["atom2"]
        ),
        "angle": lambda c: Angle(
            label=c["label"], atom1=c["atom1"], atom2=c["atom2"], atom3=c["atom3"]
        ),
        "torsion": lambda c: Torsion(
            label=c["label"],
            atom1=c["atom1"],
            atom2=c["atom2"],
            atom3=c["atom3"],
            atom4=c["atom4"],
        ),
        "coordination": lambda c: CoordinationNumber(
            label=c["label"],
            group_a=c["group_a"],
            group_b=c["group_b"],
            r_0=c.get("r_0", 2.5),
        ),
        "position": lambda c: Position(
            label=c["label"], atom=c["atom"], axis=c.get("axis", "z")
        ),
    }

    cvs = []
    for conf in cv_configs:
        cv_type = conf["type"].lower()
        if cv_type not in builders:
            raise ValueError(f"Unknown CV type: {cv_type}. Available: {list(builders)}")
        cvs.append(builders[cv_type](conf))
    return cvs
