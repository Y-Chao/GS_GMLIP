"""Tests for gs_gmlip.data module."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import fcc111

from gs_gmlip.data.database import CandidateDatabase, PrepareDatabase
from gs_gmlip.data.io import read_structures, write_structures
from gs_gmlip.data.manager import WorkflowManager


class TestCandidateDatabase:
    def test_create_and_add(self, cu111_slab, tmp_path):
        db_path = str(tmp_path / "test.db")
        prep = PrepareDatabase(db_path, slab=cu111_slab)

        db = CandidateDatabase(db_path)
        assert db.n_unrelaxed >= 0

    def test_add_candidate(self, cu111_slab, tmp_path):
        db_path = str(tmp_path / "test.db")
        prep = PrepareDatabase(db_path, slab=cu111_slab)

        db = CandidateDatabase(db_path)
        candidate = cu111_slab.copy()
        candidate.info["key_value_pairs"] = {"generation": 0}
        db.add_unrelaxed_candidate(candidate, description="test:TestCandidate")
        assert db.n_unrelaxed >= 1


class TestIO:
    def test_write_and_read(self, cu111_slab, tmp_path):
        path = tmp_path / "test.xyz"
        write_structures(str(path), [cu111_slab], format="extxyz")
        result = read_structures(str(path))
        assert len(result) == 1
        assert len(result[0]) == len(cu111_slab)


class TestWorkflowManager:
    def test_create(self, tmp_path):
        wm = WorkflowManager(workdir=str(tmp_path / "wf"))
        assert (tmp_path / "wf").exists()

    def test_checkpoint(self, tmp_path):
        wm = WorkflowManager(workdir=str(tmp_path / "wf"))
        state = {"step": 10, "best_energy": -5.0}
        wm.save_checkpoint(state)

        loaded = wm.load_checkpoint()
        assert loaded["step"] == 10
        assert loaded["best_energy"] == -5.0
