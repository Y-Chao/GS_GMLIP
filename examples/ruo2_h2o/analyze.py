"""Analyze RuO2(110)/H2O search results."""

from pathlib import Path

from ase.db import connect
from ase.io import read

from gs_gmlip.analysis.structure import StructureAnalyzer
from gs_gmlip.analysis.visualization import save_best_structures


def analyze_ga(work_dir: str = "ga_ruo2_h2o"):
    db_path = Path(work_dir) / "candidates.db"
    if not db_path.exists():
        print(f"No database at {db_path}")
        return

    db = connect(str(db_path))
    structures, energies = [], []
    for row in db.select("relaxed=1"):
        structures.append(row.toatoms())
        energies.append(row.energy)

    if not structures:
        print("No relaxed structures in GA database.")
        return

    print(f"GA: {len(structures)} relaxed structures")
    analyzer = StructureAnalyzer(structures, energies)
    summary = analyzer.energy_landscape_summary()
    print(f"  Energy: [{summary['energy_min']:.3f}, {summary['energy_max']:.3f}] eV")

    unique = analyzer.deduplicate(threshold=0.05)
    print(f"  Unique: {len(unique)} / {len(structures)}")

    save_best_structures(
        [structures[i] for i in unique],
        [energies[i] for i in unique],
        n=10,
        output_dir=f"{work_dir}/best",
        fmt="extxyz",
    )


def analyze_gcmc(work_dir: str = "gcmc_ruo2_h2o"):
    best_file = Path(work_dir) / "best.xyz"
    if best_file.exists():
        atoms = read(str(best_file))
        print(
            f"GCMC best: E = {atoms.get_potential_energy():.4f} eV, "
            f"{len(atoms)} atoms"
        )


if __name__ == "__main__":
    analyze_ga()
    analyze_gcmc()
