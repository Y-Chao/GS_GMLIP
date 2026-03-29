"""Analyze Cu/CSHO search results.

Post-processing: deduplicate, cluster, summarize energy landscape,
and export best structures.
"""

from pathlib import Path

from ase.io import read

from gs_gmlip.analysis.structure import StructureAnalyzer
from gs_gmlip.analysis.visualization import save_best_structures, write_convergence_csv


def analyze_ga_results(work_dir: str = "ga_cu111_csho"):
    """Analyze GA results."""
    db_path = Path(work_dir) / "candidates.db"
    if not db_path.exists():
        print(f"No database found at {db_path}")
        return

    from ase.db import connect

    db = connect(str(db_path))
    structures = []
    energies = []
    for row in db.select("relaxed=1"):
        structures.append(row.toatoms())
        energies.append(row.energy)

    if not structures:
        print("No relaxed structures found.")
        return

    print(f"GA: {len(structures)} relaxed structures")

    analyzer = StructureAnalyzer(structures, energies)

    # Summary
    summary = analyzer.energy_landscape_summary()
    print(
        f"  Energy range: [{summary['energy_min']:.3f}, {summary['energy_max']:.3f}] eV"
    )
    print(
        f"  Energy mean ± std: {summary['energy_mean']:.3f} ± {summary['energy_std']:.3f} eV"
    )

    # Deduplicate
    unique = analyzer.deduplicate(threshold=0.05)
    print(f"  Unique structures: {len(unique)} / {len(structures)}")

    # Save best
    unique_structs = [structures[i] for i in unique]
    unique_energies = [energies[i] for i in unique]
    save_best_structures(
        unique_structs,
        unique_energies,
        n=10,
        output_dir=f"{work_dir}/best",
        fmt="extxyz",
    )
    print(f"  Best structures saved to {work_dir}/best/")


def analyze_gcmc_results(work_dir: str = "gcmc_cu100_csho"):
    """Analyze GCMC results."""
    best_file = Path(work_dir) / "best.xyz"
    if best_file.exists():
        atoms = read(str(best_file))
        print(
            f"GCMC best: E = {atoms.get_potential_energy():.4f} eV, "
            f"{len(atoms)} atoms"
        )


if __name__ == "__main__":
    analyze_ga_results()
    analyze_gcmc_results()
