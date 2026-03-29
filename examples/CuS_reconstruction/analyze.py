"""Analyze CuS reconstruction GCMC search results.

Loads search outputs, performs deduplication, characterizes unique structures,
and generates summary plots + tables.

Usage:
    python analyze.py                      # analyze all results
    python analyze.py --workdir search     # specify work directory
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read, write

from gs_gmlip.analysis.structure import StructureAnalyzer, pair_distribution
from gs_gmlip.analysis.trajectory import TrajectoryAnalyzer


def load_trajectory(traj_file: Path) -> list[dict]:
    """Load trajectory JSON file."""
    with open(traj_file) as f:
        return json.load(f)


def analyze_single_potential(work_dir: Path) -> dict | None:
    """Analyze results for a single potential."""
    summary_file = work_dir / "summary.json"
    traj_file = work_dir / "trajectory.json"
    best_file = work_dir / "best.xyz"

    if not summary_file.exists():
        return None

    with open(summary_file) as f:
        summary = json.load(f)

    result = {"potential": summary.get("potential_V", 0.0), **summary}

    # Load trajectory
    if traj_file.exists():
        trajectory = load_trajectory(traj_file)
        traj_analyzer = TrajectoryAnalyzer(trajectory)

        conv = traj_analyzer.convergence_check(window=50, threshold=0.05)
        result["converged"] = conv.get("converged", False)
        result["energy_change"] = conv.get("energy_change", np.nan)

        summary_stats = traj_analyzer.summary()
        result["trajectory_summary"] = summary_stats

    # Load best structure
    if best_file.exists():
        best = read(str(best_file))
        result["best_n_atoms"] = len(best)
        result["best_formula"] = best.get_chemical_formula()

        # Adsorbate composition
        tags = best.get_tags()
        active_idx = [i for i, t in enumerate(tags) if t > 0]
        if active_idx:
            symbols = [best.symbols[i] for i in active_idx]
            comp = {}
            for s in symbols:
                comp[s] = comp.get(s, 0) + 1
            result["adsorbate_composition"] = comp

    return result


def plot_potential_scan(results: list[dict], output_dir: Path):
    """Plot formation energy vs potential."""
    potentials = [r["potential"] for r in results]
    energies = [r.get("best_energy_eV", np.nan) for r in results]
    n_ads = [r.get("n_adsorbates", 0) for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Energy vs potential
    axes[0].plot(potentials, energies, "bo-", markersize=8, linewidth=2)
    axes[0].set_xlabel("U (V vs RHE)", fontsize=12)
    axes[0].set_ylabel("Best Energy (eV)", fontsize=12)
    axes[0].set_title("Formation Energy vs Potential")
    axes[0].grid(True, alpha=0.3)

    # Number of adsorbates vs potential
    axes[1].bar(potentials, n_ads, width=0.3, color="steelblue", alpha=0.8)
    axes[1].set_xlabel("U (V vs RHE)", fontsize=12)
    axes[1].set_ylabel("Number of Adsorbates", fontsize=12)
    axes[1].set_title("Surface Coverage vs Potential")
    axes[1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(str(output_dir / "potential_scan.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'potential_scan.png'}")


def plot_trajectory(traj_file: Path, output_dir: Path, label: str = ""):
    """Plot energy evolution from trajectory."""
    trajectory = load_trajectory(traj_file)
    if not trajectory:
        return

    traj_analyzer = TrajectoryAnalyzer(trajectory)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # (a) Energy evolution
    steps = traj_analyzer.steps
    energies = traj_analyzer.energies
    best = traj_analyzer.best_energies
    axes[0, 0].plot(steps, energies, "b-", alpha=0.4, label="Current")
    axes[0, 0].plot(steps, best, "r-", lw=2, label="Best")
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Energy (eV)")
    axes[0, 0].set_title(f"Energy Evolution {label}")
    axes[0, 0].legend()

    # (b) Acceptance rate
    acc_rate = traj_analyzer.rolling_acceptance_rate(window=20)
    axes[0, 1].plot(range(len(acc_rate)), acc_rate, "g-", lw=1.5)
    axes[0, 1].set_xlabel("Step")
    axes[0, 1].set_ylabel("Acceptance Rate")
    axes[0, 1].set_title("Rolling Acceptance Rate")
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].axhline(0.2, color="gray", ls="--", alpha=0.5, label="20%")
    axes[0, 1].legend()

    # (c) Move statistics
    move_types = set()
    for t in trajectory:
        if "move" in t:
            move_types.add(t["move"])
    move_types = sorted(move_types)

    def _is_accepted(val):
        if isinstance(val, str):
            return val.lower() == "true"
        return bool(val)

    accepted_counts = []
    rejected_counts = []
    for mt in move_types:
        acc = sum(
            1
            for t in trajectory
            if t.get("move") == mt and _is_accepted(t.get("accepted", False))
        )
        rej = sum(
            1
            for t in trajectory
            if t.get("move") == mt and not _is_accepted(t.get("accepted", False))
        )
        accepted_counts.append(acc)
        rejected_counts.append(rej)

    x = np.arange(len(move_types))
    axes[1, 0].bar(
        x - 0.15, accepted_counts, 0.3, label="Accepted", color="green", alpha=0.7
    )
    axes[1, 0].bar(
        x + 0.15, rejected_counts, 0.3, label="Rejected", color="red", alpha=0.7
    )
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(move_types, rotation=45)
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_title("Move Statistics")
    axes[1, 0].legend()

    # (d) N_adsorbates evolution
    n_ads = [t.get("n_adsorbates", 0) for t in trajectory]
    axes[1, 1].plot(range(len(n_ads)), n_ads, "m-", alpha=0.7)
    axes[1, 1].set_xlabel("Step")
    axes[1, 1].set_ylabel("N adsorbates")
    axes[1, 1].set_title("Adsorbate Count Evolution")

    plt.tight_layout()
    safe_label = label.replace(" ", "_").replace("=", "").replace(".", "p")
    plt.savefig(
        str(output_dir / f"trajectory{safe_label}.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    print(f"Saved: {output_dir / f'trajectory{safe_label}.png'}")


def main():
    parser = argparse.ArgumentParser(description="Analyze CuS reconstruction results")
    parser.add_argument("--workdir", default="search", help="Search working directory")
    args = parser.parse_args()

    work_dir = Path(args.workdir)
    if not work_dir.exists():
        print(f"Work directory not found: {work_dir}")
        return

    # Find all potential directories
    potential_dirs = sorted(work_dir.glob("U_*V"))
    if not potential_dirs:
        print("No potential scan directories found.")
        return

    print(f"Found {len(potential_dirs)} potential directories")

    # Analyze each potential
    results = []
    for pd in potential_dirs:
        print(f"\nAnalyzing {pd.name}...")
        r = analyze_single_potential(pd)
        if r is not None:
            results.append(r)

            # Plot trajectory
            traj_file = pd / "trajectory.json"
            if traj_file.exists():
                plot_trajectory(traj_file, work_dir, label=f" ({pd.name})")

    if not results:
        print("No results to analyze.")
        return

    # Print summary table
    print("\n" + "=" * 80)
    print("CuS Reconstruction — Potential Scan Summary")
    print("=" * 80)
    print(
        f"{'U (V)':>8} {'Best E (eV)':>12} {'N_ads':>6} "
        f"{'Acc Rate':>9} {'Conv':>5} {'Formula':>25}"
    )
    print("-" * 80)
    for r in results:
        print(
            f"{r['potential']:>8.1f} "
            f"{r.get('best_energy_eV', np.nan):>12.4f} "
            f"{r.get('n_adsorbates', 0):>6d} "
            f"{r.get('acceptance_rate', 0):>9.3f} "
            f"{'Yes' if r.get('converged') else 'No':>5} "
            f"{r.get('best_formula', 'N/A'):>25}"
        )
    print("=" * 80)

    # Plot potential scan if multiple potentials
    if len(results) > 1:
        plot_potential_scan(results, work_dir)

    # Save combined analysis
    # Convert numpy types to native Python for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return obj

    serializable = json.loads(json.dumps(results, default=convert))
    with open(str(work_dir / "analysis_summary.json"), "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nAnalysis saved to {work_dir / 'analysis_summary.json'}")


if __name__ == "__main__":
    main()
