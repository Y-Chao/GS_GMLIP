"""Command-line interface for GS_GMLIP.

Usage:
    gs_gmlip --config config.yaml
    gs_gmlip --help
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

logger = logging.getLogger("gs_gmlip")


def setup_logging(level: str = "INFO"):
    """Configure logging for the package."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path) as f:
        config = yaml.safe_load(f)
    return config


def build_evaluator(eval_config: dict):
    """Build evaluator from config."""
    backend = eval_config.get("backend", "ase")
    if backend == "mace":
        from gs_gmlip.evaluate.mlip.mace_eval import MACEEvaluator

        return MACEEvaluator(
            model=eval_config.get("model", "medium"),
            device=eval_config.get("device", "cpu"),
            default_dtype=eval_config.get("dtype", "float64"),
            dispersion=eval_config.get("dispersion", False),
        )
    elif backend == "emt":
        from ase.calculators.emt import EMT

        from gs_gmlip.evaluate.mlip.ase_calc import ASECalculatorEvaluator

        return ASECalculatorEvaluator(EMT())
    elif backend == "ase":
        raise ValueError(
            "ASE backend requires specifying a calculator. "
            "Use 'mace' or 'emt' backend, or provide a calculator directly."
        )
    else:
        raise ValueError(f"Unknown evaluator backend: {backend}")


def build_searcher(search_config: dict, evaluator, workdir: str):
    """Build searcher from config."""
    method = search_config.get("method", "ga")

    if method == "ga":
        from gs_gmlip.search.ga.runner import GARunner

        return GARunner.from_config(search_config, evaluator, workdir)
    elif method == "gcmc":
        from gs_gmlip.search.gcmc.runner import GCMCRunner

        return GCMCRunner.from_config(search_config, evaluator, workdir)
    elif method == "gofee":
        from gs_gmlip.search.gofee.runner import GOFEERunner

        return GOFEERunner.from_config(search_config, evaluator, workdir)
    elif method == "ssw":
        from gs_gmlip.search.ssw.runner import SSWRunner

        return SSWRunner.from_config(search_config, evaluator, workdir)
    elif method == "metadynamics":
        from gs_gmlip.search.metadynamics.runner import MetadynamicsRunner

        return MetadynamicsRunner.from_config(search_config, evaluator, workdir)
    else:
        raise ValueError(f"Unknown search method: {method}")


def run_workflow(config: dict):
    """Execute a full search workflow from config."""
    # Setup
    workdir = config.get("workdir", "gs_gmlip_work")
    setup_logging(config.get("log_level", "INFO"))

    logger.info("Starting GS_GMLIP workflow")
    logger.info(f"Working directory: {workdir}")

    # Build evaluator
    eval_config = config.get("evaluate", {})
    evaluator = build_evaluator(eval_config)
    logger.info(f"Evaluator: {type(evaluator).__name__}")

    # Build searcher
    search_config = config.get("search", {})
    searcher = build_searcher(search_config, evaluator, workdir)
    logger.info(f"Searcher: {type(searcher).__name__}")

    # Run search
    n_steps = search_config.get("n_steps", 100)
    searcher.setup()
    searcher.run(n_steps=n_steps)

    # Analysis (if configured)
    analysis_config = config.get("analysis", {})
    if analysis_config:
        results = searcher.get_results()
        logger.info(f"Search completed. {len(results)} structures found.")

    logger.info("Workflow completed.")


def main():
    """Main entry point for gs_gmlip CLI."""
    parser = argparse.ArgumentParser(
        prog="gs_gmlip",
        description="GS_GMLIP: Global Structure Searching with MLIP",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="gs_gmlip 1.0.0",
    )

    args = parser.parse_args()

    try:
        config = load_config(args.config)
        run_workflow(config)
    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
