#!/usr/bin/env python3
"""Regenerate the seeded polygonal-chain consistency calculations."""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from wlc_path_simulation import simulate

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "seeded_path_validation_regenerated.csv"

CASES = [
    (0.1, 0.0020, 40000, 2026092601),
    (0.1, 0.0010, 40000, 2026092602),
    (0.1, 0.0005, 40000, 2026092603),
    (1.0, 0.0100, 40000, 2026092611),
    (1.0, 0.0050, 40000, 2026092612),
    (1.0, 0.0025, 40000, 2026092613),
    (10.0, 0.0200, 20000, 2026092621),
    (10.0, 0.0100, 20000, 2026092622),
    (10.0, 0.0050, 20000, 2026092623),
]


def run_case(case: tuple[float, float, int, int]) -> dict[str, float | int]:
    lam, step, paths, seed = case
    mean, sem = simulate(lam, step, paths, seed)
    return {
        "lambda": lam,
        "delta_tau": step,
        "paths": paths,
        "seed": seed,
        "F_MC": mean,
        "SEM": sem,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    rows: list[dict[str, float | int]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_case, case): case for case in CASES}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print(row, flush=True)
    rows.sort(key=lambda r: (float(r["lambda"]), -float(r["delta_tau"])))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["lambda", "delta_tau", "paths", "seed", "F_MC", "SEM"])
        writer.writeheader()
        writer.writerows(rows)
    print("Wrote", args.output)


if __name__ == "__main__":
    main()
