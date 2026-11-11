#!/usr/bin/env python3
"""Extended finite-volume spot audit with withheld finer grids.

The primary validation uses Ny={200,400,800} and Nu={24,32,48} at
lambda={0.1,1,10}.  This script adds Ny=1600 and Nu=64 at lambda=0.1, 1 and 10,
where the calculations remain inexpensive enough for routine auditing.  The
extra grids are not used by the primary spectral reference.
"""
from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from wlc_finite_volume import FVSettings, solve_fv

LAMBDAS = (0.1, 1.0, 10.0)
NY_VALUES = (200, 400, 800, 1600)
NU_VALUES = (24, 32, 48, 64)


def read_csv(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(handle)]


def write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    rows = sorted(rows, key=lambda r: (r["lambda"], r["Nu"], r["Ny"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["lambda", "Ny", "Nu", "Ymax", "F_raw"])
        writer.writeheader()
        writer.writerows(rows)


def evaluate(task: tuple[float, int, int]) -> dict[str, float]:
    lam, ny, nu = task
    return {
        "lambda": lam,
        "Ny": ny,
        "Nu": nu,
        "Ymax": lam,
        "F_raw": solve_fv(lam, FVSettings(ny, nu, lam)),
    }


def intercept(x: np.ndarray, y: np.ndarray, degree: int) -> float:
    matrix = np.column_stack([x**j for j in range(degree + 1)])
    beta, *_ = np.linalg.lstsq(matrix, y, rcond=None)
    return float(beta[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-grid", type=Path, default=(Path(__file__).resolve().parents[1] / "data" / "finite_volume_tensor_grid.csv"))
    parser.add_argument("--reference", type=Path, default=(Path(__file__).resolve().parents[1] / "data" / "wlc_mean_span_reference.csv"))
    parser.add_argument("--out-dir", type=Path, default=(Path(__file__).resolve().parents[1] / "data"))
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()

    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    path = out / "finite_volume_extended_tensor_grid.csv"

    rows: list[dict[str, float]] = [] if args.fresh else read_csv(path)
    # Reuse the primary tensor-grid calculations where applicable.
    if not rows:
        for row in read_csv(args.base_grid.resolve()):
            if row["lambda"] in LAMBDAS and int(row["Ny"]) in NY_VALUES and int(row["Nu"]) in NU_VALUES:
                rows.append(row)
    existing = {(r["lambda"], int(r["Ny"]), int(r["Nu"])) for r in rows}
    tasks = [
        (lam, ny, nu)
        for lam in LAMBDAS
        for nu in NU_VALUES
        for ny in NY_VALUES
        if (lam, ny, nu) not in existing
    ]
    print(f"Existing extended-audit tasks: {len(rows)}; pending: {len(tasks)}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(evaluate, task): task for task in tasks}
        for i, future in enumerate(as_completed(futures), 1):
            row = future.result()
            rows.append(row)
            write_csv(path, rows)
            print(
                f"[{i:02d}/{len(tasks)}] lambda={row['lambda']:g} Ny={int(row['Ny'])} "
                f"Nu={int(row['Nu'])} F={row['F_raw']:.11g}",
                flush=True,
            )
    write_csv(path, rows)

    with args.reference.resolve().open(newline="") as handle:
        spectral = {float(r["lambda"]): float(r["F_continuum"]) for r in csv.DictReader(handle)}

    summary: list[dict[str, float]] = []
    for lam in LAMBDAS:
        spatial = []
        for nu in NU_VALUES:
            subset = sorted(
                [r for r in rows if r["lambda"] == lam and int(r["Nu"]) == nu],
                key=lambda r: r["Ny"],
            )
            ny = np.array([r["Ny"] for r in subset], dtype=float)
            values = np.array([r["F_raw"] for r in subset], dtype=float)
            x = 1.0 / ny
            quad = intercept(x, values, 2)
            cubic = intercept(x, values, 3)
            fine_linear = intercept(x[-3:], values[-3:], 1)
            spatial.append((nu, quad, cubic, fine_linear))
        nu = np.array([r[0] for r in spatial], dtype=float)
        x = 1.0 / nu**2
        quad_values = np.array([r[1] for r in spatial])
        angular_quad = intercept(x, quad_values, 2)
        angular_cubic = intercept(x, quad_values, 3)
        angular_fine_linear = intercept(x[-3:], quad_values[-3:], 1)
        spatial_spread = max(max(abs(r[1] - r[2]), abs(r[1] - r[3])) for r in spatial)
        angular_spread = max(abs(angular_quad - angular_cubic), abs(angular_quad - angular_fine_linear))
        diagnostic = spatial_spread + angular_spread
        fspec = spectral[lam]
        summary.append(
            {
                "lambda": lam,
                "F_spectral": fspec,
                "F_FV_extended": angular_quad,
                "relative_difference_pct": 100.0 * (angular_quad / fspec - 1.0),
                "spatial_sensitivity_abs": spatial_spread,
                "angular_sensitivity_abs": angular_spread,
                "combined_sensitivity_abs": diagnostic,
                "combined_sensitivity_rel_pct": 100.0 * diagnostic / abs(angular_quad),
            }
        )

    summary_path = out / "finite_volume_extended_comparison.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    print(f"Wrote {path} and {summary_path}")
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
