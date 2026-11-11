#!/usr/bin/env python3
"""Generate and extrapolate the independent finite-volume validation data."""
from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from wlc_finite_volume import FVSettings, solve_fv

LAMBDAS = (0.1, 1.0, 10.0)
NY_VALUES = (200, 400, 800)
NU_VALUES = (24, 32, 48)


def evaluate(task: tuple[float, int, int]) -> dict[str, float]:
    lam, ny, nu = task
    val = solve_fv(lam, FVSettings(ny, nu, lam))
    return {"lambda": lam, "Ny": ny, "Nu": nu, "Ymax": lam, "F_raw": val}


def spatial_extrapolate(ny: np.ndarray, values: np.ndarray) -> dict[str, float]:
    x = 1.0 / ny.astype(float)
    Xq = np.column_stack([np.ones_like(x), x, x*x])
    bq, *_ = np.linalg.lstsq(Xq, values, rcond=None)
    # Fine-grid linear check using the two finest values.
    Xl = np.column_stack([np.ones(2), x[-2:]])
    bl, *_ = np.linalg.lstsq(Xl, values[-2:], rcond=None)
    return {
        "F_spatial_quad": float(bq[0]),
        "F_spatial_linear_fine": float(bl[0]),
        "spatial_sensitivity": float(abs(bq[0]-bl[0])),
    }


def angular_extrapolate(nu: np.ndarray, values: np.ndarray) -> dict[str, float]:
    x = 1.0 / nu.astype(float)**2
    Xq = np.column_stack([np.ones_like(x), x, x*x])
    bq, *_ = np.linalg.lstsq(Xq, values, rcond=None)
    Xl = np.column_stack([np.ones(3), x[-3:]])
    bl, *_ = np.linalg.lstsq(Xl, values[-3:], rcond=None)
    return {
        "F_continuum_quad": float(bq[0]),
        "F_continuum_linear_fine": float(bl[0]),
        "angular_sensitivity": float(abs(bq[0]-bl[0])),
    }


def _task_key(row: dict[str, float]) -> tuple[float, int, int]:
    return (float(row["lambda"]), int(row["Ny"]), int(row["Nu"]))


def _load_raw(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def _write_raw(path: Path, rows: list[dict[str, float]]) -> None:
    rows = sorted(rows, key=lambda r: (r["lambda"], r["Nu"], r["Ny"]))
    tmp = path.with_suffix(path.suffix+".tmp")
    with tmp.open("w", newline="") as f:
        w=csv.DictWriter(f,fieldnames=["lambda","Ny","Nu","Ymax","F_raw"]); w.writeheader(); w.writerows(rows)
    tmp.replace(path)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=(Path(__file__).resolve().parents[1] / "data"))
    p.add_argument("--reference", type=Path, default=(Path(__file__).resolve().parents[1] / "data" / "wlc_mean_span_reference.csv"))
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--fresh", action="store_true")
    args = p.parse_args()
    out = args.out_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    raw_path=out/"finite_volume_tensor_grid.csv"
    rows=[] if args.fresh else _load_raw(raw_path)
    existing={_task_key(r) for r in rows}
    tasks=[(lam,ny,nu) for lam in LAMBDAS for nu in NU_VALUES for ny in NY_VALUES if (lam,ny,nu) not in existing]
    print(f"Existing tasks: {len(rows)}; pending: {len(tasks)}", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures={ex.submit(evaluate,t):t for t in tasks}
        for i,fut in enumerate(as_completed(futures),1):
            row=fut.result(); rows.append(row); _write_raw(raw_path,rows)
            print(f"[{i:02d}/{len(tasks)}] lambda={row['lambda']:g} Ny={row['Ny']} Nu={row['Nu']} F={row['F_raw']:.10g}",flush=True)
    rows=sorted(rows,key=lambda r:(r["lambda"],r["Nu"],r["Ny"]))

    spatial_rows=[]; continuum_rows=[]
    for lam in LAMBDAS:
        for nu in NU_VALUES:
            subset=[r for r in rows if r["lambda"]==lam and r["Nu"]==nu]
            ny=np.array([r["Ny"] for r in subset],float); val=np.array([r["F_raw"] for r in subset],float)
            ext=spatial_extrapolate(ny,val); spatial_rows.append({"lambda":lam,"Nu":nu,**ext})
        ss=[r for r in spatial_rows if r["lambda"]==lam]
        nu=np.array([r["Nu"] for r in ss],float); vals=np.array([r["F_spatial_quad"] for r in ss],float)
        ext2=angular_extrapolate(nu,vals)
        spatial_sensitivity = max(r["spatial_sensitivity"] for r in ss)
        angular_sensitivity = ext2["angular_sensitivity"]
        # Conservative additive diagnostic: spatial and angular extrapolation
        # variations are kept separate and summed, rather than silently
        # replacing one by the larger of the two.
        total_sens = spatial_sensitivity + angular_sensitivity
        continuum_rows.append({
            "lambda":lam,
            **ext2,
            "spatial_sensitivity_abs_max":spatial_sensitivity,
            "angular_sensitivity_abs":angular_sensitivity,
            "combined_sensitivity_abs":total_sens,
        })
    with (out/"finite_volume_spatial_extrapolations.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(spatial_rows[0].keys()));w.writeheader();w.writerows(spatial_rows)
    with (out/"finite_volume_continuum.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(continuum_rows[0].keys()));w.writeheader();w.writerows(continuum_rows)

    # Join to the released spectral continuum table so the comparison is
    # generated by this workflow rather than maintained as a hand-edited file.
    with args.reference.resolve().open(newline="") as f:
        spectral = {float(r["lambda"]): float(r["F_continuum"]) for r in csv.DictReader(f)}
    comparison_rows = []
    for row in continuum_rows:
        lam = float(row["lambda"])
        fspec = spectral[lam]
        ffv = float(row["F_continuum_quad"])
        comparison_rows.append({
            "lambda": lam,
            "F_spectral": fspec,
            "F_continuum_quad": ffv,
            "relative_difference_pct": 100.0 * (ffv / fspec - 1.0),
            "FV_spatial_sensitivity_rel_pct": 100.0 * float(row["spatial_sensitivity_abs_max"]) / abs(ffv),
            "FV_angular_sensitivity_rel_pct": 100.0 * float(row["angular_sensitivity_abs"]) / abs(ffv),
            "FV_combined_sensitivity_rel_pct": 100.0 * float(row["combined_sensitivity_abs"]) / abs(ffv),
        })
    with (out/"finite_volume_comparison.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(comparison_rows[0].keys()));w.writeheader();w.writerows(comparison_rows)

    print(json.dumps({"continuum": continuum_rows, "comparison": comparison_rows},indent=2))


if __name__ == "__main__":
    main()
