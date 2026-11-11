#!/usr/bin/env python3
"""Pool the four heat-kernel path batches at lambda = 1e4 (4 x 20 000 chains).

Reads the retained per-chain records in data/heat_path_samples/ written by
sphere_heat_path_validation.py --long and --long-extra, and writes
data/heat_path_long_pooled.csv: the pooled span estimate, its standard error and the
paired difference from the finest grid at each of the four coupled resolutions, with
the batch means for reference.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LAM, STEPS, SEEDS = 10000.0, 131072, (8161310, 8161311, 8161312, 8161313)


def main() -> None:
    packs = [np.load(ROOT / "data" / "heat_path_samples" / f"heat_l{LAM:g}_n{STEPS}_N20000_seed{s}.npz") for s in SEEDS]
    fine = np.concatenate([p["span_stride1"] for p in packs])
    with (ROOT / "data" / "heat_path_long_pooled.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["lam", "steps", "dt", "n", "batches", "span", "span_se", "pairwise_minus_fine", "pairwise_se",
                    "R2", "R2_se", "batch_means"])
        for k in (8, 4, 2, 1):
            x = np.concatenate([p[f"span_stride{k}"] for p in packs])
            r2 = np.concatenate([p[f"R2_stride{k}"] for p in packs])
            d = x - fine
            n = len(x)
            w.writerow([repr(LAM), STEPS // k, repr(LAM * k / STEPS), n, len(packs), repr(float(x.mean())),
                        repr(float(x.std(ddof=1) / math.sqrt(n))), repr(float(d.mean())),
                        repr(float(d.std(ddof=1) / math.sqrt(n))), repr(float(r2.mean())),
                        repr(float(r2.std(ddof=1) / math.sqrt(n))),
                        " ".join(f"{float(p[f'span_stride{k}'].mean()):.4f}" for p in packs)])
    print("pooled finest-grid span:", fine.mean(), "+-", fine.std(ddof=1) / math.sqrt(len(fine)))


if __name__ == "__main__":
    main()
