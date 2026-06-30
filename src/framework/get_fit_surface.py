#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fit the scaling law to a user-specified set of already-trained cells.

Reads base_dir and dataset from the YAML config, finds each cell's
metrics.csv, extracts the requested metric, then fits and prints
the power-law surface.

Run:
    cd src/
    python framework/get_fit_surface.py \
        --config configs/experiment_sw.yaml \
        --cells  4h:3 4h:12 24h:3 24h:12 \
        --metric RMSE
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import argparse

import pandas as pd
import yaml

from framework.fit_surface import fit_surface, print_fit_summary
from framework.train import data_dir_path


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def get_metric_values(config, cells, metric_name):
    """
    Read metric_name from metrics.csv for each (resolution, horizon) cell.

    Parameters
    ----------
    config      : dict — must contain base_dir and dataset keys
    cells       : list of (resolution_str, horizon_int)
    metric_name : str  — e.g. "RMSE", "MAE", "MSE"

    Returns
    -------
    list of float, one value per cell in the same order as cells
    """
    col = metric_name.upper()
    metric_values = []

    for resolution, horizon in cells:
        ddir         = data_dir_path(config, resolution, horizon)
        metrics_path = os.path.join(ddir, "metrics.csv")

        if not os.path.exists(metrics_path):
            raise FileNotFoundError(
                f"No metrics.csv for ({resolution}, h={horizon}) at:\n"
                f"  {metrics_path}\n"
                f"  Run train.py for this cell first."
            )

        df = pd.read_csv(metrics_path)
        if col not in df.columns:
            raise KeyError(
                f"Metric '{col}' not in {metrics_path}. "
                f"Available: {[c for c in df.columns if c not in ('resolution','horizon')]}"
            )

        metric_values.append(float(df[col].iloc[0]))

    return metric_values


def save_results(config, cells, metric_name, metrics, fit):
    """Write fit_params.json and fit_cells.csv into {base_dir}/{dataset}/results/{name}/."""
    out_dir = os.path.join(
        config["base_dir"], config["dataset"], "results", config["name"]
    )
    os.makedirs(out_dir, exist_ok=True)

    # ── fit_cells.csv — input cells and their metric values ───────────────────
    cells_df = pd.DataFrame({
        "resolution": [c[0] for c in cells],
        "horizon":    [c[1] for c in cells],
        metric_name.upper(): metrics,
    })
    cells_path = os.path.join(out_dir, "fit_cells.csv")
    cells_df.to_csv(cells_path, index=False)

    # ── fit_params.csv — one row per fitted parameter ─────────────────────────
    params_df = pd.DataFrame([{
        "form":         fit["form"],
        "metric":       metric_name.upper(),
        "n_samples":    fit["n_samples"],
        "C":            fit["C"],
        "log_C":        fit["log_C"],
        "alpha":        fit["alpha"],
        "beta":         fit["beta"],
        "se_log_C":     fit["se_log_C"],
        "se_alpha":     fit["se_alpha"],
        "se_beta":      fit["se_beta"],
        "ci_alpha_lo":  fit["ci_alpha"][0],
        "ci_alpha_hi":  fit["ci_alpha"][1],
        "ci_beta_lo":   fit["ci_beta"][0],
        "ci_beta_hi":   fit["ci_beta"][1],
        "t_stat_alpha": fit["t_stat_alpha"],
        "t_stat_beta":  fit["t_stat_beta"],
        "p_alpha":      fit["p_alpha"],
        "p_beta":       fit["p_beta"],
        "r2":           fit["r2"],
        "confidence":   fit["confidence"],
    }])
    params_path = os.path.join(out_dir, "fit_params.csv")
    params_df.to_csv(params_path, index=False)

    print(f"\n  Results saved to: {out_dir}/")
    print(f"    fit_cells.csv  — input cells and {metric_name.upper()} values")
    print(f"    fit_params.csv — fitted C, alpha, beta, SE, CI, R²")


def get_fit_surface(config, cells, metric_name):
    metrics = get_metric_values(config, cells, metric_name)

    print(f"\n  Cells and {metric_name.upper()} values:")
    for (res, hor), val in zip(cells, metrics):
        print(f"    {res}  h={hor:>2d}  →  {val:.6f}")

    fit = fit_surface(cells, metric_values=metrics)
    print_fit_summary(fit)
    save_results(config, cells, metric_name, metrics, fit)

    return fit


def _parse_cells(cell_strings):
    """Convert ['4h:3', '24h:12', ...] into [('4h', 3), ('24h', 12), ...]."""
    cells = []
    for s in cell_strings:
        parts = s.split(":")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(
                f"Invalid cell '{s}' — expected resolution:horizon (e.g. 4h:3)"
            )
        cells.append((parts[0].strip(), int(parts[1].strip())))
    return cells


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit scaling law to trained cells and print surface parameters."
    )
    parser.add_argument("--config", required=True,
                        help="Path to YAML experiment config")
    parser.add_argument("--cells", required=True, nargs="+",
                        metavar="RES:HOR",
                        help="Trained cells as resolution:horizon pairs "
                             "(e.g. 4h:3 4h:12 24h:3 24h:12)")
    parser.add_argument("--metric", default="RMSE",
                        help="Error metric to fit on (default: RMSE). "
                             "Choices: RMSE MAE MSE")
    args = parser.parse_args()

    config = load_config(args.config)
    cells  = _parse_cells(args.cells)

    get_fit_surface(config, cells, args.metric)
