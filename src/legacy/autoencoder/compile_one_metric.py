#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 21:05:08 2026

@author: xl3138
"""
import pandas as pd
from os.path import join, dirname, abspath


def compile_metrics():
    forecasts = [3, 6, 12]
    frequence = ["1h", "4h", "8h", "12h", "24h"]

    current_dir = dirname(abspath(__file__))
    base_dir = dirname(current_dir)

    output_file = join(base_dir, "data", "GRU_mae_metric.csv")

    with open(output_file, "w") as f:
        # header
        f.write(",".join(["frequency"] + [str(x) for x in forecasts]) + "\n")

        for freq in frequence:
            row = [freq]

            data_dir = join(base_dir, "data", freq)

            for forecast in forecasts:
                output_dir = join(
                    data_dir,
                    "timestep" + str(forecast) * 3
                )

                metrics = pd.read_csv(join(output_dir, "metric.csv"))
                rmse = metrics[" MAE (l/s)"].values[0]

                row.append(str(rmse))

            f.write(",".join(row) + "\n")


compile_metrics()