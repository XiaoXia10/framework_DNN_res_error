#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from os.path import join, dirname, abspath
import numpy as np

def compile_metrics():
    
    loader = "test"
    forecasts = [12,6,3]
    frequence = ["24h","12h","8h","4h","1h"]
    current_dir = dirname(abspath(__file__))
    base_dir = dirname(current_dir)

    name = join(base_dir, "data", "autoencoder_model_metrics.csv")
    
    f = open(name, 'w')
    f.write("loader, data frequence, Forecast length, RMSE, MAE, NSE, KEG\n" )

    for freq in frequence:
        
        data_dir = join(base_dir, "data", freq)
        
        # df_path = "data/betteraz_"+str(timestep)+"_sims.csv"
        
        for forecast in forecasts:
            
            output_dir = join(data_dir, "timestep"+str(forecast)+str(forecast)+str(forecast))
            metrics = pd.read_csv(join(output_dir, "metric.csv"))
            
            rmse = metrics["RMSE (l/s)"].values[0]
            mae = metrics[" MAE (l/s)"].values[0]
            nse = metrics[" NSE"].values[0]
            kge = metrics[" KGE "].values[0]
            
            f.write(str(loader)+",")
            f.write(str(freq)+",")
            f.write(str(forecast)+",")
            f.write(str(rmse)+",")
            f.write(str(mae)+",")
            f.write(str(nse)+",")
            f.write(str(kge)+",")
            f.write("\n") 
        
            print(loader, "\n")
            print("RMSE:")
            print(rmse)
            print("MAE:")
            print(mae)
            print("NSE:")
            print(nse)
            print("KGE:")
            print(kge, "\n")
            


    f.close()

compile_metrics()