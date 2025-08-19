import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

heatmap_dir = "results"
for filename in os.listdir(heatmap_dir):
    if filename.startswith("heatmap_") and filename.endswith(".csv"):
        filepath = os.path.join(heatmap_dir, filename)
        df = pd.read_csv(filepath)

        grid_size = int(np.sqrt(len(df)))  # 正方形前提
        heatmap = np.zeros((grid_size, grid_size))

        for _, row in df.iterrows():
            x, y, count = int(row['X']), int(row['Y']), row['Absorption_Count']
            heatmap[y, x] = count

        plt.figure(figsize=(6, 5))
        plt.imshow(heatmap, cmap="YlGnBu", origin='lower')
        plt.colorbar(label='Absorption Count')
        plt.title(f"Heatmap - {filename.replace('heatmap_', '').replace('.csv', '')}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.tight_layout()
        plt.savefig(os.path.join("results", filename.replace(".csv", ".png")))
        plt.close()