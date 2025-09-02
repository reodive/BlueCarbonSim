from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

results_dir = Path(__file__).resolve().parent.parent / "results"
plant_data = {}

# 優先: summary_totals.csv（mgC）の吸収量を使用。無ければ result_* をフォールバック。
summary = results_dir / "summary_totals.csv"
if summary.exists():
    df = pd.read_csv(summary)
    col = 'total_absorbed_mgC' if 'total_absorbed_mgC' in df.columns else ('total_absorbed' if 'total_absorbed' in df.columns else None)
    if col:
        for _, row in df.iterrows():
            plant_data[str(row['species'])] = float(row[col])
else:
    # フォールバック: mgCファイル優先、さらに従来の result_*.csv
    for filepath in results_dir.glob("result_*_mgC.csv"):
        df = pd.read_csv(filepath)
        if 'total_absorbed_mgC' in df.columns:
            total = float(df['total_absorbed_mgC'].iloc[-1]) if len(df) > 1 else float(df['total_absorbed_mgC'].iloc[0])
            name = filepath.stem[len('result_'):-4] if filepath.stem.endswith('_mgC') else filepath.stem[len('result_'):]
            plant_data[name] = total
    if not plant_data:
        for filepath in results_dir.glob("result_*.csv"):
            df = pd.read_csv(filepath)
            if 'total_absorbed' in df.columns:
                total = float(df['total_absorbed'].iloc[-1]) if len(df) > 1 else float(df['total_absorbed'].iloc[0])
                plant_name = filepath.stem[len('result_'):]
                plant_data[plant_name] = total

# 描画（保存）
plt.figure(figsize=(10, 6))
plt.bar(plant_data.keys(), plant_data.values())
plt.xticks(rotation=45, ha='right')
plt.xlabel("Plant Species")
plt.ylabel("Total Absorbed CO₂ [mgC]")
plt.title("Total CO₂ Absorbed by Each Plant")
plt.tight_layout()
out = results_dir / "total_absorbed_bars.png"
plt.savefig(out)
plt.close()
