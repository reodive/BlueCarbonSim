from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

results_dir = Path(__file__).resolve().parent.parent / "results"
plant_data = {}

for filepath in results_dir.glob("result_*.csv"):
    df = pd.read_csv(filepath)

    # カラム名確認（デバッグ用）
    print(f"{filepath.name} → カラム: {df.columns.tolist()}")

    # カラム名が合致すれば吸収量を取得
    if 'absorbed' in df.columns:
        total = df['absorbed'].sum()
    elif 'total_absorbed' in df.columns:
        total = df['total_absorbed'].iloc[-1]
    elif 'Absorbed_CO2' in df.columns:
        total = df['Absorbed_CO2'].sum()
    else:
        print(f"⚠️ Skipped: {filepath.name} — 適切なカラムが見つかりません")
        continue

    # 植物名を安全に取得（拡張子と先頭の 'result_' を削除）
    plant_name = filepath.stem[len('result_'):]
    plant_data[plant_name] = total

# 描画
plt.figure(figsize=(10, 6))
plt.bar(plant_data.keys(), plant_data.values())
plt.xticks(rotation=45, ha='right')
plt.xlabel("Plant Species")
plt.ylabel("Total Absorbed CO₂")
plt.title("Total CO₂ Absorbed by Each Plant")
plt.tight_layout()
plt.show()
