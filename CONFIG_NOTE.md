設定メモ（最小修正の要点）

- 光減衰・深度: Beer–Lambert を使用。`kd_m_inv = 0.8 m^-1`、`max_depth_m = 8 m`。
  - `src/environment.py:get_environmental_factors` にて既定値を適用。
- 境界条件: 開境界（外向き到達で流出）を導入。
  - 左端=河川（低塩・高栄養・入流源）、右端=外洋（高塩・流出）としてコメント明示。
  - 実装: `src/particle_flow.py:diffuse_particles`（流出時に粒子を削除し質量計上）。
- 栄養塩: 季節正弦 + パルス（出水）を 0..1 でクリップ。
  - 実装: `src/environment.py:get_environmental_factors`（pulse_period=90, width=3, amp=0.3）。
