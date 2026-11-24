"""Generates the SYNTHETIC dataset street_segments.csv for PathWise v1.
Synthetic: no real NYC data yet. Values are drawn from three made-up street
types (busy/bright, quiet residential, dark/high-incident) with random noise."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
types = {
    # n, crimes/mo, streetlights/100m, night foot traffic/hr, open businesses at night, m to subway
    "busy":  (130, 2.0, 6.0, 220, 8, 250),
    "quiet": (110, 1.2, 3.5,  40, 1, 600),
    "dark":  ( 60, 6.5, 1.5,  25, 1, 750),
}
rows = []
for t, (n, crime, light, foot, biz, subway) in types.items():
    for _ in range(n):
        rows.append([
            max(0, round(rng.normal(crime, crime * 0.35), 1)),
            max(0, round(rng.normal(light, 1.0), 1)),
            max(0, round(rng.normal(foot, foot * 0.3))),
            max(0, round(rng.normal(biz, 2 if t == "busy" else 1))),
            max(50, round(rng.normal(subway, 150))),
        ])
df = pd.DataFrame(rows, columns=["night_crimes_per_month", "streetlights_per_100m",
                                 "night_foot_traffic_per_hr", "open_businesses_night",
                                 "dist_to_subway_m"])
df = df.sample(frac=1, random_state=7).reset_index(drop=True)
df.insert(0, "segment_id", range(1001, 1001 + len(df)))
df.to_csv("street_segments.csv", index=False)
print(df.shape)
