"""PathWise v1: cluster NYC street segments into safety zones (k-means).
Loads street_segments.csv, fits v1, prints the fit evaluation, returns a
routing decision for a new segment, and quantifies baseline vs v1."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

df = pd.read_csv("street_segments.csv")
features = ["night_crimes_per_month", "streetlights_per_100m",
            "night_foot_traffic_per_hr", "open_businesses_night", "dist_to_subway_m"]
print("Rows, columns:", df.shape)

# 1. Split 70/30
train, test = train_test_split(df, test_size=0.3, random_state=42)
print("Training segments:", len(train), " Test segments:", len(test))

# 2. Scale on training only
scaler = StandardScaler()
X_train = scaler.fit_transform(train[features])
X_test = scaler.transform(test[features])

# 3. Elbow: training spread and test spread from trained centers
ks = range(1, 9)
train_spread, test_spread = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X_train)
    train_spread.append(km.inertia_)
    test_spread.append(-km.score(X_test))
    print(f"k = {k}  training spread = {km.inertia_:.1f}  test spread = {test_spread[-1]:.1f}")

plt.figure()
plt.plot(ks, train_spread, marker="o", label=f"Training segments ({len(train)})")
plt.plot(ks, test_spread, marker="o", label=f"Test segments ({len(test)})")
plt.xlabel("Number of groups (k)")
plt.ylabel("Within-group spread")
plt.title("Elbow plot: training vs test")
plt.legend()
plt.savefig("elbow.png")

# Fit v1 with the chosen k
K = 3
model = KMeans(n_clusters=K, n_init=10, random_state=42).fit(X_train)
centers = pd.DataFrame(scaler.inverse_transform(model.cluster_centers_), columns=features)
centers["segments_in_training"] = pd.Series(model.labels_).value_counts().sort_index()
print("\nGroup centers (average street segment of each group):")
print(centers.round(1))

# Name groups by crime level so names stay correct whatever the label order
order = centers["night_crimes_per_month"].sort_values().index
names = {}
names[order[0]] = "Quiet residential"
names[order[1]] = "Busy & bright"
names[order[2]] = "Dark & high-risk"
# if the lowest-crime group is actually the busy one, swap names using foot traffic
if centers.loc[order[0], "night_foot_traffic_per_hr"] > centers.loc[order[1], "night_foot_traffic_per_hr"]:
    names[order[0]], names[order[1]] = "Busy & bright", "Quiet residential"
avoid = {c: (names[c] == "Dark & high-risk") for c in names}
print("\nGroup names:", names)

# 4. Decision rule
def route_decision(night_crimes_per_month, streetlights_per_100m, night_foot_traffic_per_hr,
                   open_businesses_night, dist_to_subway_m):
    """Returns the safety zone of one street segment and whether PathWise routes around it at night."""
    seg = pd.DataFrame([[night_crimes_per_month, streetlights_per_100m, night_foot_traffic_per_hr,
                         open_businesses_night, dist_to_subway_m]], columns=features)
    c = model.predict(scaler.transform(seg))[0]
    return names[c], ("AVOID at night" if avoid[c] else "OK to route")

zone, action = route_decision(5.5, 1.8, 30, 0, 700)
print("\nNew segment (5.5 crimes/mo, 1.8 lights/100m, 30 people/hr, 0 open businesses, 700 m to subway):")
print("Zone:", zone, " Decision:", action)

# 5. Quantify, baseline vs v1 (ASSUMPTIONS, not real figures)
TRIPS = 10_000          # assumed night walking trips through PathWise per month
SEGS_PER_TRIP = 8       # assumed street segments per trip
RISK_PER_CRIME = 1/20_000  # assumed chance a pass-through is affected, per monthly crime on the segment
df["zone"] = model.predict(scaler.transform(df[features]))
risk = df["night_crimes_per_month"] * RISK_PER_CRIME
baseline_incidents = TRIPS * SEGS_PER_TRIP * risk.mean()                      # shortest path: any segment
v1_incidents = TRIPS * SEGS_PER_TRIP * risk[~df["zone"].map(avoid)].mean()     # avoids high-risk zone
print("\nAll segments by zone:")
print(df.groupby(df["zone"].map(names))["night_crimes_per_month"].agg(["count", "mean"]).round(1))
print(f"\nExpected incidents per {TRIPS:,} night trips - baseline: {baseline_incidents:.1f}")
print(f"Expected incidents per {TRIPS:,} night trips - v1:       {v1_incidents:.1f}")
print(f"Reduction: {(1 - v1_incidents/baseline_incidents):.0%}")
